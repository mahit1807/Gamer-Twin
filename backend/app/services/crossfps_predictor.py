"""
CrossFPS PyTorch Predictor Service for GAMER-TWIN.
Loads genuine trained model checkpoints (transformer_crossfps_best.pt, lstm_crossfps_best.pt),
runs forward passes on actual 10-DoF sequences, and computes exact counterfactual probability shifts.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from src.crossfps_pipeline import (
    CrossFPSTransformerModel,
    CrossFPSTransformerSSL,
    CrossFPSLSTMBaseline,
    CONTINUOUS_CHANNELS,
    DISCRETE_CHANNELS,
    CHANNEL_LABELS
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CHECKPOINTS_DIR = WORKSPACE_ROOT / "outputs" / "checkpoints"
SPLITS_FILE = WORKSPACE_ROOT / "outputs" / "processed" / "crossfps_splits.npz"

ACTION_NAMES = [
    "Fire (RT)",
    "Scope / Aim (LT)",
    "Melee (R3)",
    "Jump (A)",
    "Reload (X)",
    "Switch Weapon (Y)"
]

_LOADED_MODELS: Dict[str, torch.nn.Module] = {}
_SPLIT_DATA: Optional[Dict[str, np.ndarray]] = None


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_crossfps_model(model_name: str) -> torch.nn.Module:
    """
    Loads and validates the genuine PyTorch checkpoint for the requested model.
    """
    global _LOADED_MODELS
    name_lower = model_name.lower()
    if "ssl" in name_lower:
        clean_name = "transformer_ssl"
    elif "transformer" in name_lower:
        clean_name = "transformer"
    else:
        clean_name = "lstm"

    if clean_name in _LOADED_MODELS:
        return _LOADED_MODELS[clean_name]

    ckpt_file = f"{clean_name}_crossfps_best.pt"
    ckpt_path = CHECKPOINTS_DIR / ckpt_file
    if not ckpt_path.exists():
        if clean_name == "transformer_ssl":
            raise FileNotFoundError(
                "Not trained yet — self-supervised pretraining and next-action fine-tuning are required."
            )
        raise FileNotFoundError(
            f"Checkpoint '{ckpt_file}' not found in {CHECKPOINTS_DIR}. "
            "Please ensure train_crossfps.py has produced the weights."
        )

    device = get_device()
    if clean_name == "transformer_ssl":
        model = CrossFPSTransformerSSL(input_dim=10, d_model=64, nhead=4, num_layers=2)
    elif clean_name == "transformer":
        model = CrossFPSTransformerModel(input_dim=10, d_model=64, nhead=4, num_layers=2)
    else:
        model = CrossFPSLSTMBaseline(input_dim=10, hidden_dim=64, num_layers=2)

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    _LOADED_MODELS[clean_name] = model
    return model


def get_test_splits() -> Dict[str, np.ndarray]:
    """
    Loads the verified held-out test partition from crossfps_splits.npz.
    """
    global _SPLIT_DATA
    if _SPLIT_DATA is not None:
        return _SPLIT_DATA

    if not SPLITS_FILE.exists():
        raise FileNotFoundError(f"Processed CrossFPS splits file not found at {SPLITS_FILE}.")

    loaded = np.load(SPLITS_FILE, allow_pickle=True)
    _SPLIT_DATA = {
        "X_test": loaded["X_test"],
        "y_cont_test": loaded["yc_test"],
        "y_disc_test": loaded["yd_test"],
        "clip_ids_test": loaded["clip_ids"],
        "games_test": loaded["games"]
    }
    return _SPLIT_DATA


def predict_crossfps_sequence(
    sequence: np.ndarray,
    model_name: str = "transformer",
    ground_truth_cont: Optional[np.ndarray] = None,
    ground_truth_disc: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Performs live inference using the loaded model checkpoint on a 10-DoF sequence [16, 10].
    Returns continuous joystick regression and discrete button probabilities.
    """
    model = load_crossfps_model(model_name)
    device = get_device()

    if sequence.ndim == 2:
        seq_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
    else:
        seq_tensor = torch.tensor(sequence, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred_cont, pred_disc_logits = model(seq_tensor)
        pred_disc_probs = torch.sigmoid(pred_disc_logits).squeeze(0).cpu().numpy()
        pred_cont_vals = pred_cont.squeeze(0).cpu().numpy()

    # Ranked discrete action probabilities
    ranked = []
    prob_dict = {}
    for idx, (action_code, action_name) in enumerate(zip(DISCRETE_CHANNELS, ACTION_NAMES)):
        p = float(pred_disc_probs[idx])
        prob_dict[action_code] = round(p, 4)
        prob_dict[action_name] = round(p, 4)
        ranked.append({
            "rank": 0,  # will assign below
            "channel": action_code,
            "action_name": action_name,
            "probability": round(p, 4),
            "percentage": round(p * 100, 2)
        })

    ranked.sort(key=lambda x: x["probability"], reverse=True)
    for i, item in enumerate(ranked):
        item["rank"] = i + 1

    top_disc = ranked[0]
    active_pred = [item["action_name"] for item in ranked if item["probability"] >= 0.25]
    if active_pred:
        predicted_action = " + ".join(active_pred)
    else:
        predicted_action = "Neutral Steering / Movement"
    confidence = top_disc["probability"]

    # Ground truth comparison
    ground_truth_action = None
    matches_ground_truth = None
    if ground_truth_disc is not None:
        gt_active = [ACTION_NAMES[i] for i, v in enumerate(ground_truth_disc) if v > 0.5]
        ground_truth_action = ", ".join(gt_active) if gt_active else "Neutral Steering / Movement"
        if len(active_pred) == 0 and len(gt_active) == 0:
            matches_ground_truth = True
        elif any(a in gt_active for a in active_pred):
            matches_ground_truth = True
        else:
            matches_ground_truth = False

    # Continuous predictions: LX, LY, RX, RY
    lx = round(float(pred_cont_vals[0]), 3)
    ly = round(float(pred_cont_vals[1]), 3)
    rx = round(float(pred_cont_vals[2]), 3)
    ry = round(float(pred_cont_vals[3]), 3)

    move_parts = []
    if ly >= 0.2: move_parts.append(f"Forward (+{ly:.2f})")
    elif ly <= -0.2: move_parts.append(f"Backward ({ly:.2f})")
    if lx >= 0.2: move_parts.append(f"Strafe Right (+{lx:.2f})")
    elif lx <= -0.2: move_parts.append(f"Strafe Left ({lx:.2f})")
    move_summary = ", ".join(move_parts) if move_parts else "Stationary / Centered"

    cam_parts = []
    if rx >= 0.2: cam_parts.append(f"Turn Right (+{rx:.2f})")
    elif rx <= -0.2: cam_parts.append(f"Turn Left ({rx:.2f})")
    if ry >= 0.2: cam_parts.append(f"Pitch Up (+{ry:.2f})")
    elif ry <= -0.2: cam_parts.append(f"Pitch Down ({ry:.2f})")
    cam_summary = ", ".join(cam_parts) if cam_parts else "Steady Camera"

    continuous_pred = {
        "strafe_lx": lx,
        "move_ly": ly,
        "turn_yaw_rx": rx,
        "look_pitch_ry": ry,
        "movement_summary": move_summary,
        "camera_summary": cam_summary
    }

    if "ssl" in model_name.lower():
        display_model = "Self-Supervised Transformer"
    elif "transformer" in model_name.lower():
        display_model = "Transformer Encoder"
    else:
        display_model = "LSTM Baseline"

    return {
        "model_name": display_model,
        "predicted_action": predicted_action,
        "predicted_discrete_actions": active_pred if active_pred else ["Neutral Steering / Movement"],
        "confidence": round(float(confidence), 4),
        "continuous_prediction": continuous_pred,
        "ranked_probabilities": ranked,
        "action_probabilities": prob_dict,
        "ground_truth_action": ground_truth_action,
        "matches_ground_truth": matches_ground_truth
    }


def evaluate_what_if_scenario(
    sequence: np.ndarray,
    model_name: str = "transformer",
    condition_overrides: Optional[Dict[str, Any]] = None,
    modified_scope: Optional[bool] = None,
    modified_jump: Optional[bool] = None,
    turn_intensity: Optional[float] = None
) -> Dict[str, Any]:
    """
    Applies real counterfactual modifications to the 10-DoF sequence and feeds it
    directly through the loaded PyTorch model to compute exact probability shifts.
    """
    base_res = predict_crossfps_sequence(sequence, model_name=model_name)

    what_if_seq = sequence.copy()
    modifications = {}

    if condition_overrides:
        if condition_overrides.get("aim_down_sights") is True:
            what_if_seq[-4:, 5] = 1.0
            modifications["scope"] = "Aim Down Sights (LT = 1.0)"
        elif condition_overrides.get("aim_down_sights") is False:
            what_if_seq[-4:, 5] = 0.0
            modifications["scope"] = "Unscoped (LT = 0.0)"

        if condition_overrides.get("fire") is True:
            what_if_seq[-4:, 4] = 1.0
            modifications["fire"] = "Fire Trigger (RT = 1.0)"

        if condition_overrides.get("jump") is True:
            what_if_seq[-4:, 7] = 1.0
            modifications["jump"] = "Jump Impulse (A = 1.0)"

        if condition_overrides.get("reload") is True:
            what_if_seq[-4:, 8] = 1.0
            modifications["reload"] = "Reload (X = 1.0)"

        if condition_overrides.get("switch_weapon") is True:
            what_if_seq[-4:, 9] = 1.0
            modifications["switch"] = "Switch Weapon (Y = 1.0)"

        if condition_overrides.get("melee") is True:
            what_if_seq[-4:, 6] = 1.0
            modifications["melee"] = "Melee Attack (R3 = 1.0)"

        mov = condition_overrides.get("movement")
        if mov == "forward":
            what_if_seq[-4:, 1] = 0.85
            modifications["movement"] = "Move Forward (LY = +0.85)"
        elif mov == "backward":
            what_if_seq[-4:, 1] = -0.85
            modifications["movement"] = "Move Backward (LY = -0.85)"
        elif mov == "strafe_left":
            what_if_seq[-4:, 0] = -0.85
            modifications["movement"] = "Strafe Left (LX = -0.85)"
        elif mov == "strafe_right":
            what_if_seq[-4:, 0] = 0.85
            modifications["movement"] = "Strafe Right (LX = +0.85)"
        elif mov == "stationary":
            what_if_seq[-4:, 0] = 0.0
            what_if_seq[-4:, 1] = 0.0
            modifications["movement"] = "Stationary Posture (LX=0, LY=0)"

        cam = condition_overrides.get("camera")
        if cam == "turn_left":
            what_if_seq[-4:, 2] = -0.75
            modifications["camera"] = "Turn Camera Left (RX = -0.75)"
        elif cam == "turn_right":
            what_if_seq[-4:, 2] = 0.75
            modifications["camera"] = "Turn Camera Right (RX = +0.75)"
        elif cam == "look_up":
            what_if_seq[-4:, 3] = 0.65
            modifications["camera"] = "Look Up (RY = +0.65)"
        elif cam == "look_down":
            what_if_seq[-4:, 3] = -0.65
            modifications["camera"] = "Look Down (RY = -0.65)"
        elif cam == "steady":
            what_if_seq[-4:, 2] = 0.0
            what_if_seq[-4:, 3] = 0.0
            modifications["camera"] = "Steady Camera (RX=0, RY=0)"

    # Legacy individual overrides if not already covered
    if modified_scope is not None and "scope" not in modifications:
        val = 1.0 if modified_scope else 0.0
        what_if_seq[-4:, 5] = val
        modifications["scope"] = "Aim Down Sights (LT = 1.0)" if modified_scope else "Unscoped (LT = 0.0)"

    if modified_jump is not None and "jump" not in modifications:
        val = 1.0 if modified_jump else 0.0
        what_if_seq[-4:, 7] = val
        modifications["jump"] = "Airborne Impulse (A = 1.0)" if modified_jump else "Grounded (A = 0.0)"

    if turn_intensity is not None and "camera" not in modifications:
        what_if_seq[-4:, 2] = float(turn_intensity)
        modifications["camera"] = f"Yaw Angular Velocity (RX = {turn_intensity:+.2f})"

    hyp_res = predict_crossfps_sequence(what_if_seq, model_name=model_name)

    # Compute genuine probability shifts
    shifts = {}
    for code, name in zip(DISCRETE_CHANNELS, ACTION_NAMES):
        p_base = base_res["action_probabilities"].get(name, 0.0)
        p_hyp = hyp_res["action_probabilities"].get(name, 0.0)
        shifts[name] = round((p_hyp - p_base) * 100, 2)

    # Continuous diffs
    cont_base = base_res["continuous_prediction"]
    cont_hyp = hyp_res["continuous_prediction"]
    continuous_shifts = {
        "strafe_lx_diff": round(float(cont_hyp["strafe_lx"] - cont_base["strafe_lx"]), 3),
        "move_ly_diff": round(float(cont_hyp["move_ly"] - cont_base["move_ly"]), 3),
        "turn_yaw_rx_diff": round(float(cont_hyp["turn_yaw_rx"] - cont_base["turn_yaw_rx"]), 3),
        "look_pitch_ry_diff": round(float(cont_hyp["look_pitch_ry"] - cont_base["look_pitch_ry"]), 3)
    }

    return {
        "modifications": modifications,
        "baseline_prediction": {
            "predicted_action": base_res["predicted_action"],
            "confidence": base_res["confidence"],
            "confidence_pct": round(base_res["confidence"] * 100, 2),
            "continuous": base_res["continuous_prediction"],
            "discrete_actions": base_res.get("predicted_discrete_actions", [])
        },
        "what_if_prediction": {
            "predicted_action": hyp_res["predicted_action"],
            "confidence": hyp_res["confidence"],
            "confidence_pct": round(hyp_res["confidence"] * 100, 2),
            "continuous": hyp_res["continuous_prediction"],
            "discrete_actions": hyp_res.get("predicted_discrete_actions", [])
        },
        "action_changed": base_res["predicted_action"] != hyp_res["predicted_action"],
        "probability_shifts_pct": shifts,
        "continuous_shifts": continuous_shifts,
        "ranked_hypothetical_probabilities": hyp_res["ranked_probabilities"]
    }
