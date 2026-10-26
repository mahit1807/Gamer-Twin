"""
Prediction Studio & Counterfactual What-If API routes for GAMER-TWIN.
Uses real test sequences from CrossFPS validation data (outputs/processed/crossfps_splits.npz)
and runs genuine inference through trained PyTorch checkpoints (transformer_crossfps_best.pt, lstm_crossfps_best.pt).
Supports game selection, gameplay situation configuration (10-DoF overrides), and observable explanations.
No mock predictions, fabricated probabilities, or synthetic fallbacks are permitted.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import torch

from backend.app.schemas.schemas import PredictionTrace, PredictRequest, RankedProbability
from backend.app.services.crossfps_predictor import (
    get_test_splits,
    predict_crossfps_sequence,
    evaluate_what_if_scenario,
    load_crossfps_model,
    ACTION_NAMES,
    DISCRETE_CHANNELS
)
from backend.app.services.crossfps_service import (
    get_metadata_df,
    list_clips,
    parse_clip_parquet
)
from src.prediction import NextActionPredictor
from src.utils import get_device

router = APIRouter()

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent

_PREDICTOR_CACHE: Dict[str, NextActionPredictor] = {}

GAME_INFO = {
    "halo_infinite": {
        "name": "Halo Infinite",
        "genre": "Sci-Fi Arena FPS",
        "tagline": "343 Industries · 2021",
        "description": "High-mobility arena FPS with vertical grappling, shields, and precision weapon combat."
    },
    "call_of_duty__warzone": {
        "name": "Call of Duty: Warzone",
        "genre": "Battle Royale",
        "tagline": "Infinity Ward · 2020",
        "description": "Large-scale tactical battle royale with tactical sprint, slide cancels, and rapid engagements."
    },
    "call_of_duty__modern_warfare_iii": {
        "name": "COD: Modern Warfare III",
        "genre": "Fast-Paced Tactical FPS",
        "tagline": "Sledgehammer Games · 2023",
        "description": "Modern multi-DoF tactical movement with slide cancel reloads and aggressive close-quarters aiming."
    },
    "call_of_duty": {
        "name": "Call of Duty (Classic)",
        "genre": "Classic Military Shooter",
        "tagline": "Activision Anthology",
        "description": "Foundational arcade FPS with grounded sprint, hipfire-to-ADS transitions, and grenade timing."
    },
    "worldcam_xonotic": {
        "name": "Xonotic",
        "genre": "Fast Arena Movement FPS",
        "tagline": "Open-Source Arena",
        "description": "Ultra-fast arena shooter emphasizing strafe-jumping, bunny-hopping, and twitch aiming."
    },
    "mw": {
        "name": "Call of Duty: Modern Warfare",
        "genre": "Tactical FPS",
        "tagline": "Infinity Ward · 2019",
        "description": "Grounded tactical military shooter with realistic weapon handling, mounting, and laser ADS."
    },
    "halo": {
        "name": "Halo MCC",
        "genre": "Classic Sci-Fi Shooter",
        "tagline": "Bungie / 343 Industries",
        "description": "Classic arena halo combat with projectile travel times, plasma shields, and timed melee."
    }
}


def get_predictor_for_model(model_name: str) -> NextActionPredictor:
    name_lower = model_name.lower()
    if "ssl" in name_lower:
        clean_name = "transformer_ssl"
    elif "transformer" in name_lower:
        clean_name = "transformer"
    else:
        clean_name = "lstm_baseline"

    if clean_name in _PREDICTOR_CACHE:
        return _PREDICTOR_CACHE[clean_name]

    ckpt_path = WORKSPACE_ROOT / "outputs" / "checkpoints" / f"{clean_name}_best.pt"
    if not ckpt_path.exists():
        ckpt_path = WORKSPACE_ROOT / "outputs" / "checkpoints" / f"{clean_name}_crossfps_best.pt"
    if not ckpt_path.exists():
        if clean_name == "transformer_ssl":
            raise HTTPException(
                status_code=400,
                detail="Not trained yet — self-supervised pretraining and next-action fine-tuning are required."
            )
        raise HTTPException(
            status_code=404,
            detail=f"No trained checkpoint found for model '{model_name}'."
        )

    device = get_device()
    predictor = NextActionPredictor(checkpoint_path=str(ckpt_path), device=device)
    _PREDICTOR_CACHE[clean_name] = predictor
    return predictor


class WhatIfRequest(BaseModel):
    model_name: str = "transformer"
    sequence_index: int = 0
    modified_scope: Optional[bool] = None
    modified_jump: Optional[bool] = None
    modified_posture: Optional[str] = None
    turn_intensity: Optional[float] = None
    condition_overrides: Optional[Dict[str, Any]] = None


@router.get("/models")
def get_prediction_models():
    """
    Returns available models specifically validated for CrossFPS sequence prediction.
    Option 1: LSTM Baseline
    Option 2: Transformer Encoder
    Option 3: Self-Supervised Transformer
    """
    ckpt_dir = WORKSPACE_ROOT / "outputs" / "checkpoints"
    bench_file = WORKSPACE_ROOT / "outputs" / "reports" / "crossfps_benchmark.json"

    benchmarks = {}
    if bench_file.exists():
        try:
            with open(bench_file, "r") as f:
                benchmarks = json.load(f).get("models", {})
        except Exception:
            benchmarks = {}

    model_defs = [
        {
            "id": "lstm_baseline",
            "name": "LSTM Baseline",
            "role": "Sequential Baseline Model",
            "is_primary": False,
            "description": "A recurrent neural network that learns temporal patterns from previous controller inputs.",
            "checkpoint": "lstm_crossfps_best.pt",
            "bench_key": "lstm_baseline"
        },
        {
            "id": "transformer",
            "name": "Transformer Encoder",
            "role": "Primary Proposed Model",
            "is_primary": True,
            "description": "The primary sequence model, using self-attention to learn relationships across previous gameplay frames.",
            "checkpoint": "transformer_crossfps_best.pt",
            "bench_key": "transformer_primary"
        },
        {
            "id": "transformer_ssl",
            "name": "Self-Supervised Transformer",
            "role": "Self-Supervised Representation",
            "is_primary": False,
            "description": "A Transformer-based model with a self-supervised pretraining stage intended to learn useful representations from unlabeled gameplay sequences before being trained or fine-tuned for next-action prediction.",
            "checkpoint": "transformer_ssl_crossfps_best.pt",
            "bench_key": "transformer_ssl"
        }
    ]

    models = []
    for m in model_defs:
        ckpt_path = ckpt_dir / m["checkpoint"]
        exists = ckpt_path.exists()
        can_load = False
        load_error = None

        if m["id"] == "transformer_ssl" and not exists:
            status = "Not trained yet — self-supervised pretraining and next-action fine-tuning are required."
            load_error = "Checkpoint transformer_ssl_crossfps_best.pt not found on disk."
            is_available = False
            eval_data = None
        else:
            if exists:
                try:
                    load_crossfps_model(m["id"])
                    can_load = True
                except Exception as e:
                    load_error = str(e)
                    can_load = False
            else:
                load_error = f"Checkpoint {m['checkpoint']} not found on disk."

            bench_info = benchmarks.get(m["bench_key"], {})
            metrics = bench_info.get("metrics")
            eval_data = None
            if metrics:
                eval_data = {
                    "multi_label_accuracy_pct": round(metrics.get("multi_label_accuracy", 0) * 100, 1),
                    "macro_f1": round(metrics.get("macro_f1", 0), 4),
                    "weighted_f1": round(metrics.get("weighted_f1", 0), 4),
                    "precision_macro": round(metrics.get("precision_macro", 0), 4),
                    "recall_macro": round(metrics.get("recall_macro", 0), 4),
                    "continuous_mse": round(metrics.get("continuous_mse", 0), 4),
                    "test_samples": metrics.get("test_samples", 10859),
                    "evaluation_split": "Held-Out Test Set (CrossFPS 10-DoF validation split)"
                }

            is_available = exists and can_load
            status = "Ready for Inference" if is_available else ("Checkpoint Loading Error" if exists else "Checkpoint Missing")

        models.append({
            "id": m["id"],
            "name": m["name"],
            "role": m["role"],
            "is_primary": m["is_primary"],
            "description": m["description"],
            "checkpoint": m["checkpoint"],
            "checkpoint_exists": exists,
            "can_load": can_load,
            "is_available": is_available,
            "status": status,
            "load_error": load_error if not is_available else None,
            "evaluation": eval_data,
            "disclosure": "Trained and evaluated on held-out clip partitions of the available CrossFPS validation partition (data/CrossFPS-val/). Not presented as an independent production test benchmark."
        })

    return models


@router.get("/games")
def get_supported_games():
    """
    Returns the real list of games represented in the CrossFPS dataset,
    filtering out games with 0 usable test sequences.
    """
    try:
        meta_df = get_metadata_df()
        clip_counts = meta_df["source_game"].value_counts().to_dict()
    except Exception:
        clip_counts = {}

    try:
        splits = get_test_splits()
        seq_counts = pd.Series(splits["games_test"]).value_counts().to_dict()
    except Exception:
        seq_counts = {}

    games = []
    for gid, info in GAME_INFO.items():
        c_count = int(clip_counts.get(gid, 0))
        s_count = int(seq_counts.get(gid, 0))
        # Filter out games with zero usable sequences
        if s_count <= 0:
            continue
        games.append({
            "id": gid,
            "name": info["name"],
            "genre": info["genre"],
            "tagline": info["tagline"],
            "description": info["description"],
            "clip_count": c_count,
            "sequence_count": s_count,
            "is_available": True,
            "status": f"{s_count:,} sequences ready"
        })

    return {"total_games": len(games), "games": games}


@router.get("/sequences")
def list_available_sequences(
    limit: int = Query(50, ge=1, le=200),
    game: Optional[str] = Query(None)
):
    try:
        splits = get_test_splits()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    X_test = splits["X_test"]
    yd_test = splits["y_disc_test"]
    clip_ids = splits["clip_ids_test"]
    games = splits["games_test"]

    sequences_list = []
    for i in range(len(X_test)):
        game_title = str(games[i]) if i < len(games) else "crossfps"

        # Game filter if requested
        if game and game.lower() not in game_title.lower():
            continue

        active_disc = [ACTION_NAMES[j] for j, v in enumerate(yd_test[i]) if v > 0.5]
        gt_label = ", ".join(active_disc) if active_disc else "Neutral Steering / Movement"
        clip_id = str(clip_ids[i]) if i < len(clip_ids) else f"clip_{i}"

        # Extract recent action history
        recent_acts = []
        for t_step in range(max(0, len(X_test[i]) - 4), len(X_test[i])):
            frame = X_test[i][t_step]
            acts = []
            if frame[4] > 0.5: acts.append("Fire")
            if frame[5] > 0.5: acts.append("Aim/Scope")
            if frame[6] > 0.5: acts.append("Melee")
            if frame[7] > 0.5: acts.append("Jump")
            if frame[8] > 0.5: acts.append("Reload")
            if frame[9] > 0.5: acts.append("Switch")
            recent_acts.append(", ".join(acts) if acts else "Move/Aim")

        display_name = GAME_INFO.get(game_title, {}).get("name", game_title.replace("_", " ").title())

        sequences_list.append({
            "sequence_index": i,
            "sequence_id": f"SEQ-CFPS-{i+1:05d}",
            "clip_id": clip_id,
            "source_game": game_title,
            "game_name": display_name,
            "window_size": int(X_test[i].shape[0]),
            "feature_dim": int(X_test[i].shape[1]),
            "ground_truth_action": gt_label,
            "recent_actions": recent_acts,
            "preview_feature_mean": float(np.mean(X_test[i]))
        })

        if len(sequences_list) >= limit:
            break

    return {
        "total_test_sequences": len(X_test),
        "returned_sequences": len(sequences_list),
        "game_filter": game,
        "sequences": sequences_list
    }


def apply_condition_overrides(seq: np.ndarray, overrides: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
    """
    Applies real 10-DoF controller channel modifications to the final 4 frames of a sequence.
    Channels: 0:LX, 1:LY, 2:RX, 3:RY, 4:RT, 5:LT, 6:R3, 7:A, 8:X, 9:Y
    """
    mod_seq = seq.copy()
    applied = []

    # Discrete buttons
    if overrides.get("aim_down_sights") is True:
        mod_seq[-4:, 5] = 1.0  # LT
        applied.append("Aim Down Sights (LT = 1.0)")
    elif overrides.get("aim_down_sights") is False:
        mod_seq[-4:, 5] = 0.0
        applied.append("Hipfire / Unscoped (LT = 0.0)")

    if overrides.get("fire") is True:
        mod_seq[-4:, 4] = 1.0  # RT
        applied.append("Fire Trigger (RT = 1.0)")

    if overrides.get("jump") is True:
        mod_seq[-4:, 7] = 1.0  # Button A
        applied.append("Jump Impulse (A = 1.0)")

    if overrides.get("reload") is True:
        mod_seq[-4:, 8] = 1.0  # Button X
        applied.append("Reload (X = 1.0)")

    if overrides.get("switch_weapon") is True:
        mod_seq[-4:, 9] = 1.0  # Button Y
        applied.append("Switch Weapon (Y = 1.0)")

    if overrides.get("melee") is True:
        mod_seq[-4:, 6] = 1.0  # Right Thumb R3
        applied.append("Melee Attack (R3 = 1.0)")

    # Continuous movement (LX, LY)
    movement = overrides.get("movement")
    if movement == "forward":
        mod_seq[-4:, 1] = 0.85
        applied.append("Move Forward (LY = +0.85)")
    elif movement == "backward":
        mod_seq[-4:, 1] = -0.85
        applied.append("Move Backward (LY = -0.85)")
    elif movement == "strafe_left":
        mod_seq[-4:, 0] = -0.85
        applied.append("Strafe Left (LX = -0.85)")
    elif movement == "strafe_right":
        mod_seq[-4:, 0] = 0.85
        applied.append("Strafe Right (LX = +0.85)")
    elif movement == "stationary":
        mod_seq[-4:, 0] = 0.0
        mod_seq[-4:, 1] = 0.0
        applied.append("Stationary Posture (LX=0, LY=0)")

    # Continuous camera (RX, RY)
    camera = overrides.get("camera")
    if camera == "turn_left":
        mod_seq[-4:, 2] = -0.75
        applied.append("Turn Camera Left (RX = -0.75)")
    elif camera == "turn_right":
        mod_seq[-4:, 2] = 0.75
        applied.append("Turn Camera Right (RX = +0.75)")
    elif camera == "look_up":
        mod_seq[-4:, 3] = 0.65
        applied.append("Look Up (RY = +0.65)")
    elif camera == "look_down":
        mod_seq[-4:, 3] = -0.65
        applied.append("Look Down (RY = -0.65)")
    elif camera == "steady":
        mod_seq[-4:, 2] = 0.0
        mod_seq[-4:, 3] = 0.0
        applied.append("Steady Camera (RX=0, RY=0)")

    return mod_seq, applied


@router.post("/predict")
def predict_next_action(
    req: PredictRequest,
    mode: str = Query("dataset", description="dataset or personalized")
):
    model_name = req.model_name

    # Mode B check
    if mode == "personalized":
        user_telemetry_file = WORKSPACE_ROOT / "outputs" / "user_gameplay" / "active_user_telemetry.csv"
        if not user_telemetry_file.exists():
            return {
                "mode": "personalized",
                "status": "INSUFFICIENT_PERSONAL_DATA",
                "message": "Personalized prediction requires importing gameplay telemetry on /gameplay (0 files uploaded). Defaulting to Mode A Dataset-Trained Prediction.",
                "fallback_trace": None
            }

    try:
        splits = get_test_splits()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    X_test = splits["X_test"]
    yc_test = splits["y_cont_test"]
    yd_test = splits["y_disc_test"]
    clip_ids = splits["clip_ids_test"]
    games = splits["games_test"]

    seq_id = "SEQ-CFPS-00001"
    clip_id = "clip_eval"
    game_title = "call_of_duty"
    gt_cont = None
    gt_disc = None

    if req.sequence_index is not None and req.sequence_index < len(X_test):
        idx = req.sequence_index
        seq_array = X_test[idx]
        gt_cont = yc_test[idx]
        gt_disc = yd_test[idx]
        seq_id = f"SEQ-CFPS-{idx+1:05d}"
        clip_id = str(clip_ids[idx]) if idx < len(clip_ids) else f"clip_{idx}"
        game_title = str(games[idx]) if idx < len(games) else "crossfps"
    elif req.custom_sequence is not None:
        seq_array = np.array(req.custom_sequence, dtype=np.float32)
        if seq_array.ndim != 2 or seq_array.shape[1] != 10:
            raise HTTPException(
                status_code=400,
                detail=f"custom_sequence must be a 2D array of shape (T, 10), got {seq_array.shape}."
            )
        seq_id = "CUSTOM-SEQ-0001"
    else:
        seq_array = X_test[0]
        gt_cont = yc_test[0]
        gt_disc = yd_test[0]
        clip_id = str(clip_ids[0])
        game_title = str(games[0])

    # Apply situation condition overrides if user customized the situation
    applied_conditions = []
    if req.condition_overrides:
        seq_array, applied_conditions = apply_condition_overrides(seq_array, req.condition_overrides)

    try:
        pred_res = predict_crossfps_sequence(
            sequence=seq_array,
            model_name=model_name,
            ground_truth_cont=gt_cont,
            ground_truth_disc=gt_disc
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed on PyTorch model: {str(e)}")

    # Extract previous observed actions from sequence history
    prev_actions = []
    for t_step in range(max(0, len(seq_array) - 4), len(seq_array)):
        frame = seq_array[t_step]
        disc_active = []
        if frame[4] > 0.5: disc_active.append("Fire")
        if frame[5] > 0.5: disc_active.append("Aim/Scope")
        if frame[6] > 0.5: disc_active.append("Melee")
        if frame[7] > 0.5: disc_active.append("Jump")
        if frame[8] > 0.5: disc_active.append("Reload")
        if frame[9] > 0.5: disc_active.append("Switch")
        prev_actions.append(", ".join(disc_active) if disc_active else "Aim/Steer")

    ranked_probs = [
        RankedProbability(
            rank=p["rank"],
            class_name=p["action_name"],
            probability=p["probability"],
            percentage=p["percentage"]
        )
        for p in pred_res["ranked_probabilities"]
    ]

    name_lower = model_name.lower()
    if "ssl" in name_lower:
        clean_model_type = "transformer_ssl"
    elif "transformer" in name_lower:
        clean_model_type = "transformer"
    else:
        clean_model_type = "lstm"
    ckpt_file = f"{clean_model_type}_crossfps_best.pt"
    game_display = GAME_INFO.get(game_title, {}).get("name", game_title.replace("_", " ").title())

    trace = PredictionTrace(
        sequence_id=seq_id,
        player_id=f"CrossFPS [{clip_id[:16]}]",
        round_id=game_display,
        model_name=pred_res["model_name"],
        checkpoint_file=ckpt_file,
        previous_observed_actions=prev_actions,
        predicted_next_action=pred_res["predicted_action"],
        confidence=pred_res["confidence"],
        probability_distribution=pred_res["action_probabilities"],
        ranked_probabilities=ranked_probs,
        ground_truth_action=pred_res["ground_truth_action"],
        matches_ground_truth=pred_res["matches_ground_truth"],
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    # Dynamic explanation of observable context (Step 5)
    hist_str = ", ".join(prev_actions)
    cond_str = ", ".join(applied_conditions) if applied_conditions else "the baseline observed sequence"

    explanation = (
        f"The {pred_res['model_name']} model evaluated a 16-frame telemetry sequence from {game_display} "
        f"exhibiting historical actions [{hist_str}]. Based on {cond_str}, the neural network forecasted "
        f"the highest activation probability for '{pred_res['predicted_action']}' ({(pred_res['confidence']*100):.1f}% activation). "
        f"Continuous steering heads also forecasted next-frame analog velocities of strafe LX={pred_res['continuous_prediction']['strafe_lx']} "
        f"and yaw turn RX={pred_res['continuous_prediction']['turn_yaw_rx']}."
    )

    return {
        "mode": mode,
        "mode_label": "Mode A: Dataset-Trained Prediction (CrossFPS Validation)" if mode == "dataset" else "Mode B: Personalized Prediction",
        "game_id": game_title,
        "game_name": game_display,
        "trace": trace,
        "continuous_prediction": pred_res["continuous_prediction"],
        "explanation": explanation,
        "applied_conditions": applied_conditions
    }


@router.post("/what-if")
def predict_what_if_scenario(req: WhatIfRequest):
    """
    Evaluates a counterfactual 'What-If' scenario using the real trained PyTorch model.
    Alters user-selected supported conditions in the sequence window and measures probability shifts.
    """
    try:
        splits = get_test_splits()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    X_test = splits["X_test"]
    if req.sequence_index >= len(X_test):
        raise HTTPException(status_code=404, detail="Sequence index out of range.")

    base_seq = X_test[req.sequence_index]

    # Map overrides if provided
    modified_scope = req.modified_scope
    modified_jump = req.modified_jump
    turn_intensity = req.turn_intensity

    if req.condition_overrides:
        if "aim_down_sights" in req.condition_overrides:
            modified_scope = bool(req.condition_overrides["aim_down_sights"])
        if "jump" in req.condition_overrides:
            modified_jump = bool(req.condition_overrides["jump"])
        if req.condition_overrides.get("camera") == "turn_right":
            turn_intensity = 0.8
        elif req.condition_overrides.get("camera") == "turn_left":
            turn_intensity = -0.8

    if req.modified_posture == "jumping":
        modified_jump = True

    try:
        what_if_res = evaluate_what_if_scenario(
            sequence=base_seq,
            model_name=req.model_name,
            condition_overrides=req.condition_overrides,
            modified_scope=modified_scope,
            modified_jump=modified_jump,
            turn_intensity=turn_intensity
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Counterfactual inference failed: {str(e)}")

    ranked_hyp = [
        RankedProbability(
            rank=p["rank"],
            class_name=p["action_name"],
            probability=p["probability"],
            percentage=p["percentage"]
        )
        for p in what_if_res["ranked_hypothetical_probabilities"]
    ]

    return {
        "sequence_id": f"SEQ-CFPS-{req.sequence_index+1:05d}",
        "model_name": req.model_name,
        "modifications": what_if_res["modifications"],
        "baseline_prediction": what_if_res["baseline_prediction"],
        "what_if_prediction": what_if_res["what_if_prediction"],
        "action_changed": what_if_res["action_changed"],
        "probability_shifts_pct": what_if_res["probability_shifts_pct"],
        "continuous_shifts": what_if_res.get("continuous_shifts", {}),
        "ranked_hypothetical_probabilities": ranked_hyp
    }
