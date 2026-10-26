"""
Replay Studio API routes for GAMER-TWIN.
Provides offline chronological timeline replay, tick telemetry, real model inference per step,
and Prediction Trace integration.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
import pandas as pd
import numpy as np

from src.utils import PROJECT_CLASSES, get_device, load_config
from src.preprocessing import GameplayFeaturePreprocessor
from backend.app.api.routes_prediction import get_predictor_for_model
from backend.app.schemas.schemas import ReplayTelemetryStep, PredictionTrace, RankedProbability

router = APIRouter()

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Cache preloaded replay dataframe and preprocessor
_REPLAY_CACHE = {}


def load_replay_data():
    if "df" in _REPLAY_CACHE:
        return _REPLAY_CACHE["df"], _REPLAY_CACHE["preprocessor"], _REPLAY_CACHE["is_synthetic"]

    raw_dir = WORKSPACE_ROOT / "data" / "raw"
    sample_dir = WORKSPACE_ROOT / "data" / "sample_data"

    if raw_dir.exists() and any(raw_dir.glob("*.csv")):
        csv_file = next(raw_dir.glob("*tick*.csv"), None) or next(raw_dir.glob("*.csv"), None)
        is_synthetic = False
    elif sample_dir.exists() and any(sample_dir.glob("*.csv")):
        csv_file = next(sample_dir.glob("*tick*.csv"), None) or next(sample_dir.glob("*.csv"), None)
        is_synthetic = True
    else:
        raise FileNotFoundError("No telemetry files available for replay.")

    df = pd.read_csv(csv_file)
    config = load_config()
    preprocessor = GameplayFeaturePreprocessor(config)
    preprocessor.fit(df)

    _REPLAY_CACHE["df"] = df
    _REPLAY_CACHE["preprocessor"] = preprocessor
    _REPLAY_CACHE["is_synthetic"] = is_synthetic

    return df, preprocessor, is_synthetic


@router.get("/info")
def get_replay_info():
    try:
        df, preprocessor, is_synthetic = load_replay_data()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    window_size = 16
    total_steps = max(0, len(df) - window_size)

    players = df["player_id"].unique().tolist() if "player_id" in df.columns else ["player_01"]
    rounds = df["round_id"].unique().tolist() if "round_id" in df.columns else ["round_01"]

    return {
        "total_records": len(df),
        "total_steps": total_steps,
        "window_size": window_size,
        "players": [str(p) for p in players],
        "rounds": [str(r) for r in rounds],
        "data_source": "SYNTHETIC_TEST_FIXTURE" if is_synthetic else "REAL_EGOCS_400K",
        "is_synthetic": is_synthetic,
        "video_available": False,
        "video_notice": "Offline Structured Telemetry Mode (Replay is driven directly by high-frequency synchronized game telemetry)."
    }


@router.get("/step", response_model=ReplayTelemetryStep)
def get_replay_step(
    step_idx: int = Query(0, ge=0),
    model_name: str = Query("transformer")
):
    try:
        df, preprocessor, is_synthetic = load_replay_data()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    window_size = 16
    total_steps = max(0, len(df) - window_size)
    if step_idx >= total_steps:
        raise HTTPException(status_code=400, detail=f"step_idx {step_idx} exceeds total steps {total_steps}.")

    # Current tick record
    curr_idx = step_idx + window_size - 1
    row = df.iloc[curr_idx]

    # Preprocess sequence window
    window_df = df.iloc[step_idx:step_idx + window_size]
    X_window, y_window, _ = preprocessor.transform(window_df)

    # Actual next action
    target_action_str = str(row.get("action", "Move"))
    if curr_idx + 1 < len(df):
        target_action_str = str(df.iloc[curr_idx + 1].get("action", target_action_str))

    # Recent observed actions in window
    recent_actions = [str(r) for r in window_df.get("action", ["Move"] * window_size)]

    # Query model prediction
    pred_trace = None
    try:
        predictor = get_predictor_for_model(model_name)
        pred_res = predictor.predict_sequence(
            sequence=X_window,
            ground_truth_action=target_action_str
        )
        ranked_probs = [
            RankedProbability(
                rank=p["rank"],
                class_name=p["class_name"],
                probability=p["probability"],
                percentage=p["percentage"]
            )
            for p in pred_res["ranked_probabilities"]
        ]
        pred_trace = PredictionTrace(
            sequence_id=f"REPLAY-STEP-{step_idx+1:04d}",
            player_id=str(row.get("player_id", "player_01")),
            round_id=str(row.get("round_id", "round_01")),
            model_name=model_name,
            checkpoint_file=f"{model_name}_best.pt",
            previous_observed_actions=recent_actions[-4:],
            predicted_next_action=pred_res["predicted_action"],
            confidence=pred_res["confidence"],
            probability_distribution=pred_res["class_probabilities"],
            ranked_probabilities=ranked_probs,
            ground_truth_action=target_action_str,
            matches_ground_truth=pred_res["matches_ground_truth"],
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    except Exception:
        # Checkpoint might not be loaded yet or error
        pass

    inputs = {
        "W": int(row.get("in_forward", 0)),
        "S": int(row.get("in_back", 0)),
        "A": int(row.get("in_moveleft", 0)),
        "D": int(row.get("in_moveright", 0)),
        "Attack": int(row.get("in_attack", 0)),
        "Reload": int(row.get("in_reload", 0)),
        "Jump": int(row.get("in_jump", 0)),
        "Crouch": int(row.get("in_duck", 0)),
        "Scope": int(row.get("is_scoped", 0))
    }

    return ReplayTelemetryStep(
        step_idx=step_idx,
        total_steps=total_steps,
        tick=int(row.get("tick", step_idx)),
        seconds=float(row.get("seconds", step_idx * 0.0156)),
        player_id=str(row.get("player_id", "player_01")),
        round_id=str(row.get("round_id", "round_01")),
        pos_x=round(float(row.get("pos_x", 0.0)), 2),
        pos_y=round(float(row.get("pos_y", 0.0)), 2),
        pos_z=round(float(row.get("pos_z", 0.0)), 2),
        vel_x=round(float(row.get("vel_x", 0.0)), 2),
        vel_y=round(float(row.get("vel_y", 0.0)), 2),
        vel_z=round(float(row.get("vel_z", 0.0)), 2),
        view_pitch=round(float(row.get("view_pitch", 0.0)), 2),
        view_yaw=round(float(row.get("view_yaw", 0.0)), 2),
        inputs=inputs,
        active_weapon=str(row.get("active_weapon", "ak47")),
        ammo=int(row.get("ammo", 30)),
        health=int(row.get("health", 100)),
        recent_actions=recent_actions,
        actual_next_action=target_action_str,
        prediction_trace=pred_trace,
        data_source="SYNTHETIC_TEST_FIXTURE" if is_synthetic else "REAL_EGOCS_400K",
        is_synthetic=is_synthetic
    )


@router.get("/timeline")
def get_replay_timeline(limit: int = Query(60, ge=10, le=200)):
    try:
        df, _, is_synthetic = load_replay_data()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    num_records = min(limit, len(df))
    subset = df.iloc[:num_records]

    timeline_events = []
    for idx, row in subset.iterrows():
        action = str(row.get("action", "Move"))
        timeline_events.append({
            "step_idx": int(idx),
            "tick": int(row.get("tick", idx)),
            "seconds": round(float(row.get("seconds", idx * 0.0156)), 2),
            "action": action,
            "is_combat": action in ["Fire", "Reload", "Switch Weapon", "Scope", "Grenade"],
            "weapon": str(row.get("active_weapon", "ak47")),
            "pos": [round(float(row.get("pos_x", 0)), 1), round(float(row.get("pos_y", 0)), 1)]
        })

    return {
        "total_events": len(timeline_events),
        "events": timeline_events,
        "is_synthetic": is_synthetic
    }
