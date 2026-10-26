"""
Data loading and synthetic test data generation module for GAMER-TWIN.
Supports EgoCS-400K raw tick/event schemas and synthetic test fixtures.
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import pandas as pd

from src.utils import setup_logger, PROJECT_CLASSES

logger = setup_logger("DataLoader")


def load_raw_egocs_data(
    data_dir: str,
    max_records: Optional[int] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads raw EgoCS-400K telemetry data from a directory.
    Searches for 'ticks.csv' and 'events.csv' (or combined telemetry).
    Synchronizes ticks with discrete actions chronologically.
    
    Returns:
        df (pd.DataFrame): Synchronized tick-level dataframe.
        meta (dict): Metadata including data source and detected files.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset directory '{data_dir}' does not exist. "
            f"Please check data/README.md for download instructions or use synthetic demo data."
        )

    # Search for tick files and event files
    tick_files = list(data_path.rglob("*tick*.csv*"))
    event_files = list(data_path.rglob("*event*.csv*"))

    detected_files = {
        "tick_files": [str(f.relative_to(data_path)) for f in tick_files],
        "event_files": [str(f.relative_to(data_path)) for f in event_files]
    }

    if not tick_files:
        # Check if there is any general csv file
        csv_files = list(data_path.glob("*.csv*"))
        if not csv_files:
            raise FileNotFoundError(
                f"No tick or event CSV files found in '{data_dir}'. "
                f"Refer to data/README.md for EgoCS-400K placement instructions."
            )
        tick_files = [csv_files[0]]
        logger.info(f"Using generic telemetry CSV file: {tick_files[0].name}")

    # Load tick data
    logger.info(f"Loading tick records from {tick_files[0]}...")
    ticks_df = pd.read_csv(tick_files[0], nrows=max_records)

    # Check for synthetic label
    is_synthetic = "is_synthetic" in ticks_df.columns or "data_source" in ticks_df.columns
    data_source_label = "SYNTHETIC_TEST_FIXTURE" if is_synthetic else "REAL_EGOCS_400K"

    # Merge events if event file exists and ticks doesn't already have action column
    if event_files and "action" not in ticks_df.columns:
        logger.info(f"Merging event records from {event_files[0]}...")
        events_df = pd.read_csv(event_files[0])
        ticks_df = _synchronize_ticks_and_events(ticks_df, events_df)

    metadata = {
        "data_source": data_source_label,
        "is_synthetic": is_synthetic,
        "raw_record_count": len(ticks_df),
        "detected_files": detected_files,
        "columns": list(ticks_df.columns)
    }

    logger.info(f"Successfully loaded {len(ticks_df)} records. Source: {data_source_label}")
    return ticks_df, metadata


def _synchronize_ticks_and_events(ticks_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Synchronizes tick telemetry with discrete event logs chronologically.
    Maps event actions onto matching ticks or nearest tick per player.
    """
    ticks_df = ticks_df.copy()
    if "tick" not in ticks_df.columns:
        ticks_df["tick"] = np.arange(len(ticks_df))

    if "tick" in events_df.columns and "action" in events_df.columns:
        merge_keys = ["tick"]
        if "player_id" in ticks_df.columns and "player_id" in events_df.columns:
            merge_keys.append("player_id")

        merged = pd.merge(ticks_df, events_df[merge_keys + ["action"]], on=merge_keys, how="left")
        merged["action"] = merged["action"].fillna("Move")
        return merged

    # Fallback default action
    if "action" not in ticks_df.columns:
        ticks_df["action"] = "Move"
    return ticks_df


def generate_synthetic_gameplay_data(
    num_rounds: int = 5,
    ticks_per_round: int = 500,
    output_dir: Optional[str] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates realistic, physically plausible Counter-Strike gameplay telemetry.
    Strictly labelled as SYNTHETIC TEST FIXTURE for testing the pipeline and UI.
    
    Includes:
    - 3D Coordinates (pos_x, pos_y, pos_z)
    - Velocities (vel_x, vel_y, vel_z)
    - View Angles (view_pitch, view_yaw) and changes (delta_pitch, delta_yaw)
    - Control Inputs (in_forward, in_back, in_moveleft, in_moveright, in_attack, in_reload, in_jump, in_duck)
    - Status: health, armor, active_weapon, ammo, is_scoped
    - All 9 target classes represented
    """
    np.random.seed(seed)
    records = []
    player_id = "synthetic_player_01"
    weapons = ["ak47", "m4a1", "awp", "deagle", "usp_silencer", "knife"]

    actions_pool = PROJECT_CLASSES

    for round_idx in range(1, num_rounds + 1):
        round_id = f"round_{round_idx:02d}"
        pos_x = np.random.uniform(-1000.0, 1000.0)
        pos_y = np.random.uniform(-1000.0, 1000.0)
        pos_z = 0.0
        pitch = 0.0
        yaw = np.random.uniform(-180.0, 180.0)
        current_weapon = np.random.choice(weapons[:4])
        ammo = 30
        health = 100
        prev_action = "Move"

        for tick_idx in range(ticks_per_round):
            tick = tick_idx + 1
            seconds = round(tick * 0.015625, 4)  # ~64 tick CS server

            # Random walk dynamics for smooth player movement
            vel_x = np.clip(np.random.normal(0, 120), -250, 250)
            vel_y = np.clip(np.random.normal(0, 120), -250, 250)
            vel_z = 0.0

            pos_x += vel_x * 0.015625
            pos_y += vel_y * 0.015625

            delta_pitch = np.clip(np.random.normal(0, 1.2), -15.0, 15.0)
            delta_yaw = np.clip(np.random.normal(0, 2.5), -30.0, 30.0)
            pitch = np.clip(pitch + delta_pitch, -89.0, 89.0)
            yaw = ((yaw + delta_yaw + 180.0) % 360.0) - 180.0

            # Determine action with realistic transition probabilities
            prob_event = np.random.rand()
            if prob_event < 0.55:
                action = "Move"
            elif prob_event < 0.65:
                action = "Fire"
            elif prob_event < 0.72:
                action = "Jump"
            elif prob_event < 0.79:
                action = "Crouch"
            elif prob_event < 0.85:
                action = "Scope"
            elif prob_event < 0.90:
                action = "Reload"
            elif prob_event < 0.94:
                action = "Switch Weapon"
            elif prob_event < 0.97:
                action = "Grenade"
            else:
                action = "Inspect"

            # Button states aligned with action
            in_forward = int(vel_y > 20 or (action == "Move" and np.random.rand() > 0.3))
            in_back = int(vel_y < -20 and not in_forward)
            in_moveleft = int(vel_x < -20)
            in_moveright = int(vel_x > 20 and not in_moveleft)
            in_attack = int(action == "Fire")
            in_reload = int(action == "Reload")
            in_jump = int(action == "Jump")
            in_duck = int(action == "Crouch")
            is_scoped = int(action == "Scope" or (current_weapon == "awp" and np.random.rand() > 0.6))

            if action == "Fire":
                ammo = max(0, ammo - 1)
            elif action == "Reload":
                ammo = 30
            elif action == "Switch Weapon":
                current_weapon = np.random.choice(weapons)

            if action == "Jump":
                pos_z = min(64.0, pos_z + 16.0)
                vel_z = 180.0
            else:
                pos_z = max(0.0, pos_z - 8.0)
                vel_z = 0.0

            records.append({
                "tick": tick,
                "seconds": seconds,
                "player_id": player_id,
                "round_id": round_id,
                "pos_x": round(float(pos_x), 2),
                "pos_y": round(float(pos_y), 2),
                "pos_z": round(float(pos_z), 2),
                "vel_x": round(float(vel_x), 2),
                "vel_y": round(float(vel_y), 2),
                "vel_z": round(float(vel_z), 2),
                "view_pitch": round(float(pitch), 2),
                "view_yaw": round(float(yaw), 2),
                "delta_pitch": round(float(delta_pitch), 2),
                "delta_yaw": round(float(delta_yaw), 2),
                "in_forward": in_forward,
                "in_back": in_back,
                "in_moveleft": in_moveleft,
                "in_moveright": in_moveright,
                "in_attack": in_attack,
                "in_reload": in_reload,
                "in_jump": in_jump,
                "in_duck": in_duck,
                "is_scoped": is_scoped,
                "active_weapon": current_weapon,
                "ammo": ammo,
                "health": health,
                "prev_action": prev_action,
                "action": action,
                # Explicit Synthetic provenance flag
                "data_source": "SYNTHETIC_TEST_FIXTURE",
                "is_synthetic": 1
            })
            prev_action = action

    df = pd.DataFrame(records)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        ticks_file = os.path.join(output_dir, "ticks.csv")
        events_file = os.path.join(output_dir, "events.csv")
        df.to_csv(ticks_file, index=False)
        events_df = df[["tick", "seconds", "player_id", "round_id", "action"]].copy()
        events_df.to_csv(events_file, index=False)
        logger.info(f"Synthetic demo dataset saved to {output_dir} ({len(df)} records).")

    return df
