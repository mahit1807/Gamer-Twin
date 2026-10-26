"""
Feature preprocessing and target alignment module for GAMER-TWIN.
Ensures zero data leakage by fitting normalizers and encoders on training split only.
"""

import os
import json
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

from src.utils import setup_logger, PROJECT_CLASSES, CLASS_TO_IDX

logger = setup_logger("Preprocessing")

# Transparent Synonym Mapping to the 9 Project Target Classes
ACTION_SYNONYM_MAP = {
    # 1. Fire
    "fire": "Fire",
    "shoot": "Fire",
    "attack": "Fire",
    "in_attack": "Fire",
    "primary_attack": "Fire",
    
    # 2. Reload
    "reload": "Reload",
    "in_reload": "Reload",
    
    # 3. Switch Weapon
    "switch weapon": "Switch Weapon",
    "switch_weapon": "Switch Weapon",
    "switch": "Switch Weapon",
    "drop_weapon": "Switch Weapon",
    "slot1": "Switch Weapon",
    "slot2": "Switch Weapon",
    "slot3": "Switch Weapon",
    
    # 4. Inspect
    "inspect": "Inspect",
    "inspect_weapon": "Inspect",
    
    # 5. Scope
    "scope": "Scope",
    "in_attack2": "Scope",
    "secondary_attack": "Scope",
    "zoom": "Scope",
    
    # 6. Grenade
    "grenade": "Grenade",
    "throw_grenade": "Grenade",
    "flashbang": "Grenade",
    "smokegrenade": "Grenade",
    "hegrenade": "Grenade",
    "molotov": "Grenade",
    "incgrenade": "Grenade",
    "decoy": "Grenade",
    
    # 7. Jump
    "jump": "Jump",
    "in_jump": "Jump",
    
    # 8. Crouch
    "crouch": "Crouch",
    "duck": "Crouch",
    "in_duck": "Crouch",
    
    # 9. Move
    "move": "Move",
    "walk": "Move",
    "run": "Move",
    "movement": "Move",
    "idle": "Move"  # Idle ticks without active keys treated as resting Move
}

# Explicit Exclusion Policy: events that do not represent physical player combat/movement actions
EXCLUDED_EVENT_TYPES = {
    "round_start", "round_end", "player_death", "bomb_planted", "bomb_defused",
    "chat", "buy_item", "player_connect", "team_switch", "match_pause"
}

KNOWN_WEAPONS = [
    "ak47", "m4a1", "m4a1_silencer", "awp", "deagle", "usp_silencer", 
    "glock", "knife", "mp9", "galilar", "famas", "sg553", "ssg08", "other"
]
WEAPON_TO_IDX = {w: i for i, w in enumerate(KNOWN_WEAPONS)}


def map_action_to_target_class(raw_action: Any) -> Optional[str]:
    """
    Transparently maps a raw event or action string to one of the 9 project classes.
    Returns None if the action is excluded by policy.
    """
    if pd.isna(raw_action):
        return None

    action_clean = str(raw_action).strip().lower()

    if action_clean in EXCLUDED_EVENT_TYPES:
        return None

    if action_clean in ACTION_SYNONYM_MAP:
        return ACTION_SYNONYM_MAP[action_clean]

    # Direct case-insensitive match against project classes
    for target in PROJECT_CLASSES:
        if action_clean == target.lower():
            return target

    return None


class GameplayFeaturePreprocessor:
    """
    Preprocessor for player telemetry that strictly fits on training data
    to prevent data leakage into validation or testing sets.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        schema_cfg = config.get("data", {}).get("schema_mapping", {})
        self.numerical_cols = schema_cfg.get("numerical_features", [
            "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z",
            "view_pitch", "view_yaw", "delta_pitch", "delta_yaw", "ammo", "health"
        ])
        self.binary_cols = schema_cfg.get("binary_features", [
            "in_forward", "in_back", "in_moveleft", "in_moveright",
            "in_attack", "in_reload", "in_jump", "in_duck", "is_scoped"
        ])
        self.categorical_cols = schema_cfg.get("categorical_features", ["active_weapon", "prev_action"])

        # Fitted parameters (calculated on train split only)
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}
        self.is_fitted = False
        self.feature_names: List[str] = []

    def fit(self, train_df: pd.DataFrame) -> "GameplayFeaturePreprocessor":
        """Computes means and standard deviations exclusively on training data."""
        logger.info("Fitting preprocessor transformations on training set (zero data leakage)...")

        for col in self.numerical_cols:
            if col in train_df.columns:
                mean_val = float(train_df[col].mean())
                std_val = float(train_df[col].std())
                if np.isnan(std_val) or std_val < 1e-6:
                    std_val = 1.0
                self.means[col] = mean_val
                self.stds[col] = std_val
            else:
                self.means[col] = 0.0
                self.stds[col] = 1.0

        # Construct feature names order
        self.feature_names = []
        for col in self.numerical_cols:
            self.feature_names.append(col)
        for col in self.binary_cols:
            self.feature_names.append(col)
        # Weapon index feature
        self.feature_names.append("active_weapon_idx")
        # Prev action index feature
        self.feature_names.append("prev_action_idx")

        self.is_fitted = True
        logger.info(f"Preprocessor fitted successfully. Total feature dimensionality: {len(self.feature_names)}")
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        """
        Transforms input dataframe into numerical feature matrix X and target labels y.
        Unmapped/excluded actions are excluded from the output.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted on training data before calling transform.")

        df = df.copy()
        initial_count = len(df)

        # 1. Map target classes
        df["target_class"] = df["action"].apply(map_action_to_target_class)
        usable_mask = df["target_class"].notna()
        excluded_count = int(initial_count - usable_mask.sum())
        usable_df = df[usable_mask].copy()

        if len(usable_df) == 0:
            raise ValueError("All records were excluded or no valid target classes were found in the dataset.")

        # 2. Impute and standardize numerical columns
        feature_matrices = []
        for col in self.numerical_cols:
            if col in usable_df.columns:
                vals = usable_df[col].fillna(self.means[col]).to_numpy(dtype=np.float32)
            else:
                vals = np.full(len(usable_df), self.means[col], dtype=np.float32)
            # Standardize using fitted training statistics
            normed = (vals - self.means[col]) / self.stds[col]
            feature_matrices.append(normed.reshape(-1, 1))

        # 3. Binary inputs
        for col in self.binary_cols:
            if col in usable_df.columns:
                vals = usable_df[col].fillna(0).astype(np.float32).to_numpy()
            else:
                vals = np.zeros(len(usable_df), dtype=np.float32)
            feature_matrices.append(vals.reshape(-1, 1))

        # 4. Categorical: weapon
        if "active_weapon" in usable_df.columns:
            weapon_indices = usable_df["active_weapon"].apply(
                lambda w: WEAPON_TO_IDX.get(str(w).lower(), WEAPON_TO_IDX["other"])
            ).to_numpy(dtype=np.float32)
        else:
            weapon_indices = np.full(len(usable_df), WEAPON_TO_IDX["other"], dtype=np.float32)
        feature_matrices.append(weapon_indices.reshape(-1, 1))

        # 5. Categorical: prev_action
        if "prev_action" in usable_df.columns:
            prev_act_indices = usable_df["prev_action"].apply(
                lambda a: CLASS_TO_IDX.get(map_action_to_target_class(a), 0)
            ).to_numpy(dtype=np.float32)
        else:
            prev_act_indices = np.zeros(len(usable_df), dtype=np.float32)
        feature_matrices.append(prev_act_indices.reshape(-1, 1))

        # Stack into single feature array
        X = np.hstack(feature_matrices).astype(np.float32)

        # Targets as integer indices (0 to 8)
        y = usable_df["target_class"].apply(lambda c: CLASS_TO_IDX[c]).to_numpy(dtype=np.int64)

        audit_meta = {
            "initial_records": initial_count,
            "usable_records": len(usable_df),
            "excluded_records": excluded_count
        }

        return X, y, audit_meta

    def save_state(self, filepath: str) -> None:
        """Saves fitted normalizer parameters and feature list to JSON."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        state = {
            "means": self.means,
            "stds": self.stds,
            "feature_names": self.feature_names,
            "numerical_cols": self.numerical_cols,
            "binary_cols": self.binary_cols,
            "categorical_cols": self.categorical_cols
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath: str) -> "GameplayFeaturePreprocessor":
        """Loads fitted parameters from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.means = state["means"]
        self.stds = state["stds"]
        self.feature_names = state["feature_names"]
        self.numerical_cols = state.get("numerical_cols", self.numerical_cols)
        self.binary_cols = state.get("binary_cols", self.binary_cols)
        self.categorical_cols = state.get("categorical_cols", self.categorical_cols)
        self.is_fitted = True
        return self
