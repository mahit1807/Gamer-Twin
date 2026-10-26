"""
Offline gameplay replay and simulation engine for GAMER-TWIN.
Provides tick-by-tick or step-by-step sequential inspection comparing
ground-truth player actions with the learned Digital Twin's predicted actions.
"""

from typing import Dict, Any, List, Optional, Tuple, Generator
import numpy as np
import pandas as pd

from src.utils import setup_logger, PROJECT_CLASSES
from src.prediction import NextActionPredictor
from src.preprocessing import GameplayFeaturePreprocessor

logger = setup_logger("Simulation")


class ReplaySimulator:
    """
    Offline chronological replay simulator for GAMER-TWIN.
    Replays historical match ticks and queries the trained model at each step.
    """
    def __init__(
        self,
        df_records: pd.DataFrame,
        preprocessor: GameplayFeaturePreprocessor,
        predictor: Optional[NextActionPredictor] = None,
        window_size: int = 16,
        is_synthetic: bool = False
    ):
        self.df = df_records.copy().reset_index(drop=True)
        self.preprocessor = preprocessor
        self.predictor = predictor
        self.window_size = window_size
        self.is_synthetic = is_synthetic
        self.data_source_label = "SYNTHETIC_TEST_FIXTURE" if is_synthetic else "REAL_EGOCS_400K"

        # Pre-transform records
        self.X, self.y, _ = self.preprocessor.transform(self.df)
        self.total_steps = max(0, len(self.df) - self.window_size)

    def get_step_state(self, step_idx: int) -> Dict[str, Any]:
        """
        Retrieves the gameplay state and model prediction at a specific chronological step.
        """
        if step_idx < 0 or step_idx >= self.total_steps:
            raise IndexError(f"Step index {step_idx} out of range [0, {self.total_steps - 1}].")

        # Current window of telemetry
        seq_window = self.X[step_idx:step_idx + self.window_size]
        actual_target_idx = self.y[step_idx + self.window_size - 1]
        actual_action = PROJECT_CLASSES[actual_target_idx]

        # Current tick metadata
        current_record = self.df.iloc[step_idx + self.window_size - 1].to_dict()

        # Query predictor if available
        prediction_result = None
        if self.predictor is not None and self.predictor.is_ready():
            prediction_result = self.predictor.predict_sequence(
                sequence=seq_window,
                ground_truth_action=actual_action
            )

        # Recent action history in the sequence window
        recent_actions = [
            PROJECT_CLASSES[self.y[step_idx + i]]
            for i in range(self.window_size)
        ]

        return {
            "step_index": step_idx,
            "total_steps": self.total_steps,
            "data_source": self.data_source_label,
            "is_synthetic": self.is_synthetic,
            "tick": current_record.get("tick", step_idx + self.window_size),
            "seconds": current_record.get("seconds", round((step_idx + self.window_size) * 0.0156, 2)),
            "round_id": current_record.get("round_id", "N/A"),
            "player_id": current_record.get("player_id", "N/A"),
            "current_weapon": current_record.get("active_weapon", "N/A"),
            "ammo": current_record.get("ammo", 0),
            "health": current_record.get("health", 100),
            "pos_x": current_record.get("pos_x", 0.0),
            "pos_y": current_record.get("pos_y", 0.0),
            "pos_z": current_record.get("pos_z", 0.0),
            "view_pitch": current_record.get("view_pitch", 0.0),
            "view_yaw": current_record.get("view_yaw", 0.0),
            "recent_actions": recent_actions,
            "actual_action": actual_action,
            "prediction": prediction_result,
            "matches_prediction": (
                prediction_result["matches_ground_truth"]
                if prediction_result is not None else None
            )
        }
