"""
Next-action prediction and inference engine for GAMER-TWIN.
Provides probabilities, ranked predictions, and real-time comparison for gameplay sequences.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from src.utils import setup_logger, PROJECT_CLASSES, IDX_TO_CLASS, get_device
from src.evaluation import load_model_from_checkpoint

logger = setup_logger("Prediction")


class NextActionPredictor:
    """
    Inference wrapper for trained GAMER-TWIN models.
    Takes normalized gameplay sequences and predicts next-action probability distributions.
    """
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu"
    ):
        self.device = device
        self.model = None
        self.checkpoint_meta = None

        if checkpoint_path is not None:
            self.model, self.checkpoint_meta = load_model_from_checkpoint(checkpoint_path, device=device)
        elif model is not None:
            self.model = model.to(device)
            self.model.eval()

    def is_ready(self) -> bool:
        """Returns True if a trained model is loaded and ready for inference."""
        return self.model is not None

    def predict_sequence(
        self,
        sequence: Union[np.ndarray, torch.Tensor],
        ground_truth_action: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Predicts next action from a single sequence window.
        
        Args:
            sequence: Array of shape (window_size, num_features) or (1, window_size, num_features)
            ground_truth_action: Optional recorded true next action for comparison
            
        Returns:
            Dictionary with predicted action, confidence, ranked probabilities, and ground truth match.
        """
        if not self.is_ready():
            raise RuntimeError("Model is not initialized or trained checkpoint is missing.")

        if isinstance(sequence, np.ndarray):
            if sequence.ndim == 2:
                sequence = np.expand_dims(sequence, axis=0)  # Shape (1, T, D)
            tensor_seq = torch.tensor(sequence, dtype=torch.float32, device=self.device)
        elif isinstance(sequence, torch.Tensor):
            if sequence.dim() == 2:
                tensor_seq = sequence.unsqueeze(0).to(self.device)
            else:
                tensor_seq = sequence.to(self.device)
        else:
            raise TypeError("Sequence must be a numpy ndarray or torch Tensor.")

        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor_seq)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        top_idx = int(np.argmax(probs))
        predicted_action = IDX_TO_CLASS[top_idx]
        confidence = float(probs[top_idx])

        # Ranked list of all 9 classes by probability
        ranked_indices = np.argsort(probs)[::-1]
        ranked_probabilities = [
            {
                "rank": rank + 1,
                "class_name": IDX_TO_CLASS[idx],
                "probability": round(float(probs[idx]), 4),
                "percentage": round(float(probs[idx] * 100), 2)
            }
            for rank, idx in enumerate(ranked_indices)
        ]

        class_probabilities = {
            IDX_TO_CLASS[idx]: round(float(probs[idx]), 4)
            for idx in range(len(PROJECT_CLASSES))
        }

        matches_ground_truth = None
        if ground_truth_action is not None:
            matches_ground_truth = bool(predicted_action.lower() == str(ground_truth_action).lower())

        return {
            "predicted_action": predicted_action,
            "confidence": round(confidence, 4),
            "confidence_percentage": round(confidence * 100, 2),
            "class_probabilities": class_probabilities,
            "ranked_probabilities": ranked_probabilities,
            "ground_truth_action": ground_truth_action,
            "matches_ground_truth": matches_ground_truth
        }
