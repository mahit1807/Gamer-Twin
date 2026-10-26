"""
Unit tests for evaluation metrics, top-k accuracy, and prediction engine.
"""

import pytest
import numpy as np
import torch

from src.evaluation import evaluate_model, compute_top_k_accuracy
from src.prediction import NextActionPredictor
from src.models import create_model
from src.sequence_builder import GameplaySequenceDataset
from src.utils import load_config, PROJECT_CLASSES


def test_compute_top_k_accuracy():
    """Verifies top-k accuracy calculation."""
    y_true = np.array([0, 1, 2])
    # Case 1: true label is at top-1
    y_prob1 = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8]
    ])
    assert compute_top_k_accuracy(y_true, y_prob1, k=1) == 1.0
    assert compute_top_k_accuracy(y_true, y_prob1, k=3) == 1.0

    # Case 2: true label is rank 2
    y_prob2 = np.array([
        [0.3, 0.6, 0.1],  # true 0, top-1 is 1, top-2 has 0
        [0.7, 0.2, 0.1],  # true 1, top-1 is 0, top-2 has 1
        [0.6, 0.1, 0.3]   # true 2, top-1 is 0, top-2 has 2
    ])
    assert compute_top_k_accuracy(y_true, y_prob2, k=1) == 0.0
    assert compute_top_k_accuracy(y_true, y_prob2, k=2) == 1.0


def test_evaluate_model_outputs_and_zero_division():
    """Verifies evaluate_model returns complete metric dict and handles zero support safely."""
    config = load_config()
    B, T, D = 10, 16, 23
    seqs = np.random.randn(B, T, D).astype(np.float32)
    # Only label classes 0 and 1 present in test
    targets = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)

    test_ds = GameplaySequenceDataset(seqs, targets)
    model = create_model("transformer", input_dim=D, config=config)

    results = evaluate_model(model, test_ds, config, model_name="test_transformer", data_provenance="SYNTHETIC_TEST")

    metrics = results["metrics"]
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "f1_weighted" in metrics
    assert "top_3_accuracy" in metrics
    assert len(results["confusion_matrix"]) == 9
    assert len(results["per_class"]) == 9


def test_next_action_predictor_inference():
    """Verifies NextActionPredictor yields valid probabilities, ranking, and comparison."""
    config = load_config()
    T, D = 16, 23
    seq = np.random.randn(T, D).astype(np.float32)

    model = create_model("transformer", input_dim=D, config=config)
    predictor = NextActionPredictor(model=model)

    res = predictor.predict_sequence(sequence=seq, ground_truth_action="Fire")

    assert "predicted_action" in res
    assert res["predicted_action"] in PROJECT_CLASSES
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["ranked_probabilities"]) == 9
    assert res["ground_truth_action"] == "Fire"
    assert isinstance(res["matches_ground_truth"], bool)
