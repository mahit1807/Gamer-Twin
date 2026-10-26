"""
Model evaluation module for GAMER-TWIN.
Computes multi-class metrics, confusion matrices, and top-k accuracy with strict results integrity.
Implemented using pure NumPy to guarantee portability and robustness across all environments.
"""

import os
import argparse
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils import setup_logger, load_config, save_json, PROJECT_CLASSES, get_device
from src.models import create_model
from src.sequence_builder import GameplaySequenceDataset

logger = setup_logger("Evaluation")


def compute_top_k_accuracy(y_true: np.ndarray, y_prob: np.ndarray, k: int = 3) -> float:
    """
    Computes Top-K categorical accuracy.
    Checks if the ground-truth label is among the top K highest predicted probabilities.
    """
    if len(y_true) == 0:
        return 0.0
    num_classes = y_prob.shape[1]
    k = min(k, num_classes)
    top_k_preds = np.argsort(y_prob, axis=1)[:, -k:]
    matches = [y_true[i] in top_k_preds[i] for i in range(len(y_true))]
    return float(np.mean(matches))


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 9) -> np.ndarray:
    """Computes num_classes x num_classes confusion matrix using pure NumPy."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 9
) -> Tuple[float, float, float, float, float, Dict[str, Dict[str, Any]]]:
    """
    Pure NumPy implementation of multi-class classification metrics:
    - Overall Accuracy
    - Macro Precision, Recall, F1
    - Weighted Precision, Recall, F1
    - Per-class breakdowns with zero_division=0 handling
    """
    if len(y_true) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, {}

    acc = float(np.mean(y_true == y_pred))

    precisions = []
    recalls = []
    f1s = []
    supports = []
    per_class = {}

    for idx in range(num_classes):
        cls_name = PROJECT_CLASSES[idx]
        tp = int(np.sum((y_true == idx) & (y_pred == idx)))
        fp = int(np.sum((y_true != idx) & (y_pred == idx)))
        fn = int(np.sum((y_true == idx) & (y_pred != idx)))
        support = int(np.sum(y_true == idx))
        pred_count = int(np.sum(y_pred == idx))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(support)

        per_class[cls_name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "support": support,
            "predicted_count": pred_count
        }

    total_support = sum(supports)
    if total_support > 0:
        weights = np.array(supports, dtype=np.float64) / total_support
        prec_weighted = float(np.sum(np.array(precisions) * weights))
        rec_weighted = float(np.sum(np.array(recalls) * weights))
        f1_weighted = float(np.sum(np.array(f1s) * weights))
    else:
        prec_weighted, rec_weighted, f1_weighted = 0.0, 0.0, 0.0

    prec_macro = float(np.mean(precisions))
    rec_macro = float(np.mean(recalls))
    f1_macro = float(np.mean(f1s))

    return acc, prec_macro, prec_weighted, rec_macro, rec_weighted, f1_macro, f1_weighted, per_class


def evaluate_model(
    model: nn.Module,
    test_dataset: GameplaySequenceDataset,
    config: Dict[str, Any],
    model_name: str = "Transformer",
    data_provenance: str = "REAL_OR_SYNTHETIC",
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Evaluates a trained model on the held-out test split.
    Calculates actual accuracy, macro/weighted precision, recall, f1, and top-k accuracy.
    
    Strict results integrity: Every single metric is computed directly from test forward passes.
    """
    device = get_device(config.get("training", {}).get("device", "auto"))
    model = model.to(device)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.append(probs)
            all_preds.append(preds)
            all_targets.append(batch_y.numpy())

    if not all_targets:
        raise ValueError("Test dataset is empty. Cannot evaluate model.")

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_prob = np.vstack(all_probs)

    num_classes = len(PROJECT_CLASSES)

    # 1. Compute metrics via pure NumPy
    acc, prec_m, prec_w, rec_m, rec_w, f1_m, f1_w, per_class = compute_classification_metrics(y_true, y_pred, num_classes)
    top_k_acc = compute_top_k_accuracy(y_true, y_prob, k=top_k)

    # 2. Confusion matrix
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)

    results = {
        "model_name": model_name,
        "data_provenance": data_provenance,
        "is_synthetic": "SYNTHETIC" in data_provenance.upper(),
        "total_test_samples": int(len(y_true)),
        "metrics": {
            "accuracy": round(acc, 4),
            f"top_{top_k}_accuracy": round(top_k_acc, 4),
            "precision_macro": round(prec_m, 4),
            "precision_weighted": round(prec_w, 4),
            "recall_macro": round(rec_m, 4),
            "recall_weighted": round(rec_w, 4),
            "f1_macro": round(f1_m, 4),
            "f1_weighted": round(f1_w, 4)
        },
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_labels": PROJECT_CLASSES
    }

    logger.info(
        f"Evaluation Summary for [{model_name}] on {data_provenance}:\n"
        f"  Accuracy: {acc*100:.2f}%\n"
        f"  Top-{top_k} Acc: {top_k_acc*100:.2f}%\n"
        f"  F1 (Macro): {f1_m:.4f} | F1 (Weighted): {f1_w:.4f}"
    )

    return results


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cpu") -> Tuple[nn.Module, Dict[str, Any]]:
    """Loads a model and its saved training configuration from checkpoint file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' does not exist.")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = checkpoint.get("model_name", "transformer")
    input_dim = checkpoint["input_dim"]
    config = checkpoint.get("config", load_config())

    model = create_model(model_name, input_dim, config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def save_evaluation_results(results: Dict[str, Any], output_path: str) -> None:
    """Saves structured evaluation results to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_json(results, output_path)
    logger.info(f"Evaluation report saved to {output_path}")
