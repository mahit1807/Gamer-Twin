"""
Unit tests for sliding sequence construction and group-based splitting.
"""

import pytest
import numpy as np
import pandas as pd

from src.sequence_builder import create_sequences_for_group, prepare_train_val_test_splits
from src.data_loader import generate_synthetic_gameplay_data
from src.utils import load_config


def test_create_sequences_for_group_dimensions_and_target_alignment():
    """Verifies sequence window shapes and target offset alignment."""
    # 20 timesteps with 5 features
    X_group = np.arange(100, dtype=np.float32).reshape(20, 5)
    # Targets correspond to step index
    y_group = np.arange(20, dtype=np.int64)

    window_size = 8
    stride = 2
    target_offset = 1

    seqs, targets = create_sequences_for_group(X_group, y_group, window_size=window_size, stride=stride, target_offset=target_offset)

    # Number of windows: (20 - 8 - 1) // 2 + 1 = 11 // 2 + 1 = 6
    assert len(seqs) == 6
    assert seqs.shape == (6, 8, 5)
    assert len(targets) == 6

    # Verify target alignment: for the first window [0..7], next target (offset 1) should be index 8
    assert targets[0] == 8
    # For second window [2..9], next target should be index 10
    assert targets[1] == 10


def test_prepare_train_val_test_splits_end_to_end(tmp_path):
    """Verifies full sequence preparation pipeline, group split, and pipeline report generation."""
    config = load_config()
    df = generate_synthetic_gameplay_data(num_rounds=4, ticks_per_round=120, seed=42)

    out_dir = str(tmp_path / "processed_test")
    train_ds, val_ds, test_ds, report = prepare_train_val_test_splits(df, config, output_dir=out_dir)

    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert len(test_ds) > 0
    assert report["split_counts"]["total_sequences"] == len(train_ds) + len(val_ds) + len(test_ds)
    assert "target_class_counts" in report
