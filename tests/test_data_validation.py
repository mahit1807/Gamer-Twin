"""
Unit tests for data loader and data validation modules.
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.data_loader import load_raw_egocs_data, generate_synthetic_gameplay_data
from src.data_validation import validate_dataset
from src.utils import load_config, PROJECT_CLASSES


def test_missing_dataset_directory_handling():
    """Verifies that non-existent directory raises clear FileNotFoundError."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_raw_egocs_data("non_existent_directory_path_12345")
    assert "does not exist" in str(exc_info.value) or "check data/README.md" in str(exc_info.value)


def test_synthetic_data_generation_fixture():
    """Verifies synthetic data generator output structure and strict test labelling."""
    df = generate_synthetic_gameplay_data(num_rounds=2, ticks_per_round=100, seed=123)

    assert len(df) == 200
    assert "data_source" in df.columns
    assert (df["data_source"] == "SYNTHETIC_TEST_FIXTURE").all()
    assert (df["is_synthetic"] == 1).all()

    # Check essential telemetry fields
    required_fields = ["tick", "seconds", "pos_x", "pos_y", "pos_z", "view_pitch", "view_yaw", "action"]
    for field in required_fields:
        assert field in df.columns


def test_data_validation_report_generation():
    """Verifies that validate_dataset returns correct status, counts, and reports warnings."""
    config = load_config()
    df = generate_synthetic_gameplay_data(num_rounds=2, ticks_per_round=150, seed=42)
    meta = {
        "data_source": "SYNTHETIC_TEST_FIXTURE",
        "is_synthetic": True,
        "raw_record_count": len(df),
        "detected_files": {},
        "columns": list(df.columns)
    }

    is_valid, report = validate_dataset(df, meta, config)

    assert is_valid is True
    assert report["status"] == "PASSED"
    assert report["total_records"] == 300
    assert "target_class_counts" in report
    for cls in PROJECT_CLASSES:
        assert cls in report["target_class_counts"]


def test_data_validation_missing_action_column():
    """Verifies validation failure if critical action column is absent."""
    config = load_config()
    df = pd.DataFrame({"tick": [1, 2, 3], "pos_x": [0, 0, 0]})
    meta = {"data_source": "TEST", "is_synthetic": True}

    is_valid, report = validate_dataset(df, meta, config)
    assert is_valid is False
    assert report["status"] == "FAILED"
    assert any("action" in err for err in report["errors"])
