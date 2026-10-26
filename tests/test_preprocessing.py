"""
Unit tests for feature preprocessing and target class mapping.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.preprocessing import (
    GameplayFeaturePreprocessor,
    map_action_to_target_class,
    ACTION_SYNONYM_MAP,
    EXCLUDED_EVENT_TYPES
)
from src.utils import load_config, PROJECT_CLASSES, CLASS_TO_IDX


def test_action_mapping_and_exclusion_policy():
    """Tests that synonyms correctly map to 9 classes and excluded events return None."""
    # Test valid synonyms
    assert map_action_to_target_class("fire") == "Fire"
    assert map_action_to_target_class("shoot") == "Fire"
    assert map_action_to_target_class("reload") == "Reload"
    assert map_action_to_target_class("in_jump") == "Jump"
    assert map_action_to_target_class("duck") == "Crouch"
    assert map_action_to_target_class("flashbang") == "Grenade"
    assert map_action_to_target_class("Move") == "Move"

    # Test exclusions
    assert map_action_to_target_class("chat") is None
    assert map_action_to_target_class("buy_item") is None
    assert map_action_to_target_class("player_death") is None
    assert map_action_to_target_class(None) is None


def test_preprocessor_zero_leakage_and_transform(tmp_path):
    """
    Verifies that preprocessor parameters are computed strictly on training split
    and that transform reproduces the exact normalization.
    """
    config = load_config()

    # Create dummy train and test data with known distinct distributions
    train_df = pd.DataFrame({
        "pos_x": [10.0, 20.0, 30.0],
        "pos_y": [100.0, 200.0, 300.0],
        "action": ["Move", "Fire", "Reload"]
    })

    test_df = pd.DataFrame({
        "pos_x": [40.0, 50.0],
        "pos_y": [400.0, 500.0],
        "action": ["Jump", "Scope"]
    })

    preproc = GameplayFeaturePreprocessor(config)
    preproc.fit(train_df)

    # Verify fitted parameters are derived purely from train_df
    assert preproc.means["pos_x"] == 20.0
    assert preproc.means["pos_y"] == 200.0

    X_train, y_train, audit_train = preproc.transform(train_df)
    X_test, y_test, audit_test = preproc.transform(test_df)

    assert X_train.shape[0] == 3
    assert X_test.shape[0] == 2
    assert audit_train["usable_records"] == 3
    assert audit_test["usable_records"] == 2

    # Save and reload state
    state_file = str(tmp_path / "preproc_state.json")
    preproc.save_state(state_file)

    loaded_preproc = GameplayFeaturePreprocessor(config)
    loaded_preproc.load_state(state_file)

    assert loaded_preproc.means["pos_x"] == 20.0
    assert loaded_preproc.is_fitted is True
