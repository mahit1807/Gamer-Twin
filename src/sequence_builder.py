"""
Sequence window construction and group-based dataset splitting module for GAMER-TWIN.
Ensures zero data leakage across round/session boundaries and evaluation splits.
"""

import os
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils import setup_logger, save_json, load_config, PROJECT_CLASSES, CLASS_TO_IDX
from src.preprocessing import GameplayFeaturePreprocessor

logger = setup_logger("SequenceBuilder")


class GameplaySequenceDataset(Dataset):
    """PyTorch Dataset wrapping sliding sequence windows and target action labels."""
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        """
        Args:
            sequences: Shape (N, window_size, num_features), float32
            targets: Shape (N,), int64
        """
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


def create_sequences_for_group(
    X_group: np.ndarray,
    y_group: np.ndarray,
    window_size: int = 16,
    stride: int = 2,
    target_offset: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs sliding sequence windows from contiguous gameplay records of a single round/session.
    Next-action target is the action at index (i + window_size - 1 + target_offset).
    """
    sequences = []
    targets = []
    n_samples = len(X_group)

    for start_idx in range(0, n_samples - window_size - target_offset + 1, stride):
        end_idx = start_idx + window_size
        target_idx = end_idx + target_offset - 1

        seq_window = X_group[start_idx:end_idx]
        target_action = y_group[target_idx]

        sequences.append(seq_window)
        targets.append(target_action)

    if not sequences:
        return np.empty((0, window_size, X_group.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.int64)


def prepare_train_val_test_splits(
    df: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: Optional[str] = None
) -> Tuple[GameplaySequenceDataset, GameplaySequenceDataset, GameplaySequenceDataset, Dict[str, Any]]:
    """
    End-to-end data preparation pipeline:
    1. Group-based split by round_id to prevent temporal leakage across splits.
    2. Fits feature preprocessor ONLY on training data.
    3. Transforms train, val, and test data independently.
    4. Constructs sequence windows bounded within rounds.
    5. Saves pipeline report and preprocessor artifacts.
    """
    seq_cfg = config.get("sequence", {})
    window_size = seq_cfg.get("window_size", 16)
    stride = seq_cfg.get("stride", 2)
    target_offset = seq_cfg.get("target_offset", 1)

    split_cfg = config.get("split", {})
    train_ratio = split_cfg.get("train_ratio", 0.70)
    val_ratio = split_cfg.get("val_ratio", 0.15)
    split_strategy = split_cfg.get("split_strategy", "round_id")

    if split_strategy not in df.columns:
        logger.warning(f"Split column '{split_strategy}' not found. Using 'player_id' or synthetic rounds.")
        if "player_id" in df.columns:
            split_strategy = "player_id"
        else:
            # Create synthetic round chunks if neither round_id nor player_id exist
            df["round_id"] = ["round_" + str(i // 400) for i in range(len(df))]
            split_strategy = "round_id"

    # Identify unique groups (e.g. rounds)
    unique_groups = df[split_strategy].unique()
    np.random.seed(config.get("project", {}).get("random_seed", 42))
    shuffled_groups = np.random.permutation(unique_groups)

    n_groups = len(shuffled_groups)
    if n_groups >= 3:
        n_train = max(1, int(round(train_ratio * n_groups)))
        n_val = max(1, int(round(val_ratio * n_groups)))
        train_groups = set(shuffled_groups[:n_train])
        val_groups = set(shuffled_groups[n_train:n_train + n_val])
        test_groups = set(shuffled_groups[n_train + n_val:])
        if not test_groups:
            test_groups = set(shuffled_groups[-1:])
            train_groups.discard(list(test_groups)[0])
    else:
        # Fallback for single match: split chronologically by index range to preserve temporal ordering
        logger.info("Fewer than 3 groups available; performing chronological block split.")
        n_records = len(df)
        t_split = int(train_ratio * n_records)
        v_split = int((train_ratio + val_ratio) * n_records)
        df_train = df.iloc[:t_split].copy()
        df_val = df.iloc[t_split:v_split].copy()
        df_test = df.iloc[v_split:].copy()
        train_groups, val_groups, test_groups = None, None, None

    if train_groups is not None:
        df_train = df[df[split_strategy].isin(train_groups)].copy()
        df_val = df[df[split_strategy].isin(val_groups)].copy()
        df_test = df[df[split_strategy].isin(test_groups)].copy()

    logger.info(
        f"Data split sizes: Train={len(df_train)} rows ({len(train_groups) if train_groups else 'block'} groups), "
        f"Val={len(df_val)} rows, Test={len(df_test)} rows."
    )

    # ZERO DATA LEAKAGE: Fit preprocessor strictly on train set
    preprocessor = GameplayFeaturePreprocessor(config)
    preprocessor.fit(df_train)

    # Transform each split
    X_train_raw, y_train_raw, audit_train = preprocessor.transform(df_train)
    X_val_raw, y_val_raw, audit_val = preprocessor.transform(df_val)
    X_test_raw, y_test_raw, audit_test = preprocessor.transform(df_test)

    # Build sequence windows within round groups to avoid cross-round contamination
    def extract_sequences(raw_df: pd.DataFrame, preproc: GameplayFeaturePreprocessor):
        seqs_list = []
        targets_list = []
        for _, group_rows in raw_df.groupby(split_strategy, sort=False):
            if len(group_rows) < window_size + target_offset:
                continue
            try:
                X_g, y_g, _ = preproc.transform(group_rows)
                s_g, t_g = create_sequences_for_group(X_g, y_g, window_size, stride, target_offset)
                if len(s_g) > 0:
                    seqs_list.append(s_g)
                    targets_list.append(t_g)
            except ValueError:
                continue
        if seqs_list:
            return np.vstack(seqs_list), np.concatenate(targets_list)
        return np.empty((0, window_size, len(preproc.feature_names)), dtype=np.float32), np.empty((0,), dtype=np.int64)

    X_train_seq, y_train_seq = extract_sequences(df_train, preprocessor)
    X_val_seq, y_val_seq = extract_sequences(df_val, preprocessor)
    X_test_seq, y_test_seq = extract_sequences(df_test, preprocessor)

    # Check class counts
    def count_classes(targets_array: np.ndarray) -> Dict[str, int]:
        counts = {cls: 0 for cls in PROJECT_CLASSES}
        if len(targets_array) > 0:
            unique, u_counts = np.unique(targets_array, return_counts=True)
            for idx, count in zip(unique, u_counts):
                if 0 <= idx < len(PROJECT_CLASSES):
                    counts[PROJECT_CLASSES[idx]] = int(count)
        return counts

    train_class_counts = count_classes(y_train_seq)
    val_class_counts = count_classes(y_val_seq)
    test_class_counts = count_classes(y_test_seq)

    pipeline_report = {
        "sequence_parameters": {
            "window_size": int(window_size),
            "stride": int(stride),
            "target_offset": int(target_offset),
            "feature_dim": int(X_train_seq.shape[2]) if len(X_train_seq) > 0 else 0
        },
        "split_counts": {
            "train_sequences": int(len(X_train_seq)),
            "val_sequences": int(len(X_val_seq)),
            "test_sequences": int(len(X_test_seq)),
            "total_sequences": int(len(X_train_seq) + len(X_val_seq) + len(X_test_seq))
        },
        "records_audit": {
            "train": audit_train,
            "val": audit_val,
            "test": audit_test
        },
        "target_class_counts": {
            "train": train_class_counts,
            "val": val_class_counts,
            "test": test_class_counts
        }
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        preprocessor.save_state(os.path.join(output_dir, "preprocessor_state.json"))
        save_json(pipeline_report, os.path.join(output_dir, "pipeline_report.json"))
        # Save preprocessed sequence arrays for quick loading
        np.savez_compressed(
            os.path.join(output_dir, "processed_splits.npz"),
            X_train=X_train_seq, y_train=y_train_seq,
            X_val=X_val_seq, y_val=y_val_seq,
            X_test=X_test_seq, y_test=y_test_seq
        )
        logger.info(f"Processed sequences and pipeline report saved to {output_dir}")

    train_ds = GameplaySequenceDataset(X_train_seq, y_train_seq)
    val_ds = GameplaySequenceDataset(X_val_seq, y_val_seq)
    test_ds = GameplaySequenceDataset(X_test_seq, y_test_seq)

    return train_ds, val_ds, test_ds, pipeline_report
