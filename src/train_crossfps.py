"""
CrossFPS Training and Held-Out Benchmark Script for GAMER-TWIN.
Extracts 10-DoF telemetry windows from verified CrossFPS clips,
trains LSTM and Transformer sequence models, and evaluates generalization without leakage.
"""

import os
import sys
import glob
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.crossfps_pipeline import (
    CrossFPSSequenceDataset,
    CrossFPSTransformerModel,
    CrossFPSLSTMBaseline,
    CONTINUOUS_CHANNELS,
    DISCRETE_CHANNELS
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = WORKSPACE_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
REPORTS_DIR = OUTPUTS_DIR / "reports"
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def parse_clip_dataframe(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parses a 100-frame actions_raw.parquet into:
      continuous: [100, 4] -> LX, LY, RX, RY
      discrete:   [100, 6] -> RT (Fire), LT (Scope), R3 (Melee), A (Jump), X (Reload), Y (Switch)
    """
    # Extract continuous
    lx = [float(v[0]) if isinstance(v, (list, np.ndarray)) and len(v) > 0 else 0.0 for v in df['j_left']]
    ly = [float(v[1]) if isinstance(v, (list, np.ndarray)) and len(v) > 1 else 0.0 for v in df['j_left']]
    rx = [float(v[0]) if isinstance(v, (list, np.ndarray)) and len(v) > 0 else 0.0 for v in df['j_right']]
    ry = [float(v[1]) if isinstance(v, (list, np.ndarray)) and len(v) > 1 else 0.0 for v in df['j_right']]
    cont = np.stack([lx, ly, rx, ry], axis=-1).astype(np.float32)

    # Extract discrete binary
    rt = df.get('right_trigger', [0]*len(df)).to_numpy(dtype=np.float32)
    lt = df.get('left_trigger', [0]*len(df)).to_numpy(dtype=np.float32)
    r3 = df.get('right_thumb', [0]*len(df)).to_numpy(dtype=np.float32)
    a_btn = df.get('south', [0]*len(df)).to_numpy(dtype=np.float32)
    x_btn = df.get('west', [0]*len(df)).to_numpy(dtype=np.float32)
    y_btn = df.get('north', [0]*len(df)).to_numpy(dtype=np.float32)
    disc = np.stack([rt, lt, r3, a_btn, x_btn, y_btn], axis=-1)

    return cont, disc


def build_crossfps_splits(
    eval_dir: Path,
    window_size: int = 16,
    stride: int = 4
) -> Tuple[CrossFPSSequenceDataset, CrossFPSSequenceDataset, CrossFPSSequenceDataset]:
    """
    Finds all extracted clip folders and builds sequence datasets partitioned by clip ID.
    """
    clip_folders = [p for p in eval_dir.glob("*") if p.is_dir() and (p / "actions_raw.parquet").exists()]
    if not clip_folders:
        raise FileNotFoundError(f"No valid clip folders containing actions_raw.parquet found in {eval_dir}.")

    print(f"Found {len(clip_folders)} clip folders with actions_raw.parquet.")
    random.seed(42)
    random.shuffle(clip_folders)

    # 70% Train, 15% Val, 15% Test by clip
    n_clips = len(clip_folders)
    n_train = max(1, int(0.70 * n_clips))
    n_val = max(1, int(0.15 * n_clips))

    train_clips = clip_folders[:n_train]
    val_clips = clip_folders[n_train:n_train + n_val]
    test_clips = clip_folders[n_train + n_val:]

    def process_clip_list(clips: List[Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
        seqs, y_cont, y_disc, ids, games = [], [], [], [], []
        for c_path in clips:
            try:
                df = pd.read_parquet(c_path / "actions_raw.parquet")
                meta_path = c_path / "metadata.json"
                game = "call_of_duty"
                if meta_path.exists():
                    with open(meta_path) as mf:
                        game = json.load(mf).get("game", "call_of_duty")

                cont, disc = parse_clip_dataframe(df)
                full_features = np.concatenate([cont, disc], axis=-1)  # [100, 10]
                T_total = len(full_features)

                for t in range(0, T_total - window_size - 1, stride):
                    seq = full_features[t:t + window_size]
                    target_c = cont[t + window_size]
                    target_d = disc[t + window_size]

                    seqs.append(seq)
                    y_cont.append(target_c)
                    y_disc.append(target_d)
                    ids.append(c_path.name)
                    games.append(game)
            except Exception as e:
                continue

        return np.array(seqs, dtype=np.float32), np.array(y_cont, dtype=np.float32), np.array(y_disc, dtype=np.float32), ids, games

    X_tr, yc_tr, yd_tr, id_tr, g_tr = process_clip_list(train_clips)
    X_va, yc_va, yd_va, id_va, g_va = process_clip_list(val_clips)
    X_te, yc_te, yd_te, id_te, g_te = process_clip_list(test_clips)

    print(f"Generated Sequences: Train={len(X_tr)}, Val={len(X_va)}, Test={len(X_te)} (Clip-separated, 0% leakage)")

    # Save processed splits for fast API inference
    splits_save_path = OUTPUTS_DIR / "processed" / "crossfps_splits.npz"
    os.makedirs(splits_save_path.parent, exist_ok=True)
    np.savez_compressed(
        splits_save_path,
        X_test=X_te,
        yc_test=yc_te,
        yd_test=yd_te,
        clip_ids=np.array(id_te),
        games=np.array(g_te)
    )

    ds_train = CrossFPSSequenceDataset(X_tr, yc_tr, yd_tr, id_tr, g_tr)
    ds_val = CrossFPSSequenceDataset(X_va, yc_va, yd_va, id_va, g_va)
    ds_test = CrossFPSSequenceDataset(X_te, yc_te, yd_te, id_te, g_te)
    return ds_train, ds_val, ds_test


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 5,
    lr: float = 1e-3,
    device: str = "cpu"
) -> Dict[str, Any]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_criterion = nn.MSELoss()
    bce_criterion = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_loss": []}
    best_loss = float("inf")
    best_state = None

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            seq = batch["sequence"].to(device)
            tc = batch["target_continuous"].to(device)
            td = batch["target_discrete"].to(device)

            optimizer.zero_grad()
            pc, pd = model(seq)
            loss_c = mse_criterion(pc, tc)
            loss_d = bce_criterion(pd, td)
            loss = loss_c + loss_d
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_tr_loss = total_loss / max(1, len(train_loader))

        # Validation
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                seq = batch["sequence"].to(device)
                tc = batch["target_continuous"].to(device)
                td = batch["target_discrete"].to(device)
                pc, pd = model(seq)
                l = mse_criterion(pc, tc) + bce_criterion(pd, td)
                v_loss += l.item()

        avg_val_loss = v_loss / max(1, len(val_loader))
        history["train_loss"].append(round(avg_tr_loss, 4))
        history["val_loss"].append(round(avg_val_loss, 4))

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        print(f"Epoch {ep+1}/{epochs} - Train Loss: {avg_tr_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return {"history": history, "best_val_loss": best_loss}


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: str = "cpu") -> Dict[str, Any]:
    model.eval()
    model.to(device)
    all_pc, all_tc = [], []
    all_pd, all_td = [], []

    mse_loss = nn.MSELoss()
    with torch.no_grad():
        for batch in test_loader:
            seq = batch["sequence"].to(device)
            pc, pd = model(seq)
            all_pc.append(pc.cpu().numpy())
            all_tc.append(batch["target_continuous"].numpy())
            all_pd.append(torch.sigmoid(pd).cpu().numpy())
            all_td.append(batch["target_discrete"].numpy())

    pred_cont = np.concatenate(all_pc, axis=0)
    true_cont = np.concatenate(all_tc, axis=0)
    pred_disc_prob = np.concatenate(all_pd, axis=0)
    true_disc = np.concatenate(all_td, axis=0)

    # 1. Continuous Steering Metrics (MSE)
    cont_mse = float(np.mean((pred_cont - true_cont) ** 2))

    # 2. Discrete Multi-Label Action Metrics
    pred_disc_bin = (pred_disc_prob >= 0.5).astype(int)
    exact_acc = float(accuracy_score(true_disc, pred_disc_bin))
    macro_f1 = float(f1_score(true_disc, pred_disc_bin, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(true_disc, pred_disc_bin, average="weighted", zero_division=0))
    prec_macro = float(precision_score(true_disc, pred_disc_bin, average="macro", zero_division=0))
    rec_macro = float(recall_score(true_disc, pred_disc_bin, average="macro", zero_division=0))

    per_channel = {}
    for i, ch in enumerate(DISCRETE_CHANNELS):
        per_channel[ch] = {
            "f1": round(float(f1_score(true_disc[:, i], pred_disc_bin[:, i], zero_division=0)), 3),
            "precision": round(float(precision_score(true_disc[:, i], pred_disc_bin[:, i], zero_division=0)), 3),
            "recall": round(float(recall_score(true_disc[:, i], pred_disc_bin[:, i], zero_division=0)), 3),
            "positive_rate": round(float(np.mean(true_disc[:, i])), 3)
        }

    return {
        "test_samples": len(true_disc),
        "continuous_mse": round(cont_mse, 4),
        "multi_label_accuracy": round(exact_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "per_action_metrics": per_channel
    }


def run_pipeline():
    eval_dir = WORKSPACE_ROOT / "data" / "CrossFPS-val" / "extracted" / "eval"
    if not eval_dir.exists():
        print(f"Directory {eval_dir} does not exist yet. Please run extraction first.")
        return

    ds_tr, ds_va, ds_te = build_crossfps_splits(eval_dir, window_size=16)
    train_loader = DataLoader(ds_tr, batch_size=32, shuffle=True)
    val_loader = DataLoader(ds_va, batch_size=32, shuffle=False)
    test_loader = DataLoader(ds_te, batch_size=32, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    # 1. Train LSTM Baseline
    print("\n--- Training 1. LSTM Baseline ---")
    lstm = CrossFPSLSTMBaseline(input_dim=10, hidden_dim=64, num_layers=2)
    lstm_res = train_model(lstm, train_loader, val_loader, epochs=5, device=device)
    torch.save(lstm.state_dict(), CHECKPOINTS_DIR / "lstm_crossfps_best.pt")
    lstm_metrics = evaluate_model(lstm, test_loader, device=device)
    print(f"LSTM Test Multi-Label F1: {lstm_metrics['macro_f1']:.4f}, Accuracy: {lstm_metrics['multi_label_accuracy']:.4f}")

    # 2. Train Transformer Model (Primary)
    print("\n--- Training 2. CrossFPS Transformer Model (Primary) ---")
    transformer = CrossFPSTransformerModel(input_dim=10, d_model=64, nhead=4, num_layers=2)
    tf_res = train_model(transformer, train_loader, val_loader, epochs=5, device=device)
    torch.save(transformer.state_dict(), CHECKPOINTS_DIR / "transformer_crossfps_best.pt")
    tf_metrics = evaluate_model(transformer, test_loader, device=device)
    print(f"Transformer Test Multi-Label F1: {tf_metrics['macro_f1']:.4f}, Accuracy: {tf_metrics['multi_label_accuracy']:.4f}")

    # 3. Save Comparative Report
    benchmark_report = {
        "dataset": "CrossFPS 10-DoF Frame-Aligned Controller Telemetry",
        "evaluation_split": "Held-Out Test Set (Clip-Group Split, 0% Sequence Leakage)",
        "models": {
            "lstm_baseline": {
                "name": "2-Layer LSTM Recurrent Baseline",
                "checkpoint": "lstm_crossfps_best.pt",
                "metrics": lstm_metrics,
                "history": lstm_res["history"]
            },
            "transformer_primary": {
                "name": "Transformer Encoder Next-Action Digital Twin",
                "checkpoint": "transformer_crossfps_best.pt",
                "metrics": tf_metrics,
                "history": tf_res["history"]
            }
        }
    }

    with open(REPORTS_DIR / "crossfps_benchmark.json", "w") as f:
        json.dump(benchmark_report, f, indent=2)

    print("\nBenchmark report saved successfully to outputs/reports/crossfps_benchmark.json")


if __name__ == "__main__":
    run_pipeline()
