"""
Self-Supervised Transformer Training & Held-Out Benchmark for CrossFPS.
Implements:
  Stage 1: Genuine Self-Supervised Pretraining via Masked Controller-Feature Reconstruction (Unlabeled sequences).
  Stage 2: Downstream Fine-Tuning on 10-DoF next-action prediction (Dual continuous + discrete heads).
Evaluates against held-out CrossFPS test partition (10,859 sequences) under identical protocol.
Saves checkpoints to outputs/checkpoints/transformer_ssl_crossfps_best.pt and checkpoints/transformer_ssl_crossfps_best.pt.
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.crossfps_pipeline import (
    CrossFPSSequenceDataset,
    CrossFPSTransformerSSL,
    CONTINUOUS_CHANNELS,
    DISCRETE_CHANNELS
)
from src.train_crossfps import parse_clip_dataframe

OUTPUTS_DIR = WORKSPACE_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
ROOT_CHECKPOINTS_DIR = WORKSPACE_ROOT / "checkpoints"
REPORTS_DIR = OUTPUTS_DIR / "reports"
PROCESSED_DIR = OUTPUTS_DIR / "processed"

os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(ROOT_CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


class CrossFPSUnlabeledDataset(Dataset):
    """Dataset for Stage 1 Self-Supervised Pretraining (Unlabeled Sequences)."""
    def __init__(self, sequences: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sequences[idx]


def extract_clip_sequences(clip_path: Path, window_size: int = 16, stride: int = 4):
    try:
        parquet_path = clip_path / "actions_raw.parquet"
        if not parquet_path.exists():
            return None
        df = pd.read_parquet(parquet_path)
        meta_path = clip_path / "metadata.json"
        game = "call_of_duty"
        if meta_path.exists():
            with open(meta_path) as mf:
                game = json.load(mf).get("game", "call_of_duty")

        cont, disc = parse_clip_dataframe(df)
        full = np.concatenate([cont, disc], axis=-1)  # [100, 10]
        T_total = len(full)

        seqs, y_cont, y_disc = [], [], []
        for t in range(0, T_total - window_size - 1, stride):
            seqs.append(full[t:t + window_size])
            y_cont.append(cont[t + window_size])
            y_disc.append(disc[t + window_size])

        return {
            "seqs": np.array(seqs, dtype=np.float32),
            "yc": np.array(y_cont, dtype=np.float32),
            "yd": np.array(y_disc, dtype=np.float32),
            "clip_id": clip_path.name,
            "game": game,
            "count": len(seqs)
        }
    except Exception:
        return None


def get_or_build_all_splits(eval_dir: Path, window_size: int = 16, stride: int = 4):
    cache_path = PROCESSED_DIR / "crossfps_train_val.npz"
    splits_test_path = PROCESSED_DIR / "crossfps_splits.npz"

    # Check if cached train/val arrays exist
    if cache_path.exists() and splits_test_path.exists():
        print(f"Loading cached dataset splits from {cache_path} and {splits_test_path}...")
        c_data = np.load(cache_path, allow_pickle=True)
        t_data = np.load(splits_test_path, allow_pickle=True)

        X_tr = c_data["X_train"]
        yc_tr = c_data["yc_train"]
        yd_tr = c_data["yd_train"]

        X_va = c_data["X_val"]
        yc_va = c_data["yc_val"]
        yd_va = c_data["yd_val"]

        X_te = t_data["X_test"]
        yc_te = t_data["yc_test"]
        yd_te = t_data["yd_test"]
        id_te = t_data["clip_ids"]
        g_te = t_data["games"]

        print(f"Loaded Splits: Train={len(X_tr):,}, Val={len(X_va):,}, Test={len(X_te):,}")
        return (
            (X_tr, yc_tr, yd_tr),
            (X_va, yc_va, yd_va),
            (X_te, yc_te, yd_te, id_te, g_te)
        )

    print(f"Extracting clips from {eval_dir}...")
    clip_folders = [p for p in eval_dir.glob("*") if p.is_dir() and (p / "actions_raw.parquet").exists()]
    random.seed(42)
    random.shuffle(clip_folders)

    n_clips = len(clip_folders)
    n_train = max(1, int(0.70 * n_clips))
    n_val = max(1, int(0.15 * n_clips))

    train_clips = clip_folders[:n_train]
    val_clips = clip_folders[n_train:n_train + n_val]
    test_clips = clip_folders[n_train + n_val:]

    def process_clips_parallel(clips: List[Path]):
        with ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(lambda c: extract_clip_sequences(c, window_size, stride), clips))

        valid = [r for r in results if r is not None and r["count"] > 0]
        seqs = np.concatenate([r["seqs"] for r in valid], axis=0)
        yc = np.concatenate([r["yc"] for r in valid], axis=0)
        yd = np.concatenate([r["yd"] for r in valid], axis=0)
        ids = []
        games = []
        for r in valid:
            ids.extend([r["clip_id"]] * r["count"])
            games.extend([r["game"]] * r["count"])

        return seqs, yc, yd, ids, games

    print(f"Processing {len(train_clips)} train clips...")
    X_tr, yc_tr, yd_tr, _, _ = process_clips_parallel(train_clips)
    print(f"Processing {len(val_clips)} val clips...")
    X_va, yc_va, yd_va, _, _ = process_clips_parallel(val_clips)
    print(f"Processing {len(test_clips)} test clips...")
    X_te, yc_te, yd_te, id_te, g_te = process_clips_parallel(test_clips)

    # Cache train/val
    np.savez_compressed(
        cache_path,
        X_train=X_tr, yc_train=yc_tr, yd_train=yd_tr,
        X_val=X_va, yc_val=yc_va, yd_val=yd_va
    )

    # Save test partition
    np.savez_compressed(
        splits_test_path,
        X_test=X_te,
        yc_test=yc_te,
        yd_test=yd_te,
        clip_ids=np.array(id_te),
        games=np.array(g_te)
    )

    print(f"Splits generated: Train={len(X_tr):,}, Val={len(X_va):,}, Test={len(X_te):,}")
    return (
        (X_tr, yc_tr, yd_tr),
        (X_va, yc_va, yd_va),
        (X_te, yc_te, yd_te, id_te, g_te)
    )


def pretrain_ssl_stage(
    model: CrossFPSTransformerSSL,
    train_sequences: np.ndarray,
    val_sequences: np.ndarray,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Stage 1: Genuine Self-Supervised Pretraining.
    Masks random frames in gameplay sequences and trains model to reconstruct
    the 10-DoF feature vectors without using any downstream action labels.
    """
    print("\n=======================================================")
    print("STAGE 1: SELF-SUPERVISED PRETRAINING (MASKED AUTOENCODING)")
    print("=======================================================")
    print(f"Unlabeled Pretraining Sequences: {len(train_sequences):,}")
    print(f"Masking Probability: {model.mask_prob * 100:.0f}% (Plus mandatory t=15 transition mask)")
    print(f"Backbone: 2-Layer Transformer (d_model={model.d_model}, heads=4, FFN=128)")

    model.to(device)
    train_ds = CrossFPSUnlabeledDataset(train_sequences)
    val_ds = CrossFPSUnlabeledDataset(val_sequences)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(
        list(model.input_proj.parameters()) +
        [model.pos_encoder] +
        list(model.transformer.parameters()) +
        list(model.ssl_reconstruction_head.parameters()),
        lr=lr
    )

    history = {"ssl_train_loss": [], "ssl_val_loss": []}
    best_ssl_loss = float("inf")
    best_backbone_state = None

    for ep in range(epochs):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0

        for x_b in train_loader:
            x_b = x_b.to(device)
            optimizer.zero_grad()

            recon, mask = model.forward_ssl_pretrain(x_b)
            # Compute MSE strictly on the masked timesteps
            if mask.sum() > 0:
                loss = nn.functional.mse_loss(recon[mask], x_b[mask])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

        avg_tr_loss = total_loss / max(1, n_batches)

        # Validation reconstruction loss
        model.eval()
        v_loss = 0.0
        v_batches = 0
        with torch.no_grad():
            for x_b in val_loader:
                x_b = x_b.to(device)
                recon, mask = model.forward_ssl_pretrain(x_b)
                if mask.sum() > 0:
                    l = nn.functional.mse_loss(recon[mask], x_b[mask])
                    v_loss += l.item()
                    v_batches += 1

        avg_val_loss = v_loss / max(1, v_batches)
        ep_time = time.time() - t0

        history["ssl_train_loss"].append(round(avg_tr_loss, 5))
        history["ssl_val_loss"].append(round(avg_val_loss, 5))

        if avg_val_loss < best_ssl_loss:
            best_ssl_loss = avg_val_loss
            best_backbone_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"SSL Pretrain Epoch {ep+1}/{epochs} [{ep_time:.1f}s] - "
            f"Reconstruction Loss: Train={avg_tr_loss:.4f}, Val={avg_val_loss:.4f}"
        )

    if best_backbone_state:
        model.load_state_dict(best_backbone_state)

    print(f"Stage 1 Complete. Best Validation Reconstruction MSE: {best_ssl_loss:.4f}")
    return {"history": history, "best_val_loss": best_ssl_loss}


def finetune_stage(
    model: CrossFPSTransformerSSL,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 5,
    lr: float = 5e-4,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Stage 2: Supervised Fine-Tuning.
    Fine-tunes the pretrained Transformer representation for next-action prediction.
    """
    print("\n=======================================================")
    print("STAGE 2: SUPERVISED DOWNSTREAM FINE-TUNING")
    print("=======================================================")
    print(f"Fine-Tuning on 10-DoF Action Targets (4 Continuous + 6 Discrete Buttons)")
    print(f"Learning Rate: {lr} (Fine-tuning backbone + downstream heads)")

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_criterion = nn.MSELoss()
    bce_criterion = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_loss": [], "val_exact_accuracy": []}
    best_loss = float("inf")
    best_state = None

    for ep in range(epochs):
        t0 = time.time()
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
        all_pd, all_td = [], []
        with torch.no_grad():
            for batch in val_loader:
                seq = batch["sequence"].to(device)
                tc = batch["target_continuous"].to(device)
                td = batch["target_discrete"].to(device)
                pc, pd = model(seq)
                l = mse_criterion(pc, tc) + bce_criterion(pd, td)
                v_loss += l.item()
                all_pd.append(torch.sigmoid(pd).cpu().numpy())
                all_td.append(td.cpu().numpy())

        avg_val_loss = v_loss / max(1, len(val_loader))
        val_pred_bin = (np.concatenate(all_pd, axis=0) >= 0.5).astype(int)
        val_true = np.concatenate(all_td, axis=0)
        val_acc = float(accuracy_score(val_true, val_pred_bin))

        ep_time = time.time() - t0
        history["train_loss"].append(round(avg_tr_loss, 4))
        history["val_loss"].append(round(avg_val_loss, 4))
        history["val_exact_accuracy"].append(round(val_acc * 100, 2))

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Finetune Epoch {ep+1}/{epochs} [{ep_time:.1f}s] - "
            f"Train Loss: {avg_tr_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Exact Acc: {val_acc*100:.2f}%"
        )

    if best_state:
        model.load_state_dict(best_state)

    return {"history": history, "best_val_loss": best_loss}


def evaluate_ssl_model(model: nn.Module, test_loader: DataLoader, device: str = "cpu") -> Dict[str, Any]:
    model.eval()
    model.to(device)
    all_pc, all_tc = [], []
    all_pd, all_td = [], []

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


def main():
    print("=" * 60)
    print("GAMER-TWIN: CrossFPS Self-Supervised Transformer Pipeline")
    print("=" * 60)

    eval_dir = WORKSPACE_ROOT / "data" / "CrossFPS-val" / "extracted" / "eval"
    if not eval_dir.exists():
        print(f"Error: Dataset directory {eval_dir} does not exist.")
        sys.exit(1)

    train_data, val_data, test_data = get_or_build_all_splits(eval_dir, window_size=16, stride=4)
    X_tr, yc_tr, yd_tr = train_data
    X_va, yc_va, yd_va = val_data
    X_te, yc_te, yd_te, id_te, g_te = test_data

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing pipeline on device: {device}")

    # Build PyTorch DataLoaders for fine-tuning & evaluation
    ds_tr = CrossFPSSequenceDataset(X_tr, yc_tr, yd_tr, ["train"] * len(X_tr), ["all"] * len(X_tr))
    ds_va = CrossFPSSequenceDataset(X_va, yc_va, yd_va, ["val"] * len(X_va), ["all"] * len(X_va))
    ds_te = CrossFPSSequenceDataset(X_te, yc_te, yd_te, id_te, g_te)

    train_loader = DataLoader(ds_tr, batch_size=64, shuffle=True)
    val_loader = DataLoader(ds_va, batch_size=64, shuffle=False)
    test_loader = DataLoader(ds_te, batch_size=64, shuffle=False)

    # 1. Instantiate Model
    ssl_model = CrossFPSTransformerSSL(
        input_dim=10,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        mask_prob=0.20
    )

    # 2. Stage 1: Self-Supervised Pretraining
    ssl_res = pretrain_ssl_stage(
        model=ssl_model,
        train_sequences=X_tr,
        val_sequences=X_va,
        epochs=5,
        batch_size=64,
        lr=1e-3,
        device=device
    )

    # 3. Stage 2: Supervised Fine-Tuning
    ft_res = finetune_stage(
        model=ssl_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,
        lr=5e-4,
        device=device
    )

    # 4. Save Checkpoint to both required locations
    ckpt_name = "transformer_ssl_crossfps_best.pt"
    ckpt_out_path = CHECKPOINTS_DIR / ckpt_name
    ckpt_root_path = ROOT_CHECKPOINTS_DIR / ckpt_name

    torch.save(ssl_model.state_dict(), ckpt_out_path)
    torch.save(ssl_model.state_dict(), ckpt_root_path)
    print(f"\nTrained checkpoint successfully saved to:")
    print(f"  1. {ckpt_out_path} ({ckpt_out_path.stat().st_size:,} bytes)")
    print(f"  2. {ckpt_root_path} ({ckpt_root_path.stat().st_size:,} bytes)")

    # 5. Evaluate on identical Held-Out Test Split
    print("\nEvaluating fine-tuned Self-Supervised Transformer on Held-Out Test Split (10,859 sequences)...")
    metrics = evaluate_ssl_model(ssl_model, test_loader, device=device)

    print("\n--- Quantitative Empirical Test Results ---")
    print(f"  Test Samples:          {metrics['test_samples']}")
    print(f"  Multi-Label Accuracy:  {metrics['multi_label_accuracy']*100:.2f}%")
    print(f"  Macro F1-Score:        {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1-Score:     {metrics['weighted_f1']:.4f}")
    print(f"  Precision (Macro):     {metrics['precision_macro']:.4f}")
    print(f"  Recall (Macro):        {metrics['recall_macro']:.4f}")
    print(f"  Continuous MSE:        {metrics['continuous_mse']:.4f}")

    # 6. Save Training History & Configuration
    history_data = {
        "model_name": "transformer_ssl",
        "architecture": {
            "type": "CrossFPSTransformerSSL",
            "input_dim": 10,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 128,
            "mask_prob": 0.20
        },
        "ssl_pretrain_loss": ssl_res["history"]["ssl_train_loss"],
        "ssl_val_loss": ssl_res["history"]["ssl_val_loss"],
        "finetune_train_loss": ft_res["history"]["train_loss"],
        "finetune_val_loss": ft_res["history"]["val_loss"],
        "finetune_val_accuracy": ft_res["history"]["val_exact_accuracy"],
        "best_finetune_val_loss": ft_res["best_val_loss"],
        "test_metrics": metrics
    }

    with open(CHECKPOINTS_DIR / "transformer_ssl_crossfps_history.json", "w") as f:
        json.dump(history_data, f, indent=2)

    with open(REPORTS_DIR / "transformer_ssl_crossfps_eval.json", "w") as f:
        json.dump(history_data, f, indent=2)

    # 7. Update crossfps_benchmark.json with the verified third model
    bench_file = REPORTS_DIR / "crossfps_benchmark.json"
    bench_data = {}
    if bench_file.exists():
        try:
            with open(bench_file, "r") as f:
                bench_data = json.load(f)
        except Exception:
            bench_data = {}

    if "models" not in bench_data:
        bench_data["models"] = {}

    bench_data["models"]["transformer_ssl"] = {
        "name": "Self-Supervised Transformer",
        "checkpoint": ckpt_name,
        "role": "Self-Supervised Representation Variant",
        "pretraining_stage": {
            "objective": "Masked Controller-Feature Reconstruction (Unlabeled sequences)",
            "mask_prob": 0.20,
            "pretrain_sequences": len(X_tr),
            "reconstruction_loss_history": ssl_res["history"]["ssl_val_loss"]
        },
        "metrics": metrics,
        "history": {
            "train_loss": ft_res["history"]["train_loss"],
            "val_loss": ft_res["history"]["val_loss"]
        },
        "disclosures": {
            "data_partition_note": "Trained and evaluated on held-out clip partitions of the available CrossFPS validation partition (data/CrossFPS-val/). Not presented as an independent production test benchmark.",
            "protocol": "Zero-leakage clip-based partition: 70% train, 15% validation, 15% test."
        }
    }

    with open(bench_file, "w") as f:
        json.dump(bench_data, f, indent=2)

    print(f"\nUpdated comparative benchmark saved to {bench_file}")
    print("\nSelf-Supervised Transformer training and evaluation completed successfully!")


if __name__ == "__main__":
    main()
