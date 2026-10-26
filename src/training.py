"""
Training pipeline for GAMER-TWIN deep learning models.
Supports standard supervised training and two-stage Self-Supervised Learning (SSL).
"""

import os
import time
import argparse
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils import setup_logger, load_config, set_seed, get_device, save_json, PROJECT_CLASSES
from src.models import create_model, TransformerSSLNextActionClassifier
from src.sequence_builder import GameplaySequenceDataset, prepare_train_val_test_splits
from src.data_loader import load_raw_egocs_data, generate_synthetic_gameplay_data

logger = setup_logger("Training")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float]:
    """Runs one training epoch and returns (average_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()

        # Gradient clipping to ensure training stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += loss.item() * len(batch_y)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == batch_y).sum().item()
        total += len(batch_y)

    avg_loss = total_loss / max(1, total)
    avg_acc = correct / max(1, total)
    return avg_loss, avg_acc


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float]:
    """Runs one validation epoch and returns (average_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            total_loss += loss.item() * len(batch_y)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)

    avg_loss = total_loss / max(1, total)
    avg_acc = correct / max(1, total)
    return avg_loss, avg_acc


def train_ssl_pretrain_epoch(
    model: TransformerSSLNextActionClassifier,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str
) -> float:
    """
    Stage 1 of Model 4: Trains the model to reconstruct masked frames in past gameplay telemetry.
    Objective: Masked Autoencoder MSE loss on masked features.
    """
    model.train()
    total_loss = 0.0
    total_masked_tokens = 0

    for batch_x, _ in dataloader:
        batch_x = batch_x.to(device)
        optimizer.zero_grad()

        reconstructed, mask = model.forward_ssl_pretrain(batch_x)

        # Compute MSE loss specifically on the masked frames
        if mask.sum() > 0:
            loss = nn.functional.mse_loss(reconstructed[mask], batch_x[mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += loss.item() * mask.sum().item()
            total_masked_tokens += mask.sum().item()

    return total_loss / max(1, total_masked_tokens)


def train_model(
    model_name: str,
    train_dataset: GameplaySequenceDataset,
    val_dataset: GameplaySequenceDataset,
    input_dim: int,
    config: Dict[str, Any],
    checkpoint_dir: str = "outputs/checkpoints",
    epochs: Optional[int] = None,
    progress_callback: Optional[callable] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Trains one of the 4 model variants with early stopping and checkpoint saving.
    """
    set_seed(config.get("project", {}).get("random_seed", 42))
    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 32)
    lr = train_cfg.get("learning_rate", 0.001)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    max_epochs = epochs or train_cfg.get("epochs", 10)
    patience = train_cfg.get("early_stopping_patience", 4)
    device_pref = train_cfg.get("device", "auto")
    device = get_device(device_pref)

    logger.info(f"Initializing {model_name} on device '{device}' (Batch size={batch_size}, LR={lr}, Epochs={max_epochs})...")
    model = create_model(model_name, input_dim, config).to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    history = {
        "model_name": model_name,
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "ssl_pretrain_loss": [],
        "best_epoch": 0,
        "best_val_loss": float("inf")
    }

    # Distinct Stage 1 for Model 4: Self-Supervised Pretraining
    if model_name.lower() in ("transformer_ssl", "ssl"):
        ssl_epochs = config.get("models", {}).get("transformer_ssl", {}).get("ssl_pretrain_epochs", 4)
        ssl_lr = config.get("models", {}).get("transformer_ssl", {}).get("ssl_learning_rate", 0.001)
        logger.info(f"--- Stage 1: Starting Self-Supervised Pretraining ({ssl_epochs} epochs) ---")

        ssl_optimizer = torch.optim.AdamW(model.parameters(), lr=ssl_lr, weight_decay=weight_decay)
        for ssl_ep in range(1, ssl_epochs + 1):
            ep_loss = train_ssl_pretrain_epoch(model, train_loader, ssl_optimizer, device)
            history["ssl_pretrain_loss"].append(round(ep_loss, 5))
            logger.info(f"SSL Pretrain Epoch {ssl_ep}/{ssl_epochs} - Reconstruction Loss: {ep_loss:.5f}")

        logger.info("--- Stage 2: Supervised Fine-Tuning on Next-Action Classification ---")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state_dict = None
    no_improve_count = 0

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_best.pt")

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        epoch_time = time.time() - t0

        history["epochs"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))

        logger.info(
            f"Epoch {epoch:02d}/{max_epochs:02d} [{epoch_time:.1f}s] - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%"
        )

        if progress_callback:
            progress_callback(epoch, max_epochs, train_loss, train_acc, val_loss, val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history["best_epoch"] = epoch
            history["best_val_loss"] = round(best_val_loss, 4)
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve_count = 0

            # Save checkpoint
            checkpoint = {
                "model_name": model_name,
                "model_state_dict": best_state_dict,
                "input_dim": input_dim,
                "config": config,
                "classes": PROJECT_CLASSES,
                "history": history,
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc
            }
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"--> Saved improved checkpoint to {checkpoint_path}")
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs without validation improvement.")
                break

    # Load best weights before returning
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Save training history JSON
    history_path = os.path.join(checkpoint_dir, f"{model_name}_history.json")
    save_json(history, history_path)

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GAMER-TWIN Model")
    parser.add_argument("--model", type=str, default="transformer", choices=["lstm_baseline", "lstm_transformer", "transformer", "transformer_ssl"])
    parser.add_argument("--epochs", type=int, default=8, help="Number of epochs")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw telemetry")
    args = parser.parse_args()

    cfg = load_config()
    try:
        raw_df, _ = load_raw_egocs_data(args.data_dir)
    except FileNotFoundError:
        sample_p = cfg.get("data", {}).get("sample_dir", "data/sample_data")
        ticks_f = os.path.join(sample_p, "ticks.csv")
        if not os.path.exists(ticks_f):
            raw_df = generate_synthetic_gameplay_data(num_rounds=5, ticks_per_round=400, output_dir=sample_p)
        else:
            raw_df = pd.read_csv(ticks_f)

    train_ds, val_ds, test_ds, _ = prepare_train_val_test_splits(raw_df, cfg, "outputs/processed")
    in_dim = train_ds.sequences.shape[2]

    trained_model, train_hist = train_model(
        model_name=args.model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        input_dim=in_dim,
        config=cfg,
        epochs=args.epochs
    )
