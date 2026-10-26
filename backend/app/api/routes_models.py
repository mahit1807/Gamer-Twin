"""
Model Lab API routes for GAMER-TWIN.
Supports all 4 model architectures, parameter configurations, live background training,
loss/accuracy tracking, and checkpoint status.
"""

import os
import json
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
import torch

from src.utils import load_config, PROJECT_CLASSES, get_device
from src.training import train_model
from src.data_loader import load_raw_egocs_data
from src.sequence_builder import prepare_train_val_test_splits
from backend.app.schemas.schemas import (
    ModelArchitectureInfo,
    TrainRequest,
    TrainingProgressResponse
)

router = APIRouter()

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# In-memory training manager state
training_state = {
    "is_training": False,
    "model_type": None,
    "current_epoch": 0,
    "total_epochs": 0,
    "train_loss": None,
    "val_loss": None,
    "train_acc": None,
    "val_acc": None,
    "stage": "idle",
    "history": {"epochs": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []},
    "message": "Ready"
}
training_lock = threading.Lock()


MODELS_METADATA = [
    {
        "id": "lstm_baseline",
        "name": "LSTM Baseline",
        "category": "Recurrent Neural Network",
        "description": "2-layer LSTM processing sequential telemetry vectors frame-by-frame. Uses final hidden state for next-action classification.",
        "is_primary": False,
        "default_hyperparameters": {
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout": 0.1,
            "sequence_length": 16,
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 10
        }
    },
    {
        "id": "lstm_transformer",
        "name": "LSTM + Transformer Hybrid",
        "category": "Recurrent + Self-Attention Hybrid",
        "description": "Combines inductive recurrent bias for local micro-movements with a multi-head Transformer encoder to capture long-range tactical context.",
        "is_primary": False,
        "default_hyperparameters": {
            "hidden_dim": 64,
            "lstm_layers": 1,
            "transformer_layers": 1,
            "num_heads": 4,
            "dropout": 0.1,
            "sequence_length": 16,
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 10
        }
    },
    {
        "id": "transformer",
        "name": "Transformer Encoder",
        "category": "Pure Attention Architecture",
        "description": "Primary GAMER-TWIN architecture. Multi-head self-attention with sinusoidal positional encoding and temporal pooling for next-action probability prediction.",
        "is_primary": True,
        "default_hyperparameters": {
            "hidden_dim": 64,
            "num_layers": 2,
            "num_heads": 4,
            "dim_feedforward": 128,
            "dropout": 0.1,
            "sequence_length": 16,
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 10
        }
    },
    {
        "id": "transformer_ssl",
        "name": "Transformer + Self-Supervised Learning",
        "category": "Self-Supervised Pretrained Model",
        "description": "Two-stage learning: Stage 1 pretrains the backbone via masked feature frame reconstruction on historical gameplay; Stage 2 fine-tunes on next-action prediction.",
        "is_primary": False,
        "default_hyperparameters": {
            "hidden_dim": 64,
            "num_layers": 2,
            "num_heads": 4,
            "ssl_pretrain_epochs": 4,
            "dropout": 0.1,
            "sequence_length": 16,
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 10
        }
    }
]


@router.get("", response_model=List[ModelArchitectureInfo])
def list_models():
    ckpt_dir = WORKSPACE_ROOT / "outputs" / "checkpoints"
    results = []

    for m in MODELS_METADATA:
        crossfps_ckpt = ckpt_dir / f"{m['id']}_crossfps_best.pt"
        std_ckpt = ckpt_dir / f"{m['id']}_best.pt"
        ckpt_path = crossfps_ckpt if crossfps_ckpt.exists() else std_ckpt
        has_ckpt = ckpt_path.exists()
        results.append(
            ModelArchitectureInfo(
                id=m["id"],
                name=m["name"],
                category=m["category"],
                description=m["description"],
                is_primary=m["is_primary"],
                checkpoint_available=has_ckpt,
                checkpoint_path=str(ckpt_path.relative_to(WORKSPACE_ROOT)) if has_ckpt else None,
                default_hyperparameters=m["default_hyperparameters"]
            )
        )
    return results


@router.get("/training-status", response_model=TrainingProgressResponse)
def get_training_status():
    with training_lock:
        return TrainingProgressResponse(
            is_training=training_state["is_training"],
            model_type=training_state["model_type"],
            current_epoch=training_state["current_epoch"],
            total_epochs=training_state["total_epochs"],
            train_loss=training_state["train_loss"],
            val_loss=training_state["val_loss"],
            train_acc=training_state["train_acc"],
            val_acc=training_state["val_acc"],
            stage=training_state["stage"],
            history=training_state["history"],
            message=training_state["message"]
        )


def _execute_training_job(model_type: str, epochs: int, batch_size: int, lr: float, seq_len: int):
    global training_state
    try:
        config = load_config()
        # Override config parameters
        config["training"]["epochs"] = epochs
        config["training"]["batch_size"] = batch_size
        config["training"]["learning_rate"] = lr
        config["sequence"]["window_size"] = seq_len

        # Determine dataset source
        raw_dir = WORKSPACE_ROOT / "data" / "raw"
        sample_dir = WORKSPACE_ROOT / "data" / "sample_data"
        data_dir = str(raw_dir if raw_dir.exists() and any(raw_dir.glob("*.csv")) else sample_dir)

        with training_lock:
            training_state["stage"] = "loading_data"
            training_state["message"] = f"Loading dataset from {data_dir}..."

        df, meta = load_raw_egocs_data(data_dir)
        splits, preprocessor = prepare_train_val_test_splits(
            df=df,
            config=config,
            output_dir=str(WORKSPACE_ROOT / "outputs" / "processed")
        )

        input_dim = preprocessor.feature_dim

        def on_epoch_progress(epoch_idx, t_loss, v_loss, t_acc, v_acc):
            with training_lock:
                training_state["current_epoch"] = epoch_idx
                training_state["train_loss"] = round(float(t_loss), 4)
                training_state["val_loss"] = round(float(v_loss), 4)
                training_state["train_acc"] = round(float(t_acc * 100), 2)
                training_state["val_acc"] = round(float(v_acc * 100), 2)
                training_state["history"]["epochs"].append(epoch_idx)
                training_state["history"]["train_loss"].append(round(float(t_loss), 4))
                training_state["history"]["val_loss"].append(round(float(v_loss), 4))
                training_state["history"]["train_acc"].append(round(float(t_acc * 100), 2))
                training_state["history"]["val_acc"].append(round(float(v_acc * 100), 2))
                training_state["message"] = f"Epoch {epoch_idx}/{epochs}: Train Loss {t_loss:.4f}, Val Loss {v_loss:.4f}"

        with training_lock:
            training_state["stage"] = "training"
            training_state["message"] = f"Training {model_type} for {epochs} epochs..."

        model, history = train_model(
            model_name=model_type,
            train_dataset=splits["train"],
            val_dataset=splits["val"],
            input_dim=input_dim,
            config=config,
            checkpoint_dir=str(WORKSPACE_ROOT / "outputs" / "checkpoints"),
            epochs=epochs,
            progress_callback=on_epoch_progress
        )

        with training_lock:
            training_state["is_training"] = False
            training_state["stage"] = "completed"
            training_state["message"] = f"Training completed successfully! Saved checkpoint for {model_type}."

    except Exception as e:
        with training_lock:
            training_state["is_training"] = False
            training_state["stage"] = "error"
            training_state["message"] = f"Training failed: {str(e)}"


@router.post("/train")
def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    global training_state
    with training_lock:
        if training_state["is_training"]:
            raise HTTPException(status_code=400, detail="A training job is already running.")

        training_state["is_training"] = True
        training_state["model_type"] = req.model_type
        training_state["current_epoch"] = 0
        training_state["total_epochs"] = req.epochs
        training_state["train_loss"] = None
        training_state["val_loss"] = None
        training_state["train_acc"] = None
        training_state["val_acc"] = None
        training_state["stage"] = "initializing"
        training_state["history"] = {"epochs": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        training_state["message"] = f"Initiating training for {req.model_type}..."

    # Run in background thread
    thread = threading.Thread(
        target=_execute_training_job,
        args=(req.model_type, req.epochs, req.batch_size, req.learning_rate, req.sequence_length),
        daemon=True
    )
    thread.start()

    return {"status": "started", "model_type": req.model_type, "epochs": req.epochs}


@router.get("/history/{model_name}")
def get_model_history(model_name: str):
    history_file = WORKSPACE_ROOT / "outputs" / "checkpoints" / f"{model_name}_history.json"
    if not history_file.exists():
        raise HTTPException(status_code=404, detail=f"No training history file found for model '{model_name}'.")

    with open(history_file, "r") as f:
        data = json.load(f)
    return data
