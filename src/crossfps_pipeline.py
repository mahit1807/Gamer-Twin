"""
CrossFPS Telemetry Dataset Pipeline & Sequence Modeling Core for GAMER-TWIN.
Supports extraction of 10-DoF gamepad controller telemetry (LX, LY, RX, RY, RT, LT, R3, A, X, Y),
sliding sequence windowing (T=16), and multi-task next-action prediction across 7 FPS titles.
"""

import os
import glob
import json
import tarfile
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

# 10-DoF Channel Taxonomy
CONTINUOUS_CHANNELS = ["LX", "LY", "RX", "RY"]  # Joysticks [-1, 1]
DISCRETE_CHANNELS = ["RT", "LT", "R3", "A", "X", "Y"]  # Fire, Scope/ADS, Melee, Jump, Reload, Switch

CHANNEL_LABELS = {
    "RT": "Fire (Right Trigger)",
    "LT": "Aim / Scope (Left Trigger)",
    "R3": "Melee (Right Stick Click)",
    "A": "Jump (Button A)",
    "X": "Reload (Button X)",
    "Y": "Switch Weapon (Button Y)",
    "LX": "Strafe Left / Right",
    "LY": "Move Forward / Back",
    "RX": "Turn Camera Yaw",
    "RY": "Look Camera Pitch",
}


class ChainedTarStream:
    """
    Combines split tar chunks (cross-fps-eval_chunk0000.tar, chunk0001.tar)
    into a continuous binary stream for direct in-memory extraction.
    """
    def __init__(self, file_paths: List[Path]):
        self.file_paths = file_paths
        self.idx = 0
        self.current_f = open(file_paths[0], "rb") if file_paths else None

    def read(self, size: int = 65536) -> bytes:
        if not self.current_f:
            return b""
        data = self.current_f.read(size)
        if not data:
            self.current_f.close()
            self.idx += 1
            if self.idx < len(self.file_paths):
                self.current_f = open(self.file_paths[self.idx], "rb")
                return self.read(size)
            else:
                self.current_f = None
                return b""
        return data


def extract_eval_chunks(val_dir: Optional[Path] = None, extract_dir: Optional[Path] = None, max_clips: Optional[int] = None) -> int:
    """
    Extracts split cross-fps-eval_chunk*.tar files into extract_dir using streaming reassembly.
    Returns the number of clips successfully extracted.
    """
    val_dir = val_dir or (WORKSPACE_ROOT / "data" / "CrossFPS-val")
    extract_dir = extract_dir or (val_dir / "extracted")
    os.makedirs(extract_dir, exist_ok=True)

    tar_files = sorted(list(val_dir.glob("cross-fps-eval_chunk*.tar")))
    if not tar_files:
        print(f"No cross-fps-eval_chunk*.tar files found in {val_dir}.")
        return 0

    print(f"Streaming and extracting {len(tar_files)} split chunks into {extract_dir}...")
    stream = ChainedTarStream(tar_files)
    extracted_clips = set()
    try:
        with tarfile.open(fileobj=stream, mode="r|") as tar:
            for member in tar:
                # Filter to extract telemetry parquets and metadata
                tar.extract(member, path=extract_dir)
                clip_part = member.name.split("/")[0] if "/" in member.name else member.name
                extracted_clips.add(clip_part)
                if max_clips and len(extracted_clips) >= max_clips:
                    print(f"Reached requested limit of {max_clips} clips.")
                    break
    except Exception as e:
        print(f"Streaming extraction notice: {e}")

    print(f"Extraction complete: {len(extracted_clips)} clips extracted.")
    return len(extracted_clips)


def load_clip_telemetry(clip_dir: Path) -> Optional[pd.DataFrame]:
    """
    Loads actions_raw.parquet or actions.parquet from a single clip directory.
    """
    parquet_path = clip_dir / "actions_raw.parquet"
    if not parquet_path.exists():
        parquet_path = clip_dir / "actions.parquet"
    if not parquet_path.exists():
        return None
    try:
        df = pd.read_parquet(parquet_path)
        return df
    except Exception:
        return None


class CrossFPSSequenceDataset(Dataset):
    """
    PyTorch Dataset building historical sequence windows (T=16) from 10-DoF telemetry.
    Input: [T, 10] normalized continuous and binary controls.
    Target:
      - Continuous next-frame controls [4]: (LX, LY, RX, RY)
      - Discrete next-frame actions [6]: (RT, LT, R3, A, X, Y)
    """
    def __init__(
        self,
        sequences: np.ndarray,
        target_continuous: np.ndarray,
        target_discrete: np.ndarray,
        clip_ids: Optional[List[str]] = None,
        source_games: Optional[List[str]] = None
    ):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.target_continuous = torch.tensor(target_continuous, dtype=torch.float32)
        self.target_discrete = torch.tensor(target_discrete, dtype=torch.float32)
        self.clip_ids = clip_ids or [f"clip_{i}" for i in range(len(sequences))]
        self.source_games = source_games or ["crossfps" for _ in range(len(sequences))]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "sequence": self.sequences[idx],
            "target_continuous": self.target_continuous[idx],
            "target_discrete": self.target_discrete[idx],
            "clip_id": self.clip_ids[idx],
            "source_game": self.source_games[idx]
        }


class CrossFPSTransformerModel(nn.Module):
    """
    Transformer-Based Next-Action Digital Twin for CrossFPS.
    Applies Multi-Head Self-Attention over sequence history T=16,
    with dual output heads:
      1. Continuous joystick trajectory head (Tanh/Linear regression for LX, LY, RX, RY)
      2. Multi-label discrete combat button head (Sigmoid for RT, LT, R3, A, X, Y)
    """
    def __init__(
        self,
        input_dim: int = 10,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 32, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output heads
        self.head_continuous = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Tanh()  # Continuous joysticks bounded in [-1, 1]
        )
        self.head_discrete = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 6)  # 6 discrete binary action logits
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [Batch, T, input_dim]
        b, t, d = x.shape
        h = self.input_proj(x) + self.pos_encoder[:, :t, :]
        encoded = self.transformer(h)
        context = encoded.mean(dim=1) + encoded[:, -1, :]
        pred_continuous = self.head_continuous(context)
        pred_discrete = self.head_discrete(context)
        return pred_continuous, pred_discrete



class CrossFPSLSTMBaseline(nn.Module):
    """
    Recurrent Baseline (2-Layer LSTM) for sequential comparison against Transformer.
    """
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head_continuous = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Tanh()
        )
        self.head_discrete = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        pred_continuous = self.head_continuous(last_hidden)
        pred_discrete = self.head_discrete(last_hidden)
        return pred_continuous, pred_discrete



class CrossFPSTransformerSSL(nn.Module):
    """
    Self-Supervised Transformer for CrossFPS 10-DoF Telemetry.
    Stage 1: Masked controller-feature reconstruction (self-supervised pretraining)
             reconstructing masked frames across the 16-frame telemetry sequence.
    Stage 2: Downstream fine-tuning for continuous movement regression (4-DoF)
             and multi-label discrete combat button classification (6-DoF).
    """
    def __init__(
        self,
        input_dim: int = 10,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        mask_prob: float = 0.20
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.mask_prob = mask_prob
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 32, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # SSL Reconstruction Head (Stage 1)
        self.ssl_reconstruction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, input_dim)
        )

        # Downstream Heads (Stage 2)
        self.head_continuous = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Tanh()
        )
        self.head_discrete = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )

    def forward_ssl_pretrain(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stage 1 forward pass:
        Masks random timesteps in the sequence and trains the model to reconstruct them.
        """
        b, t, d = x.shape
        rand_vals = torch.rand(b, t, device=x.device)
        mask = rand_vals < self.mask_prob
        mask[:, -1] = True  # Always mask the transition frame
        masked_x = x.clone()
        masked_x[mask] = 0.0
        h = self.input_proj(masked_x) + self.pos_encoder[:, :t, :]
        encoded = self.transformer(h)
        reconstructed = self.ssl_reconstruction_head(encoded)
        return reconstructed, mask

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stage 2 forward pass:
        Predicts continuous movement trajectory and discrete button activations.
        """
        b, t, d = x.shape
        h = self.input_proj(x) + self.pos_encoder[:, :t, :]
        encoded = self.transformer(h)
        context = encoded.mean(dim=1) + encoded[:, -1, :]
        pred_continuous = self.head_continuous(context)
        pred_discrete = self.head_discrete(context)
        return pred_continuous, pred_discrete


