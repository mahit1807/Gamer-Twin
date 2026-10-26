"""
Deep Learning Model Architectures for GAMER-TWIN.
Includes:
1. Model 1: LSTM Baseline
2. Model 2: LSTM + Transformer Hybrid
3. Model 3: Transformer Encoder (Primary Model)
4. Model 4: Transformer + Self-Supervised Learning (SSL Pretrained Backbone)
"""

import math
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import setup_logger, PROJECT_CLASSES

logger = setup_logger("Models")


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence tokens."""
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================================
# Model 1: LSTM Baseline
# ============================================================================
class LSTMNextActionClassifier(nn.Module):
    """
    Model 1: Baseline Recurrent Neural Network for sequential gameplay modeling.
    Takes sequence (B, T, D_in), passes through multi-layer LSTM, and classifies
    the final hidden state into the 9 project target classes.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 9,
        dropout: float = 0.1
    ):
        super().__init__()
        self.model_type = "LSTM_Baseline"
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, seq_len, input_dim)
        Returns:
            logits: Tensor of shape (B, num_classes)
        """
        B, T, D = x.shape
        h_proj = F.relu(self.input_proj(x))
        lstm_out, (hn, _) = self.lstm(h_proj)
        # Use final time-step representation
        last_step = lstm_out[:, -1, :]
        last_step = self.dropout(last_step)
        logits = self.classifier(last_step)
        return logits


# ============================================================================
# Model 2: LSTM + Transformer Hybrid
# ============================================================================
class LSTMTransformerHybridClassifier(nn.Module):
    """
    Model 2: Hybrid combining recurrent inductive bias with self-attention.
    Processes the raw telemetry with an initial LSTM recurrent layer to capture
    local frame-to-frame temporal momentum, then refines the sequence representations
    with a multi-head Transformer encoder to model long-range tactical context.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        lstm_layers: int = 1,
        transformer_layers: int = 1,
        num_heads: int = 4,
        dim_feedforward: int = 128,
        num_classes: int = 9,
        dropout: float = 0.1
    ):
        super().__init__()
        self.model_type = "LSTM_Transformer_Hybrid"
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, seq_len, input_dim)
        Returns:
            logits: Tensor of shape (B, num_classes)
        """
        h_proj = F.relu(self.input_proj(x))
        lstm_out, _ = self.lstm(h_proj)
        # Pass recurrent sequence through Transformer self-attention
        pe_out = self.pos_encoder(lstm_out)
        trans_out = self.transformer_encoder(pe_out)
        # Sequence-level representation via mean pooling across sequence length
        pooled = torch.mean(trans_out, dim=1)
        logits = self.classifier(self.dropout(pooled))
        return logits


# ============================================================================
# Model 3: Transformer Encoder (Primary Model)
# ============================================================================
class TransformerNextActionClassifier(nn.Module):
    """
    Model 3: Primary pure-Transformer architecture for GAMER-TWIN.
    Maps input telemetry features into embedding space, injects positional encodings,
    processes with stacked Transformer Encoder layers, and predicts next action probabilities.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 128,
        num_classes: int = 9,
        dropout: float = 0.1
    ):
        super().__init__()
        self.model_type = "Transformer_Encoder"
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Input feature projection to d_model
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification prediction head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns sequence feature representations of shape (B, seq_len, hidden_dim)."""
        emb = self.embedding(x)
        pe = self.pos_encoder(emb)
        out = self.transformer(pe)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, seq_len, input_dim)
        Returns:
            logits: Tensor of shape (B, num_classes)
        """
        features = self.extract_features(x)
        # Temporal pooling: combine last-token focus with average sequence context
        avg_pool = torch.mean(features, dim=1)
        last_token = features[:, -1, :]
        combined = 0.5 * (avg_pool + last_token)
        logits = self.classifier(combined)
        return logits


# ============================================================================
# Model 4: Transformer + Self-Supervised Learning (SSL Pretraining)
# ============================================================================
class TransformerSSLNextActionClassifier(nn.Module):
    """
    Model 4: Transformer with explicit two-stage Self-Supervised Learning.
    Stage 1 (Self-Supervised Pretraining): Learns game dynamics by reconstructing
    masked telemetry feature frames across the past sequence (Masked Autoencoding).
    Stage 2 (Supervised Fine-Tuning): The learned backbone is fine-tuned to classify
    the observed next player action.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 128,
        num_classes: int = 9,
        dropout: float = 0.1,
        mask_prob: float = 0.20
    ):
        super().__init__()
        self.model_type = "Transformer_SSL"
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.mask_prob = mask_prob

        # Shared Transformer Backbone
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Stage 1: Self-Supervised Reconstruction Head
        self.ssl_reconstruction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # Stage 2: Supervised Next-Action Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward_ssl_pretrain(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stage 1 forward pass:
        Masks random timesteps in the sequence and trains the model to reconstruct them.
        
        Returns:
            reconstructed: Shape (B, seq_len, input_dim)
            mask: Bool tensor of shape (B, seq_len) where True indicates masked frames
        """
        B, T, D = x.shape
        # Create random mask across sequence length
        rand_vals = torch.rand(B, T, device=x.device)
        mask = rand_vals < self.mask_prob
        # Ensure at least one token is masked per sequence
        mask[:, -1] = True

        # Zero-out masked timesteps in the input
        masked_x = x.clone()
        masked_x[mask] = 0.0

        emb = self.embedding(masked_x)
        pe = self.pos_encoder(emb)
        encoded = self.transformer(pe)
        reconstructed = self.ssl_reconstruction_head(encoded)

        return reconstructed, mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Stage 2 forward pass (Next Action Classification):
        Takes the unmasked gameplay sequence and predicts the next action logits.
        """
        emb = self.embedding(x)
        pe = self.pos_encoder(emb)
        encoded = self.transformer(pe)

        avg_pool = torch.mean(encoded, dim=1)
        last_token = encoded[:, -1, :]
        combined = 0.5 * (avg_pool + last_token)
        logits = self.classifier(combined)
        return logits


def create_model(model_name: str, input_dim: int, config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to instantiate any of the 4 supported model variants.
    
    Supported names:
    - 'lstm_baseline'
    - 'lstm_transformer'
    - 'transformer'
    - 'transformer_ssl'
    """
    models_cfg = config.get("models", {})
    common = models_cfg.get("common", {})
    hidden_dim = common.get("hidden_dim", 64)
    dropout = common.get("dropout", 0.1)
    num_classes = common.get("num_classes", 9)

    model_key = model_name.lower().replace("-", "_").replace(" ", "_")

    if model_key in ("lstm", "lstm_baseline"):
        m_cfg = models_cfg.get("lstm_baseline", {})
        return LSTMNextActionClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=m_cfg.get("num_layers", 2),
            num_classes=num_classes,
            dropout=dropout
        )

    elif model_key in ("lstm_transformer", "lstm_transformer_hybrid", "hybrid"):
        m_cfg = models_cfg.get("lstm_transformer", {})
        return LSTMTransformerHybridClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            lstm_layers=m_cfg.get("lstm_layers", 1),
            transformer_layers=m_cfg.get("transformer_layers", 1),
            num_heads=m_cfg.get("num_heads", 4),
            dim_feedforward=m_cfg.get("dim_feedforward", 128),
            num_classes=num_classes,
            dropout=dropout
        )

    elif model_key in ("transformer", "transformer_only", "primary"):
        m_cfg = models_cfg.get("transformer", {})
        return TransformerNextActionClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=m_cfg.get("num_layers", 2),
            num_heads=m_cfg.get("num_heads", 4),
            dim_feedforward=m_cfg.get("dim_feedforward", 128),
            num_classes=num_classes,
            dropout=dropout
        )

    elif model_key in ("transformer_ssl", "ssl"):
        m_cfg = models_cfg.get("transformer_ssl", {})
        return TransformerSSLNextActionClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=m_cfg.get("num_layers", 2),
            num_heads=m_cfg.get("num_heads", 4),
            dim_feedforward=m_cfg.get("dim_feedforward", 128),
            num_classes=num_classes,
            dropout=dropout,
            mask_prob=m_cfg.get("ssl_mask_prob", 0.20)
        )

    else:
        raise ValueError(
            f"Unknown model variant '{model_name}'. "
            f"Supported options: ['lstm_baseline', 'lstm_transformer', 'transformer', 'transformer_ssl']"
        )
