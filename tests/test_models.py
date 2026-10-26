"""
Unit tests for all 4 deep learning model architectures.
"""

import os
import pytest
import torch
import torch.nn as nn

from src.models import (
    create_model,
    LSTMNextActionClassifier,
    LSTMTransformerHybridClassifier,
    TransformerNextActionClassifier,
    TransformerSSLNextActionClassifier
)
from src.utils import load_config, PROJECT_CLASSES


@pytest.fixture
def dummy_batch():
    """Provides a synthetic batch of shape (batch_size=4, seq_len=16, feature_dim=23)."""
    torch.manual_seed(42)
    B, T, D = 4, 16, 23
    x = torch.randn(B, T, D)
    y = torch.randint(0, 9, (B,), dtype=torch.long)
    return x, y


def test_model1_lstm_baseline_forward_and_backward(dummy_batch):
    """Tests LSTM baseline forward pass, output shape (B, 9), and backward gradient step."""
    x, y = dummy_batch
    config = load_config()
    model = create_model("lstm_baseline", input_dim=x.shape[2], config=config)

    logits = model(x)
    assert logits.shape == (x.shape[0], 9)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, y)
    assert not torch.isnan(loss)
    loss.backward()

    # Check gradients exist
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_model2_lstm_transformer_hybrid_forward(dummy_batch):
    """Tests LSTM+Transformer hybrid forward pass and output shape (B, 9)."""
    x, y = dummy_batch
    config = load_config()
    model = create_model("lstm_transformer", input_dim=x.shape[2], config=config)

    logits = model(x)
    assert logits.shape == (x.shape[0], 9)


def test_model3_transformer_primary_forward(dummy_batch):
    """Tests primary Transformer encoder forward pass and output shape (B, 9)."""
    x, y = dummy_batch
    config = load_config()
    model = create_model("transformer", input_dim=x.shape[2], config=config)

    logits = model(x)
    assert logits.shape == (x.shape[0], 9)


def test_model4_transformer_ssl_two_stages(dummy_batch):
    """
    Tests Model 4:
    Stage 1: SSL pretraining forward pass (reconstruction and masking).
    Stage 2: Downstream next-action classification forward pass.
    """
    x, y = dummy_batch
    config = load_config()
    model = create_model("transformer_ssl", input_dim=x.shape[2], config=config)

    # Stage 1: SSL reconstruction
    reconstructed, mask = model.forward_ssl_pretrain(x)
    assert reconstructed.shape == x.shape
    assert mask.shape == (x.shape[0], x.shape[1])
    assert mask.sum() > 0

    # Stage 2: Supervised classification
    logits = model(x)
    assert logits.shape == (x.shape[0], 9)


def test_checkpoint_saving_and_loading(tmp_path, dummy_batch):
    """Tests saving model checkpoint and loading it back."""
    x, y = dummy_batch
    config = load_config()
    model = create_model("transformer", input_dim=x.shape[2], config=config)

    ckpt_path = str(tmp_path / "test_model.pt")
    checkpoint = {
        "model_name": "transformer",
        "model_state_dict": model.state_dict(),
        "input_dim": x.shape[2],
        "config": config,
        "classes": PROJECT_CLASSES
    }
    torch.save(checkpoint, ckpt_path)

    # Reload
    loaded_ckpt = torch.load(ckpt_path, weights_only=False)
    loaded_model = create_model("transformer", input_dim=loaded_ckpt["input_dim"], config=loaded_ckpt["config"])
    loaded_model.load_state_dict(loaded_ckpt["model_state_dict"])
    loaded_model.eval()

    model.eval()
    with torch.no_grad():
        out_orig = model(x)
        out_loaded = loaded_model(x)
    assert torch.allclose(out_orig, out_loaded, atol=1e-5)
