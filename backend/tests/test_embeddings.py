"""
Unit and Integration Tests for Player Behaviour Embeddings & Gameplay Pattern Explorer.
Verifies real Transformer representation extraction, PCA 2D dimensionality reduction,
telemetry alignment, metadata integrity, and FastAPI endpoints.
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.embedding_service import (
    compute_gameplay_embeddings,
    analyze_sequence_telemetry
)
from backend.app.services.crossfps_predictor import get_test_splits

client = TestClient(app)


def test_analyze_sequence_telemetry_factual():
    """Verify factual calculation of physical stick means and trigger rates."""
    # Synthetic controlled sequence [16 frames, 10 channels]
    seq = np.zeros((16, 10), dtype=np.float32)
    # Forward sprint
    seq[:, 1] = 0.8
    # Fire trigger active for first 8 frames
    seq[:8, 4] = 1.0

    stats = analyze_sequence_telemetry(seq)
    assert stats["mean_move_ly"] == 0.8
    assert stats["fire_rate"] == 0.5
    assert "Forward" in stats["movement_label"]
    assert stats["dominant_action"] == "Fire (RT)"


def test_compute_gameplay_embeddings_transformer():
    """Verify 64-D embedding extraction and PCA 2D transformation from real Transformer checkpoint."""
    res = compute_gameplay_embeddings(
        model_name="transformer",
        sample_size=100,
        reduction_method="pca",
        force_refresh=True
    )

    assert res["model_id"] == "transformer"
    assert res["embedding_dim"] == 64
    assert res["reduction_method"] == "PCA (2 Components)"
    assert len(res["explained_variance_ratio"]) == 2
    assert res["total_variance_explained_pct"] > 30.0  # Real PCA explains significant variance
    assert len(res["points"]) == 100
    assert "scientific_disclosure" in res

    # Verify first point metadata alignment
    pt = res["points"][0]
    assert pt["id"].startswith("SEQ-CFPS-")
    assert isinstance(pt["sequence_index"], int)
    assert len(pt["clip_id"]) > 5
    assert pt["game"] in ["halo_infinite", "call_of_duty__warzone", "call_of_duty__modern_warfare_iii", "call_of_duty"]
    assert isinstance(pt["x"], float)
    assert isinstance(pt["y"], float)
    assert "controller_stats" in pt


def test_compute_gameplay_embeddings_ssl():
    """Verify embedding extraction from the fine-tuned Self-Supervised Transformer."""
    res = compute_gameplay_embeddings(
        model_name="transformer_ssl",
        sample_size=50,
        reduction_method="pca",
        force_refresh=True
    )

    assert res["model_id"] == "transformer_ssl"
    assert res["embedding_dim"] == 64
    assert len(res["points"]) == 50
    assert res["total_sequences_available"] == 10859


def test_embedding_points_correspond_to_real_splits():
    """Verify that returned points map directly to genuine CrossFPS test split data."""
    splits = get_test_splits()
    X_test = splits["X_test"]
    clip_ids = splits["clip_ids_test"]

    res = compute_gameplay_embeddings(model_name="transformer", sample_size=50)
    for pt in res["points"]:
        idx = pt["sequence_index"]
        assert 0 <= idx < len(X_test)
        # Verify genuine clip ID matches
        assert pt["clip_id"] == str(clip_ids[idx])


def test_api_get_embeddings_endpoint():
    """Verify FastAPI GET /api/embeddings route."""
    response = client.get("/api/embeddings?model_name=transformer&sample_size=80")
    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == "transformer"
    assert len(data["points"]) == 80
    assert "legend" in data
    assert "games" in data["legend"]
    assert "actions" in data["legend"]


def test_api_get_sequence_detail():
    """Verify GET /api/embeddings/sequence/{sequence_index} endpoint."""
    response = client.get("/api/embeddings/sequence/0")
    assert response.status_code == 200
    detail = response.json()
    assert detail["sequence_index"] == 0
    assert "clip_id" in detail
    assert "frames" in detail
    assert len(detail["frames"]) == 16
    assert "target_next_action" in detail


def test_api_get_sequence_detail_out_of_range():
    """Verify 404 error on out-of-range sequence index."""
    response = client.get("/api/embeddings/sequence/999999")
    assert response.status_code == 404
