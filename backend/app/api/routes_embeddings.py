"""
Player Behaviour Embeddings & Gameplay Pattern Explorer API Routes.
Exposes real 64-D latent embeddings extracted from trained Transformer encoders
over CrossFPS test sequences, projected to 2D via PCA.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from pathlib import Path
import numpy as np

from backend.app.services.embedding_service import compute_gameplay_embeddings
from backend.app.services.crossfps_predictor import get_test_splits, ACTION_NAMES, DISCRETE_CHANNELS

router = APIRouter()

@router.get("")
@router.get("/")
def get_embeddings(
    model_name: str = Query("transformer", description="Transformer model variant: 'transformer' or 'transformer_ssl'"),
    sample_size: int = Query(600, ge=50, le=2000, description="Number of genuine CrossFPS test sequences to embed"),
    refresh: bool = Query(False, description="Force recomputation bypassing cache")
):
    """
    Extracts fixed-length 64-D sequence embeddings from the selected Transformer encoder,
    fits PCA 2D projection, and returns coordinates aligned with genuine CrossFPS telemetry metadata.
    """
    try:
        data = compute_gameplay_embeddings(
            model_name=model_name,
            sample_size=sample_size,
            reduction_method="pca",
            force_refresh=refresh
        )
        return data
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AttributeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract embeddings: {str(e)}")


@router.get("/sequence/{sequence_index}")
def get_sequence_detail(sequence_index: int):
    """
    Returns full 16-frame 10-DoF telemetry for a specific sequence from the CrossFPS test split.
    """
    try:
        splits = get_test_splits()
        X_test = splits["X_test"]
        clip_ids = splits["clip_ids_test"]
        games = splits["games_test"]
        yd_test = splits["y_disc_test"]

        if sequence_index < 0 or sequence_index >= len(X_test):
            raise HTTPException(
                status_code=404,
                detail=f"Sequence index {sequence_index} out of range [0, {len(X_test) - 1}]."
            )

        seq = X_test[sequence_index]  # [16, 10]
        clip_id = str(clip_ids[sequence_index])
        game = str(games[sequence_index])
        disc_vec = yd_test[sequence_index]
        active_acts = [DISCRETE_CHANNELS[k] for k in range(min(len(DISCRETE_CHANNELS), len(disc_vec))) if disc_vec[k] > 0.5]
        target_action = ", ".join(active_acts) if active_acts else "Neutral Steering / Movement"

        frames = []
        for t in range(len(seq)):
            frames.append({
                "frame": t,
                "lx": round(float(seq[t, 0]), 3),
                "ly": round(float(seq[t, 1]), 3),
                "rx": round(float(seq[t, 2]), 3),
                "ry": round(float(seq[t, 3]), 3),
                "rt_fire": round(float(seq[t, 4]), 2),
                "lt_ads": round(float(seq[t, 5]), 2),
                "r3_melee": round(float(seq[t, 6]), 2),
                "a_jump": round(float(seq[t, 7]), 2),
                "x_reload": round(float(seq[t, 8]), 2),
                "y_switch": round(float(seq[t, 9]), 2),
            })

        return {
            "sequence_index": sequence_index,
            "clip_id": clip_id,
            "game": game,
            "target_next_action": target_action,
            "target_discrete_vector": [float(v) for v in disc_vec],
            "frames": frames
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving sequence detail: {str(e)}")
