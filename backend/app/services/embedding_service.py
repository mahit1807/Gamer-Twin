"""
Player Behaviour Embeddings & Gameplay Pattern Explorer Service.
Extracts 64-dimensional latent representations from genuine trained Transformer checkpoints
over verified CrossFPS controller telemetry sequences (data/CrossFPS-val/).
Applies PCA 2D projection with factual, verifiable metadata and strict results integrity.
No synthetic data, random vectors, or speculative personality profiling.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch
from sklearn.decomposition import PCA

from backend.app.services.crossfps_predictor import (
    load_crossfps_model,
    get_test_splits,
    ACTION_NAMES,
    DISCRETE_CHANNELS,
    get_device
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PROCESSED_DIR = WORKSPACE_ROOT / "outputs" / "processed"
CHECKPOINTS_DIR = WORKSPACE_ROOT / "outputs" / "checkpoints"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# In-memory cache
_EMBEDDINGS_CACHE: Dict[str, Dict[str, Any]] = {}

GAME_DISPLAY_NAMES = {
    "halo_infinite": "Halo Infinite",
    "call_of_duty__warzone": "Call of Duty: Warzone",
    "call_of_duty__modern_warfare_iii": "COD: Modern Warfare III",
    "call_of_duty": "Call of Duty (Classic)"
}

GAME_GENRES = {
    "halo_infinite": "Sci-Fi Arena FPS",
    "call_of_duty__warzone": "Battle Royale",
    "call_of_duty__modern_warfare_iii": "Fast Tactical FPS",
    "call_of_duty": "Classic Military Shooter"
}

GAME_COLORS = {
    "halo_infinite": "#C66A4E",                  # Terracotta primary
    "call_of_duty__warzone": "#B95D3B",          # Burnt orange
    "call_of_duty__modern_warfare_iii": "#82917A",# Muted sage
    "call_of_duty": "#302721"                    # Deep espresso
}

ACTION_COLORS = {
    "Scope / Aim (LT)": "#4A6FA5",
    "Fire (RT)": "#C66A4E",
    "Jump (A)": "#82917A",
    "Reload (X)": "#B95D3B",
    "Melee (R3)": "#8E7DBE",
    "Switch Weapon (Y)": "#E5A93C",
    "Neutral Steering / Movement": "#81786E"
}


def analyze_sequence_telemetry(seq: np.ndarray) -> Dict[str, Any]:
    """
    Factual extraction of physical controller inputs over 16 frames.
    Channels: 0:LX, 1:LY, 2:RX, 3:RY, 4:RT, 5:LT, 6:R3, 7:A, 8:X, 9:Y
    """
    lx_mean = float(np.mean(seq[:, 0]))
    ly_mean = float(np.mean(seq[:, 1]))
    rx_mean = float(np.mean(seq[:, 2]))
    ry_mean = float(np.mean(seq[:, 3]))

    # Channel occurrence rates
    rt_rate = float(np.mean(seq[:, 4] > 0.5))
    lt_rate = float(np.mean(seq[:, 5] > 0.5))
    r3_rate = float(np.mean(seq[:, 6] > 0.5))
    a_rate  = float(np.mean(seq[:, 7] > 0.5))
    x_rate  = float(np.mean(seq[:, 8] > 0.5))
    y_rate  = float(np.mean(seq[:, 9] > 0.5))

    # Dominant physical action
    action_rates = [
        ("Fire (RT)", rt_rate),
        ("Scope / Aim (LT)", lt_rate),
        ("Melee (R3)", r3_rate),
        ("Jump (A)", a_rate),
        ("Reload (X)", x_rate),
        ("Switch Weapon (Y)", y_rate)
    ]
    action_rates.sort(key=lambda item: item[1], reverse=True)
    top_act, top_val = action_rates[0]
    dominant_action = top_act if top_val > 0.2 else "Neutral Steering / Movement"

    # Factual movement description
    move_desc = []
    if ly_mean > 0.3: move_desc.append("Forward Sprint")
    elif ly_mean < -0.3: move_desc.append("Backpedal")
    if lx_mean > 0.3: move_desc.append("Right Strafe")
    elif lx_mean < -0.3: move_desc.append("Left Strafe")
    movement_label = " + ".join(move_desc) if move_desc else "Stationary / Micro-adjustment"

    # Factual camera description
    cam_desc = []
    if abs(rx_mean) > 0.2: cam_desc.append(f"Yaw Turn ({rx_mean:+.2f})")
    if abs(ry_mean) > 0.2: cam_desc.append(f"Pitch Adjust ({ry_mean:+.2f})")
    camera_label = ", ".join(cam_desc) if cam_desc else "Centering / Steady Aim"

    return {
        "dominant_action": dominant_action,
        "mean_strafe_lx": round(lx_mean, 3),
        "mean_move_ly": round(ly_mean, 3),
        "mean_turn_rx": round(rx_mean, 3),
        "mean_pitch_ry": round(ry_mean, 3),
        "fire_rate": round(rt_rate, 2),
        "ads_rate": round(lt_rate, 2),
        "jump_rate": round(a_rate, 2),
        "movement_label": movement_label,
        "camera_label": camera_label
    }


def compute_gameplay_embeddings(
    model_name: str = "transformer",
    sample_size: int = 600,
    reduction_method: str = "pca",
    force_refresh: bool = False
) -> Dict[str, Any]:
    """
    Extracts fixed-length 64-D sequence embeddings from the trained Transformer encoder,
    applies PCA 2D projection, and formats sequence points with factual telemetry statistics.
    """
    clean_model = "transformer_ssl" if "ssl" in model_name.lower() else "transformer"
    cache_key = f"{clean_model}_{sample_size}_{reduction_method}"

    # Check memory cache
    if not force_refresh and cache_key in _EMBEDDINGS_CACHE:
        return _EMBEDDINGS_CACHE[cache_key]

    # Check file cache
    cache_file = PROCESSED_DIR / f"embeddings_cache_{cache_key}.json"
    if not force_refresh and cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
                _EMBEDDINGS_CACHE[cache_key] = data
                return data
        except Exception:
            pass

    # 1. Load model and verify compatibility
    ckpt_file = f"{clean_model}_crossfps_best.pt"
    ckpt_path = CHECKPOINTS_DIR / ckpt_file
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint '{ckpt_file}' not found on disk. "
            f"Please ensure model has been trained before generating embeddings."
        )

    model = load_crossfps_model(clean_model)
    if not hasattr(model, "extract_embeddings"):
        raise AttributeError(f"Model {type(model).__name__} does not implement extract_embeddings.")

    device = get_device()
    model.to(device)
    model.eval()

    # 2. Load held-out test splits (10,859 genuine CrossFPS sequences)
    splits = get_test_splits()
    X_test = splits["X_test"]
    clip_ids = splits["clip_ids_test"]
    games = splits["games_test"]
    total_available = len(X_test)

    # 3. Deterministic, stratified sampling across real games
    sample_size = min(sample_size, total_available)
    unique_games = np.unique(games)
    sampled_indices = []

    # Sample proportionally from each game
    per_game_target = sample_size // len(unique_games)
    rng = np.random.RandomState(42)

    for g in unique_games:
        g_indices = np.where(games == g)[0]
        if len(g_indices) > 0:
            chosen = rng.choice(g_indices, size=min(per_game_target, len(g_indices)), replace=False)
            sampled_indices.extend(chosen.tolist())

    # Fill remainder if any
    if len(sampled_indices) < sample_size:
        remaining = list(set(range(total_available)) - set(sampled_indices))
        extra = rng.choice(remaining, size=sample_size - len(sampled_indices), replace=False)
        sampled_indices.extend(extra.tolist())

    sampled_indices = sorted(sampled_indices)
    sampled_X = X_test[sampled_indices]  # [N, 16, 10]

    # 4. Extract 64-D Latent Sequence Embeddings
    batch_size = 64
    all_embeddings = []

    with torch.no_grad():
        for start_idx in range(0, len(sampled_X), batch_size):
            end_idx = min(start_idx + batch_size, len(sampled_X))
            batch_tensor = torch.tensor(sampled_X[start_idx:end_idx], dtype=torch.float32).to(device)
            emb = model.extract_embeddings(batch_tensor)  # [B, 64]
            all_embeddings.append(emb.cpu().numpy())

    embeddings_array = np.concatenate(all_embeddings, axis=0)  # [N, 64]
    embedding_dim = int(embeddings_array.shape[1])

    # 5. Dimensionality Reduction (PCA to 2D)
    pca = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(embeddings_array)
    explained_var = [round(float(v), 4) for v in pca.explained_variance_ratio_]

    # 6. Build point metadata
    points = []
    for i, seq_idx in enumerate(sampled_indices):
        seq = sampled_X[i]
        c_id = str(clip_ids[seq_idx])
        g_id = str(games[seq_idx])
        g_name = GAME_DISPLAY_NAMES.get(g_id, g_id.replace("_", " ").title())
        genre = GAME_GENRES.get(g_id, "FPS Telemetry")

        stats = analyze_sequence_telemetry(seq)
        summary = (
            f"16-frame sequence from {g_name} with {stats['movement_label'].lower()} "
            f"and {stats['camera_label'].lower()}."
        )

        points.append({
            "id": f"SEQ-CFPS-{seq_idx+1:05d}",
            "sequence_index": int(seq_idx),
            "clip_id": c_id,
            "game": g_id,
            "game_name": g_name,
            "genre": genre,
            "x": round(float(coords_2d[i, 0]), 3),
            "y": round(float(coords_2d[i, 1]), 3),
            "dominant_action": stats["dominant_action"],
            "controller_stats": stats,
            "summary": summary
        })

    model_display = "Self-Supervised Transformer" if clean_model == "transformer_ssl" else "Transformer Encoder"
    result = {
        "model_id": clean_model,
        "model_name": model_display,
        "checkpoint": ckpt_file,
        "embedding_dim": embedding_dim,
        "reduction_method": "PCA (2 Components)",
        "explained_variance_ratio": explained_var,
        "total_variance_explained_pct": round(sum(explained_var) * 100, 1),
        "total_sequences_embedded": len(points),
        "total_sequences_available": total_available,
        "data_provenance": "CrossFPS Verified 10-DoF Telemetry (outputs/processed/crossfps_splits.npz)",
        "scientific_disclosure": (
            "Points close together share similar pooled hidden-state activations in the Transformer encoder. "
            "Proximity reflects correlated temporal control inputs (analog stick velocities and button timing), "
            "but does not prove identical player intent, skill, or fixed personality traits."
        ),
        "legend": {
            "games": GAME_COLORS,
            "actions": ACTION_COLORS
        },
        "points": points
    }

    # Save cache
    try:
        with open(cache_file, "w") as f:
            json.dump(result, f, indent=2)
    except Exception:
        pass

    _EMBEDDINGS_CACHE[cache_key] = result
    return result
