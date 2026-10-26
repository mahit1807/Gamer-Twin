"""
Player Behaviour Signature API routes for GAMER-TWIN.
Computes descriptive behavioral statistics from actual recorded player telemetry:
- Action frequencies & empirical distributions
- Movement vs weapon/combat event ratio
- Action transition matrix (empirical 1st-order Markov transitions)
- Common observed action sequences (2-grams and 3-grams)
- Temporal action distribution shifts over time windows
"""

import os
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query
import pandas as pd
import numpy as np

from src.utils import PROJECT_CLASSES
from backend.app.schemas.schemas import PlayerSignatureResponse, ActionTransition

router = APIRouter()

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent

COMBAT_ACTIONS = {"Fire", "Reload", "Switch Weapon", "Scope", "Grenade"}
MOVEMENT_ACTIONS = {"Move", "Jump", "Crouch"}


def get_telemetry_df():
    raw_dir = WORKSPACE_ROOT / "data" / "raw"
    sample_dir = WORKSPACE_ROOT / "data" / "sample_data"

    if raw_dir.exists() and any(raw_dir.glob("*.csv")):
        csv_file = next(raw_dir.glob("*tick*.csv"), None) or next(raw_dir.glob("*.csv"), None)
        is_synthetic = False
    elif sample_dir.exists() and any(sample_dir.glob("*.csv")):
        csv_file = next(sample_dir.glob("*tick*.csv"), None) or next(sample_dir.glob("*.csv"), None)
        is_synthetic = True
    else:
        raise FileNotFoundError("No telemetry data found for calculating player signature.")

    df = pd.read_csv(csv_file)
    return df, is_synthetic


@router.get("", response_model=PlayerSignatureResponse)
def get_player_behaviour_signature(player_id: Optional[str] = Query(None)):
    try:
        df, is_synthetic = get_telemetry_df()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if player_id and "player_id" in df.columns:
        player_df = df[df["player_id"].astype(str) == str(player_id)]
        if player_df.empty:
            player_df = df
    else:
        player_df = df

    total_records = len(player_df)
    if total_records < 10:
        raise HTTPException(status_code=400, detail="Insufficient data to compute player signature.")

    action_col = "action" if "action" in player_df.columns else "event"
    actions = player_df[action_col].dropna().astype(str).tolist()

    # 1. Observed action frequencies and proportions
    action_counts = Counter(actions)
    total_actions = max(1, len(actions))
    action_proportions = {
        cls: round(action_counts.get(cls, 0) / total_actions, 4)
        for cls in PROJECT_CLASSES
    }
    action_frequencies = {
        cls: int(action_counts.get(cls, 0))
        for cls in PROJECT_CLASSES
    }

    # 2. Movement vs Combat ratio
    combat_count = sum(action_counts.get(a, 0) for a in COMBAT_ACTIONS)
    movement_count = sum(action_counts.get(a, 0) for a in MOVEMENT_ACTIONS)
    other_count = max(0, total_actions - combat_count - movement_count)

    movement_vs_combat = {
        "movement_percentage": round((movement_count / total_actions) * 100, 2),
        "combat_percentage": round((combat_count / total_actions) * 100, 2),
        "utility_percentage": round((other_count / total_actions) * 100, 2)
    }

    # 3. Action transition frequencies (first-order empirical transitions)
    transitions = defaultdict(int)
    from_counts = defaultdict(int)
    for i in range(len(actions) - 1):
        a1, a2 = actions[i], actions[i + 1]
        transitions[(a1, a2)] += 1
        from_counts[a1] += 1

    top_transitions_list = []
    # Sort by count descending
    sorted_trans = sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:15]
    for (a1, a2), cnt in sorted_trans:
        prob = cnt / max(1, from_counts[a1])
        top_transitions_list.append(
            ActionTransition(
                from_action=a1,
                to_action=a2,
                probability=round(float(prob), 4),
                count=int(cnt)
            )
        )

    # 4. Common observed action sequences (2-grams and 3-grams)
    bigrams = Counter([f"{actions[i]} -> {actions[i+1]}" for i in range(len(actions) - 1)])
    trigrams = Counter([f"{actions[i]} -> {actions[i+1]} -> {actions[i+2]}" for i in range(len(actions) - 2)])

    common_ngrams = []
    for bg, c in bigrams.most_common(5):
        common_ngrams.append({
            "ngram_type": "2-gram",
            "sequence": bg,
            "count": c,
            "frequency_pct": round((c / max(1, len(actions) - 1)) * 100, 2)
        })
    for tg, c in trigrams.most_common(5):
        common_ngrams.append({
            "ngram_type": "3-gram",
            "sequence": tg,
            "count": c,
            "frequency_pct": round((c / max(1, len(actions) - 2)) * 100, 2)
        })

    # 5. Temporal shifts across replay windows (split actions into 4 chronological quarters)
    quarter_size = max(1, len(actions) // 4)
    window_entropy = []
    for q in range(4):
        window_slice = actions[q * quarter_size:(q + 1) * quarter_size]
        if not window_slice:
            continue
        q_counter = Counter(window_slice)
        q_total = len(window_slice)
        # Shannon entropy of action diversity
        entropy = -sum((cnt / q_total) * np.log2(cnt / q_total) for cnt in q_counter.values())
        window_entropy.append({
            "window_quarter": f"Quarter {q+1}",
            "sample_count": q_total,
            "entropy": round(float(entropy), 3),
            "dominant_action": q_counter.most_common(1)[0][0] if q_counter else "None",
            "combat_intensity": round((sum(q_counter.get(a, 0) for a in COMBAT_ACTIONS) / q_total) * 100, 1)
        })

    resolved_player = str(player_df["player_id"].iloc[0]) if "player_id" in player_df.columns else "Player_01"

    return PlayerSignatureResponse(
        player_id=resolved_player,
        data_source="SYNTHETIC_TEST_FIXTURE (Demo Fixture)" if is_synthetic else "REAL_EGOCS_400K",
        is_synthetic=is_synthetic,
        total_observed_events=total_actions,
        action_frequencies=action_frequencies,
        action_proportions=action_proportions,
        movement_vs_combat_ratio=movement_vs_combat,
        top_transitions=top_transitions_list,
        common_ngrams=common_ngrams,
        window_entropy=window_entropy
    )
