"""
User Gameplay Upload & Real Dataset Clip Exploration API routes for GAMER-TWIN.
Supports exploring genuine CrossFPS validation clips (10-DoF actions_raw.parquet)
and importing user-recorded gameplay telemetry CSV/JSON exports.
No synthetic profiles or fabricated gameplay sessions are permitted.
"""

import os
import io
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Query
import pandas as pd

from backend.app.services.crossfps_service import (
    list_clips,
    parse_clip_parquet,
    get_dataset_stats,
    CONTINUOUS_CHANNELS,
    DISCRETE_CHANNELS,
    DISCRETE_NAMES
)

router = APIRouter()

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
USER_UPLOAD_DIR = WORKSPACE_ROOT / "outputs" / "user_gameplay"
os.makedirs(USER_UPLOAD_DIR, exist_ok=True)

# In-memory store for active player or dataset clip profile
_CURRENT_PROFILE: Dict[str, Any] = {
    "has_profile": False,
    "is_dataset_clip": False,
    "filename": None,
    "clip_id": None,
    "total_records": 0,
    "player_id": None,
    "source_game": None,
    "action_counts": {},
    "movement_vs_combat": {"movement_pct": 0.0, "combat_pct": 0.0, "utility_pct": 0.0},
    "timeline": [],
    "schema_compatible": False,
    "missing_fields": []
}


@router.get("/profile")
def get_user_profile():
    return _CURRENT_PROFILE


@router.get("/clips")
def get_available_dataset_clips(
    offset: int = Query(0, ge=0),
    limit: int = Query(30, ge=1, le=100),
    game: Optional[str] = Query(None)
):
    """
    Returns verified CrossFPS validation clips for exploration on My Gameplay.
    """
    return list_clips(offset=offset, limit=limit, game_filter=game)


@router.get("/clip/{clip_id}")
def load_dataset_clip_profile(clip_id: str):
    """
    Loads genuine 100-frame controller telemetry for a specific CrossFPS clip,
    derives its empirical action distributions and timeline, and sets it as the active profile.
    """
    global _CURRENT_PROFILE

    try:
        parsed = parse_clip_parquet(clip_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Convert timeline to format expected by UI
    ui_timeline = []
    for item in parsed["timeline"]:
        ui_timeline.append({
            "step": item["frame"],
            "tick": item["frame"],
            "seconds": item["seconds"],
            "action": item["primary_action"],
            "weapon": parsed["source_game"].replace("_", " ").title(),
            "pos": [item["lx"], item["ly"]]
        })

    _CURRENT_PROFILE = {
        "has_profile": True,
        "is_dataset_clip": True,
        "filename": f"{clip_id}/actions_raw.parquet",
        "clip_id": clip_id,
        "total_records": parsed["total_frames"],
        "player_id": f"Dataset Clip [{clip_id[:16]}...]",
        "source_game": parsed["source_game"],
        "map": parsed["source_game"],  # Backwards compatibility for existing UI components
        "action_counts": parsed["action_counts"],
        "movement_vs_combat": parsed["movement_vs_combat"],
        "timeline": ui_timeline,
        "schema_compatible": True,
        "missing_fields": []
    }

    return _CURRENT_PROFILE


@router.post("/upload")
async def upload_gameplay_file(file: UploadFile = File(...)):
    global _CURRENT_PROFILE

    filename = file.filename or "uploaded_telemetry.csv"
    lower_name = filename.lower()

    # Reject video files with explicit error explaining why raw video cannot be used directly
    if lower_name.endswith(".mp4") or lower_name.endswith(".mkv") or lower_name.endswith(".avi"):
        raise HTTPException(
            status_code=400,
            detail=(
                "Uploaded file is a raw video. Video files do not contain structured controller telemetry. "
                "The CrossFPS framework requires aligned 10-DoF gamepad telemetry (LX, LY, RX, RY, RT, LT, R3, A, X, Y). "
                "Because video-to-telemetry extraction is not implemented in this workspace, video cannot yet be analysed directly."
            )
        )

    if lower_name.endswith(".dem"):
        raise HTTPException(
            status_code=400,
            detail="Raw CS2 demo files (.dem) are not supported. CrossFPS uses frame-aligned 10-DoF controller telemetry CSV/JSON exports."
        )

    if not (lower_name.endswith(".csv") or lower_name.endswith(".json")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload a structured telemetry CSV or JSON export."
        )

    content = await file.read()
    try:
        if lower_name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_json(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse upload as structured telemetry: {str(e)}")

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file contains 0 records.")

    # Validate against CrossFPS 10-DoF or standard coordinate fields
    crossfps_fields = ["LX", "LY", "RX", "RY", "RT", "LT"]
    coord_fields = ["pos_x", "pos_y", "in_attack"]
    is_crossfps = all(f in df.columns for f in crossfps_fields)
    is_coord = any(f in df.columns for f in coord_fields)
    is_compatible = is_crossfps or is_coord
    missing_fields = [] if is_compatible else [f for f in crossfps_fields if f not in df.columns]

    action_col = "action" if "action" in df.columns else ("event" if "event" in df.columns else None)
    action_counts = {}
    if action_col:
        counts = df[action_col].dropna().value_counts().to_dict()
        action_counts = {str(k): int(v) for k, v in counts.items()}
    elif is_crossfps:
        action_counts = {
            "Fire (RT)": int(df.get("RT", [0]).sum()),
            "Scope / Aim (LT)": int(df.get("LT", [0]).sum()),
            "Melee (R3)": int(df.get("R3", [0]).sum()) if "R3" in df.columns else 0,
            "Jump (A)": int(df.get("A", [0]).sum()) if "A" in df.columns else 0,
            "Reload (X)": int(df.get("X", [0]).sum()) if "X" in df.columns else 0,
            "Switch Weapon (Y)": int(df.get("Y", [0]).sum()) if "Y" in df.columns else 0
        }

    total_actions = max(1, sum(action_counts.values()))
    combat_actions = {"Fire", "Reload", "Switch Weapon", "Scope", "Grenade", "Fire (RT)", "Scope / Aim (LT)", "Reload (X)", "Switch Weapon (Y)", "Melee (R3)"}
    movement_actions = {"Move", "Jump", "Crouch", "Jump (A)"}

    combat_cnt = sum(action_counts.get(a, 0) for a in combat_actions)
    move_cnt = sum(action_counts.get(a, 0) for a in movement_actions)
    other_cnt = max(0, total_actions - combat_cnt - move_cnt)

    # Timeline of first 50 records
    timeline = []
    sample_df = df.head(50)
    for idx, row in sample_df.iterrows():
        act = str(row.get(action_col, "Move")) if action_col else ("Fire" if row.get("RT", 0) > 0.5 else "Neutral")
        timeline.append({
            "step": int(idx),
            "tick": int(row.get("tick", idx)),
            "seconds": round(float(row.get("seconds", idx * 0.02)), 3),
            "action": act,
            "weapon": str(row.get("active_weapon", "Gamepad Controller")),
            "pos": [round(float(row.get("LX", row.get("pos_x", 0))), 2), round(float(row.get("LY", row.get("pos_y", 0))), 2)]
        })

    saved_path = USER_UPLOAD_DIR / "active_user_telemetry.csv"
    df.to_csv(saved_path, index=False)

    _CURRENT_PROFILE = {
        "has_profile": True,
        "is_dataset_clip": False,
        "filename": filename,
        "clip_id": None,
        "total_records": len(df),
        "player_id": str(df["player_id"].iloc[0]) if "player_id" in df.columns else "Personal_Gamer",
        "source_game": "User Uploaded Telemetry",
        "map": "User Telemetry Session",
        "action_counts": action_counts,
        "movement_vs_combat": {
            "movement_pct": round((move_cnt / total_actions) * 100, 1),
            "combat_pct": round((combat_cnt / total_actions) * 100, 1),
            "utility_pct": round((other_cnt / total_actions) * 100, 1)
        },
        "timeline": timeline,
        "schema_compatible": is_compatible,
        "missing_fields": missing_fields
    }

    return {
        "status": "success",
        "message": f"Successfully imported {len(df)} records from {filename}.",
        "schema_compatible": is_compatible,
        "missing_fields": missing_fields,
        "profile": _CURRENT_PROFILE
    }


@router.get("/statistics")
def get_gameplay_statistics():
    if not _CURRENT_PROFILE["has_profile"]:
        raise HTTPException(
            status_code=404,
            detail="No gameplay data loaded yet. Please select a genuine CrossFPS clip or upload a valid telemetry export on /gameplay."
        )
    return _CURRENT_PROFILE
