"""
System & Diagnostics API endpoints for GAMER-TWIN.
Reports real environment status, dataset repo availability, checkpoints, and hardware capabilities.
"""

import os
from pathlib import Path
from fastapi import APIRouter
import torch

from src.utils import load_config, get_device
from backend.app.schemas.schemas import SystemStatusResponse

router = APIRouter()

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@router.get("/status", response_model=SystemStatusResponse)
def get_system_status():
    config = load_config()

    # Check EgoCS-400K official cloned repository
    dataset_repo_dir = WORKSPACE_ROOT / "Dataset"
    dataset_repo_cloned = dataset_repo_dir.exists() and (dataset_repo_dir / ".git").exists()

    # Check CrossFPS validation dataset
    crossfps_dir = WORKSPACE_ROOT / "data" / "CrossFPS-val"
    has_crossfps = (crossfps_dir / "extracted" / "eval").exists()

    if has_crossfps:
        active_dataset_source = "CrossFPS-val (Real 10-DoF Telemetry / 3,443 Clips)"
        has_real_data = True
        has_sample_data = False
    else:
        active_dataset_source = "No Telemetry Data Detected"
        has_real_data = False
        has_sample_data = False

    # Check checkpoints
    ckpt_dir = WORKSPACE_ROOT / "outputs" / "checkpoints"
    trained_checkpoints = []
    if ckpt_dir.exists():
        trained_checkpoints = [f.name for f in ckpt_dir.glob("*.pt")]

    device = get_device()

    return SystemStatusResponse(
        project_name="GAMER-TWIN",
        full_title="A Personalized Deep Learning Framework for Modeling and Predicting Player Behaviour",
        subtitle="A Transformer-Based Digital Twin for Next-Action Prediction in Interactive Games",
        institution="Symbiosis Institute of Technology, Pune",
        department="Department of Artificial Intelligence and Machine Learning",
        students=["Mahit Turakhia", "Het Shah"],
        faculty="Dr. Ananya Bhattacharjee",
        dataset_repo_cloned=dataset_repo_cloned,
        dataset_repo_path=str(dataset_repo_dir),
        real_dataset_present=has_real_data,
        sample_dataset_present=has_sample_data,
        active_dataset_source=active_dataset_source,
        pytorch_device=device,
        total_models=4,
        trained_checkpoints=trained_checkpoints,
        api_version="1.0.0"
    )
