"""
GAMER-TWIN FastAPI Application Main Entry Point
Symbiosis Institute of Technology, Pune - Department of AIML
A Personalized Deep Learning Framework for Modeling and Predicting Player Behaviour
"""

import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure workspace root and src/ are in python path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.app.api import (
    routes_system,
    routes_dataset,
    routes_models,
    routes_prediction,
    routes_replay,
    routes_experiments,
    routes_signature,
    routes_methodology,
    routes_gameplay,
    routes_embeddings,
)

app = FastAPI(
    title="GAMER-TWIN Research API",
    description="Deep Learning Digital Twin API for Next-Action Prediction in Interactive Games (EgoCS-400K)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for local Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(routes_system.router, prefix="/api/system", tags=["System"])
app.include_router(routes_dataset.router, prefix="/api/dataset", tags=["Dataset"])
app.include_router(routes_models.router, prefix="/api/models", tags=["Models"])
app.include_router(routes_prediction.router, prefix="/api/prediction", tags=["Prediction"])
app.include_router(routes_prediction.router, prefix="/api/predict", tags=["Prediction"])
app.include_router(routes_replay.router, prefix="/api/replay", tags=["Replay"])
app.include_router(routes_experiments.router, prefix="/api/experiments", tags=["Experiments"])
app.include_router(routes_experiments.router, prefix="/api/evaluation", tags=["Evaluation"])
app.include_router(routes_signature.router, prefix="/api/signature", tags=["Player Signature"])
app.include_router(routes_methodology.router, prefix="/api/methodology", tags=["Methodology"])
app.include_router(routes_gameplay.router, prefix="/api/gameplay", tags=["Gameplay"])
app.include_router(routes_embeddings.router, prefix="/api/embeddings", tags=["Embeddings"])


@app.get("/")
def root():
    return {
        "project": "GAMER-TWIN",
        "title": "A Personalized Deep Learning Framework for Modeling and Predicting Player Behaviour",
        "institution": "Symbiosis Institute of Technology, Pune - AIML",
        "authors": ["Mahit Turakhia", "Het Shah"],
        "faculty_advisor": "Dr. Ananya Bhattacharjee",
        "api_docs": "/docs",
        "status": "online"
    }
