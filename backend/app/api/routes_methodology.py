"""
Methodology & Educational documentation API for GAMER-TWIN.
Provides formal research methodology, feature mappings, loss formulations,
and architecture descriptions.
"""

from fastapi import APIRouter
from src.utils import PROJECT_CLASSES

router = APIRouter()

METHODOLOGY_DETAILS = {
    "project": {
        "name": "GAMER-TWIN",
        "full_title": "A Personalized Deep Learning Framework for Modeling and Predicting Player Behaviour",
        "subtitle": "A Transformer-Based Digital Twin for Next-Action Prediction in Interactive Games",
        "institution": "Symbiosis Institute of Technology, Pune (AIML Department)",
        "authors": ["Mahit Turakhia", "Het Shah"],
        "faculty_advisor": "Dr. Ananya Bhattacharjee"
    },
    "dataset": {
        "name": "EgoCS-400K",
        "url": "https://github.com/EgoCS-400K/Dataset",
        "huggingface": "https://huggingface.co/spaces/Cooler-Master/cs2-action-annotation-viewer-preview",
        "description": "A 10,000+ hour synchronized egocentric Counter-Strike gameplay dataset aligning first-person POV video with 64Hz/32Hz tick telemetry, keyboard/mouse inputs, and atomic action annotations.",
        "cloned_repo_status": "Cloned into /Dataset in development workspace"
    },
    "target_classes": {
        "count": 9,
        "classes": PROJECT_CLASSES,
        "note": "These 9 classes are project-defined targets synthesized from raw player controls and synchronized game events to represent distinct tactical gameplay behaviors."
    },
    "feature_dimensions": {
        "total_dim": 23,
        "features": [
            {"name": "pos_x, pos_y, pos_z", "type": "Continuous (Z-score normalized)", "description": "3D spatial coordinates in the game map."},
            {"name": "vel_x, vel_y, vel_z", "type": "Continuous (Z-score normalized)", "description": "Velocity vectors indicating player momentum."},
            {"name": "view_pitch, view_yaw", "type": "Continuous (Z-score normalized)", "description": "Camera orientation and gaze angles."},
            {"name": "delta_pitch, delta_yaw", "type": "Continuous (Z-score normalized)", "description": "Frame-to-frame change in camera orientation (mouse movement)."},
            {"name": "in_forward, in_back, in_moveleft, in_moveright", "type": "Binary {0, 1}", "description": "Keyboard inputs (W, S, A, D)."},
            {"name": "in_attack, in_reload, in_jump, in_duck", "type": "Binary {0, 1}", "description": "Control actions (Fire, Reload, Jump, Crouch)."},
            {"name": "is_scoped", "type": "Binary {0, 1}", "description": "Weapon optical zoom scope state."},
            {"name": "active_weapon", "type": "One-hot / Categorical index", "description": "Current weapon equipped by player."},
            {"name": "ammo", "type": "Continuous normalized", "description": "Remaining ammunition in magazine."},
            {"name": "health", "type": "Continuous normalized", "description": "Player health points [0, 100]."},
            {"name": "prev_action", "type": "Categorical embedding / index", "description": "Immediate prior atomic action."}
        ]
    },
    "sequence_construction": {
        "window_size": 16,
        "stride": 2,
        "sampling_rate": "32 Hz / 64 Hz tick aggregation",
        "target_offset": 1,
        "description": "Temporal sliding window of length T=16 past telemetry vectors used to forecast the action at time step T+1."
    },
    "architectures": [
        {
            "id": "lstm_baseline",
            "name": "Model 1: LSTM Baseline",
            "role": "Recurrent sequence baseline",
            "formulation": "h_t = LSTM(x_t, h_{t-1}), \\hat{y} = Softmax(W \\cdot h_T + b)",
            "description": "Standard 2-layer LSTM sequence model that processes frame by frame and uses the final hidden state to predict the next action."
        },
        {
            "id": "lstm_transformer",
            "name": "Model 2: LSTM + Transformer Hybrid",
            "role": "Inductive bias + Self-Attention",
            "formulation": "z = LSTM(x), e = MultiHeadAttention(z + PE), \\hat{y} = Softmax(MLP(MeanPool(e)))",
            "description": "Combines local temporal inductive bias of recurrence for smooth camera/movement tracking with multi-head self-attention for global tactical context."
        },
        {
            "id": "transformer",
            "name": "Model 3: Transformer Encoder (Primary)",
            "role": "Primary predictive digital twin",
            "formulation": "z_0 = W_e x + PE, z_l = TransformerLayer(z_{l-1}), \\hat{y} = Softmax(MLP(0.5(MeanPool(z_L) + z_L[T])))",
            "description": "Pure Transformer architecture with sinusoidal positional encodings, multi-head self-attention, and dual pooling combining sequence mean with the final frame token."
        },
        {
            "id": "transformer_ssl",
            "name": "Model 4: Transformer + Self-Supervised Learning",
            "role": "Self-supervised foundation",
            "formulation": "L_{pretrain} = \\frac{1}{|M|} \\sum_{t \\in M} || \\hat{x}_t - x_t ||^2, L_{finetune} = - \\sum c_k \\log(\\hat{y}_k)",
            "description": "Two-stage learning strategy: Stage 1 pretrains the Transformer backbone via Masked Autoencoding on historical telemetry frames; Stage 2 fine-tunes on supervised next-action prediction."
        }
    ],
    "reinforcement_learning_scope": {
        "current_approach": "Supervised Behavioral Modeling & Imitation / Next-Action Forecasting",
        "is_game_controller": False,
        "dqn_status": "Future Work Only",
        "distinction": "GAMER-TWIN models an individual player's behavioral patterns and forecasts their next discrete action given telemetry history. It is an analytical predictive digital twin, not an interactive RL game-playing bot or live game controller."
    }
}


@router.get("/info")
def get_methodology_info():
    return METHODOLOGY_DETAILS
