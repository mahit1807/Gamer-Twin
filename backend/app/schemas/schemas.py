"""
Pydantic Schemas for GAMER-TWIN API Endpoints.
Ensures strong typing, request validation, and self-documenting OpenAPI schemas.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


# --- System Schemas ---
class SystemStatusResponse(BaseModel):
    project_name: str
    full_title: str
    subtitle: str
    institution: str
    department: str
    students: List[str]
    faculty: str
    dataset_repo_cloned: bool
    dataset_repo_path: str
    real_dataset_present: bool
    sample_dataset_present: bool
    active_dataset_source: str
    pytorch_device: str
    total_models: int
    trained_checkpoints: List[str]
    api_version: str


# --- Dataset Schemas ---
class DatasetFileInfo(BaseModel):
    filename: str
    path: str
    size_bytes: int
    size_formatted: str
    is_sample_fixture: bool
    format: str


class DatasetSummaryResponse(BaseModel):
    repo_cloned: bool
    repo_path: str
    dataset_source: str
    is_synthetic: bool
    detected_files: List[DatasetFileInfo]
    total_records: int
    usable_records: int
    excluded_records: int
    features_count: int
    feature_names: List[str]
    target_classes: List[str]
    class_distribution: Dict[str, int]
    missing_values_summary: Dict[str, int]
    validation_status: str
    validation_warnings: List[str]
    pipeline_split_info: Optional[Dict[str, Any]] = None


# --- Model Schemas ---
class ModelArchitectureInfo(BaseModel):
    id: str
    name: str
    category: str
    description: str
    is_primary: bool
    checkpoint_available: bool
    checkpoint_path: Optional[str] = None
    default_hyperparameters: Dict[str, Any]


class TrainRequest(BaseModel):
    model_type: str = Field(..., description="lstm_baseline, lstm_transformer, transformer, or transformer_ssl")
    epochs: int = Field(5, ge=1, le=50)
    batch_size: int = Field(32, ge=8, le=256)
    learning_rate: float = Field(0.001, ge=0.00001, le=0.1)
    sequence_length: int = Field(16, ge=4, le=64)


class TrainingProgressResponse(BaseModel):
    is_training: bool
    model_type: Optional[str] = None
    current_epoch: int = 0
    total_epochs: int = 0
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    train_acc: Optional[float] = None
    val_acc: Optional[float] = None
    stage: str = "idle"
    history: Optional[Dict[str, List[float]]] = None
    message: str = "Ready"


# --- Prediction & Trace Schemas ---
class RankedProbability(BaseModel):
    rank: int
    class_name: str
    probability: float
    percentage: float


class PredictionTrace(BaseModel):
    sequence_id: str
    player_id: str
    round_id: str
    model_name: str
    checkpoint_file: str
    previous_observed_actions: List[str]
    predicted_next_action: str
    confidence: float
    probability_distribution: Dict[str, float]
    ranked_probabilities: List[RankedProbability]
    ground_truth_action: Optional[str] = None
    matches_ground_truth: Optional[bool] = None
    timestamp: str


class PredictRequest(BaseModel):
    model_name: str
    sequence_index: Optional[int] = None
    custom_sequence: Optional[List[List[float]]] = None
    condition_overrides: Optional[Dict[str, Any]] = None
    clip_id: Optional[str] = None


# --- Replay Schemas ---
class ReplayTelemetryStep(BaseModel):
    step_idx: int
    total_steps: int
    tick: int
    seconds: float
    player_id: str
    round_id: str
    pos_x: float
    pos_y: float
    pos_z: float
    vel_x: float
    vel_y: float
    vel_z: float
    view_pitch: float
    view_yaw: float
    inputs: Dict[str, int]
    active_weapon: str
    ammo: int
    health: int
    recent_actions: List[str]
    actual_next_action: str
    prediction_trace: Optional[PredictionTrace] = None
    data_source: str
    is_synthetic: bool


# --- Experiments Schemas ---
class ModelEvaluationSummary(BaseModel):
    model_variant: str
    model_display_name: str
    data_provenance: str
    accuracy: float
    top_3_accuracy: float
    f1_macro: float
    f1_weighted: float
    precision_macro: float
    recall_macro: float
    test_samples: int
    status: str
    per_class_metrics: Optional[Dict[str, Dict[str, float]]] = None
    confusion_matrix: Optional[List[List[int]]] = None
    training_loss_history: Optional[List[float]] = None
    validation_loss_history: Optional[List[float]] = None


# --- Player Behaviour Signature Schemas ---
class ActionTransition(BaseModel):
    from_action: str
    to_action: str
    probability: float
    count: int


class PlayerSignatureResponse(BaseModel):
    player_id: str
    data_source: str
    is_synthetic: bool
    total_observed_events: int
    action_frequencies: Dict[str, int]
    action_proportions: Dict[str, float]
    movement_vs_combat_ratio: Dict[str, float]
    top_transitions: List[ActionTransition]
    common_ngrams: List[Dict[str, Any]]
    window_entropy: List[Dict[str, Any]]
