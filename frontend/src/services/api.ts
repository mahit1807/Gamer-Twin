/**
 * GAMER-TWIN Central API Client
 * Connects React frontend directly to the FastAPI deep learning backend.
 */

const API_BASE = '/api';

export interface SystemStatus {
  project_name: string;
  full_title: string;
  subtitle: string;
  institution: string;
  department: string;
  students: string[];
  faculty: string;
  dataset_repo_cloned: boolean;
  dataset_repo_path: string;
  real_dataset_present: boolean;
  sample_dataset_present: boolean;
  active_dataset_source: string;
  pytorch_device: string;
  total_models: number;
  trained_checkpoints: string[];
  api_version: string;
}

export interface DatasetFileInfo {
  filename: string;
  path: string;
  size_bytes: number;
  size_formatted: string;
  is_sample_fixture: boolean;
  format: string;
}

export interface DatasetSummary {
  repo_cloned: boolean;
  repo_path: string;
  dataset_source: string;
  is_synthetic: boolean;
  detected_files: DatasetFileInfo[];
  total_records: number;
  usable_records: number;
  excluded_records: number;
  features_count: number;
  feature_names: string[];
  target_classes: string[];
  class_distribution: Record<string, number>;
  missing_values_summary: Record<string, number>;
  validation_status: string;
  validation_warnings: string[];
  pipeline_split_info?: {
    split_counts?: {
      train_sequences: number;
      val_sequences: number;
      test_sequences: number;
      total_sequences: number;
    };
    records_audit?: Record<string, any>;
  };
}

export interface ModelInfo {
  id: string;
  name: string;
  category: string;
  description: string;
  is_primary: boolean;
  checkpoint_available: boolean;
  checkpoint_path?: string;
  default_hyperparameters: Record<string, any>;
}

export interface TrainingProgress {
  is_training: boolean;
  model_type?: string;
  current_epoch: number;
  total_epochs: number;
  train_loss?: number;
  val_loss?: number;
  train_acc?: number;
  val_acc?: number;
  stage: string;
  history?: {
    epochs: number[];
    train_loss: number[];
    val_loss: number[];
    train_acc: number[];
    val_acc: number[];
  };
  message: string;
}

export interface RankedProbability {
  rank: number;
  class_name: string;
  probability: number;
  percentage: number;
}

export interface PredictionTrace {
  sequence_id: string;
  player_id: string;
  round_id: string;
  model_name: string;
  checkpoint_file: string;
  previous_observed_actions: string[];
  predicted_next_action: string;
  confidence: number;
  probability_distribution: Record<string, number>;
  ranked_probabilities: RankedProbability[];
  ground_truth_action?: string;
  matches_ground_truth?: boolean;
  timestamp: string;
}

export interface ReplayTelemetryStep {
  step_idx: number;
  total_steps: number;
  tick: number;
  seconds: number;
  player_id: string;
  round_id: string;
  pos_x: number;
  pos_y: number;
  pos_z: number;
  vel_x: number;
  vel_y: number;
  vel_z: number;
  view_pitch: number;
  view_yaw: number;
  inputs: Record<string, number>;
  active_weapon: string;
  ammo: number;
  health: number;
  recent_actions: string[];
  actual_next_action: string;
  prediction_trace?: PredictionTrace;
  data_source: string;
  is_synthetic: boolean;
}

export interface ModelComparisonItem {
  model_variant: string;
  model_display_name: string;
  data_provenance: string;
  checkpoint?: string;
  continuous_mse?: number;
  multi_label_accuracy?: number;
  accuracy: number;
  top_3_accuracy?: number;
  f1_macro: number;
  f1_weighted: number;
  precision_macro: number;
  recall_macro: number;
  test_samples: number;
  status: string;
  is_available?: boolean;
  per_action_metrics?: Record<string, { f1: number; precision: number; recall: number; positive_rate: number }>;
  per_class_metrics?: Record<string, { precision: number; recall: number; f1_score: number; support: number }>;
  confusion_matrix?: number[][];
  training_loss_history?: number[];
  validation_loss_history?: number[];
}


export interface ActionTransition {
  from_action: string;
  to_action: string;
  probability: number;
  count: number;
}

export interface PlayerSignature {
  player_id: string;
  data_source: string;
  is_synthetic: boolean;
  total_observed_events: number;
  action_frequencies: Record<string, number>;
  action_proportions: Record<string, number>;
  movement_vs_combat_ratio: {
    movement_percentage: number;
    combat_percentage: number;
    utility_percentage: number;
  };
  top_transitions: ActionTransition[];
  common_ngrams: Array<{
    ngram_type: string;
    sequence: string;
    count: number;
    frequency_pct: number;
  }>;
  window_entropy: Array<{
    window_quarter: string;
    sample_count: number;
    entropy: number;
    dominant_action: string;
    combat_intensity: number;
  }>;
}

export interface DatasetGateStatus {
  gate_status: string;
  gate_message: string;
  active_data_source: string;
  is_synthetic: boolean;
  repo_clone_status: boolean;
  total_records: number;
  required_next_actions: string[];
}

export interface UserGameplayProfile {
  has_profile: boolean;
  is_dataset_clip?: boolean;
  filename: string | null;
  clip_id?: string | null;
  total_records: number;
  player_id: string;
  source_game?: string;
  map: string;
  action_counts: Record<string, number>;
  movement_vs_combat: {
    movement_pct: number;
    combat_pct: number;
    utility_pct: number;
  };
  timeline: Array<{
    step: number;
    tick: number;
    seconds: number;
    action: string;
    weapon: string;
    pos: [number, number];
  }>;
  schema_compatible: boolean;
  missing_fields: string[];
}

export interface WhatIfResponse {
  sequence_id: string;
  model_name: string;
  modifications: Record<string, string>;
  baseline_prediction: {
    predicted_action: string;
    confidence: number;
    confidence_pct: number;
  };
  what_if_prediction: {
    predicted_action: string;
    confidence: number;
    confidence_pct: number;
  };
  action_changed: boolean;
  probability_shifts_pct: Record<string, number>;
  ranked_hypothetical_probabilities: RankedProbability[];
}

export interface PredictionModelInfo {
  id: string;
  name: string;
  role: string;
  is_primary: boolean;
  description: string;
  checkpoint: string;
  checkpoint_exists: boolean;
  can_load: boolean;
  is_available: boolean;
  status: string;
  load_error?: string | null;
  evaluation?: {
    multi_label_accuracy_pct: number;
    macro_f1: number;
    weighted_f1: number;
    precision_macro: number;
    recall_macro: number;
    continuous_mse: number;
    test_samples: number;
    evaluation_split: string;
  } | null;
  disclosure?: string;
}

export interface GameInfo {
  id: string;
  name: string;
  genre: string;
  tagline: string;
  description: string;
  clip_count: number;
  sequence_count: number;
  is_available: boolean;
  status: string;
}

export interface GameSequence {
  sequence_index: number;
  sequence_id: string;
  clip_id: string;
  source_game: string;
  game_name: string;
  window_size: number;
  feature_dim: number;
  ground_truth_action: string;
  recent_actions: string[];
  preview_feature_mean: number;
}

export interface ConditionOverrides {
  aim_down_sights?: boolean;
  fire?: boolean;
  jump?: boolean;
  reload?: boolean;
  switch_weapon?: boolean;
  melee?: boolean;
  movement?: 'forward' | 'backward' | 'strafe_left' | 'strafe_right' | 'stationary';
  camera?: 'turn_left' | 'turn_right' | 'look_up' | 'look_down' | 'steady';
}

export interface PredictResponse {
  mode: string;
  mode_label: string;
  game_id: string;
  game_name: string;
  trace: PredictionTrace;
  continuous_prediction: {
    strafe_lx: number;
    move_ly: number;
    turn_yaw_rx: number;
    look_pitch_ry: number;
    movement_summary: string;
    camera_summary: string;
  };
  explanation: string;
  applied_conditions: string[];
  status?: string;
  message?: string;
}

export interface WhatIfDetailedResponse {
  sequence_id: string;
  model_name: string;
  modifications: Record<string, string>;
  baseline_prediction: {
    predicted_action: string;
    confidence: number;
    confidence_pct: number;
    continuous: {
      strafe_lx: number;
      move_ly: number;
      turn_yaw_rx: number;
      look_pitch_ry: number;
      movement_summary?: string;
      camera_summary?: string;
    };
    discrete_actions?: string[];
  };
  what_if_prediction: {
    predicted_action: string;
    confidence: number;
    confidence_pct: number;
    continuous: {
      strafe_lx: number;
      move_ly: number;
      turn_yaw_rx: number;
      look_pitch_ry: number;
      movement_summary?: string;
      camera_summary?: string;
    };
    discrete_actions?: string[];
  };
  action_changed: boolean;
  probability_shifts_pct: Record<string, number>;
  continuous_shifts?: {
    strafe_lx_diff: number;
    move_ly_diff: number;
    turn_yaw_rx_diff: number;
    look_pitch_ry_diff: number;
  };
  ranked_hypothetical_probabilities: RankedProbability[];
}

export interface ControllerTelemetryStats {
  dominant_action: string;
  mean_strafe_lx: number;
  mean_move_ly: number;
  mean_turn_rx: number;
  mean_pitch_ry: number;
  fire_rate: number;
  ads_rate: number;
  jump_rate: number;
  movement_label: string;
  camera_label: string;
}

export interface VerifiedDemoSequenceResult {
  checkpoint: string;
  model_name: string;
  predicted_continuous: {
    strafe_lx: number;
    move_ly: number;
    turn_yaw_rx: number;
    look_pitch_ry: number;
  };
  predicted_discrete_probs: Record<string, number>;
  predicted_actions: string[];
  predicted_action_label: string;
  continuous_mse: number;
  discrete_match: boolean;
  evaluation_rule: string;
  verification_status: string;
  passed: boolean;
}

export interface VerifiedDemoSequence {
  sequence_id: string;
  sequence_index: number;
  clip_id: string;
  source_game: string;
  game_name: string;
  input_frame_range: [number, number];
  target_frame: number;
  ground_truth: {
    continuous: {
      strafe_lx: number;
      move_ly: number;
      turn_yaw_rx: number;
      look_pitch_ry: number;
    };
    discrete: {
      rt_fire: number;
      lt_ads: number;
      r3_melee: number;
      a_jump: number;
      x_reload: number;
      y_switch: number;
    };
    active_actions: string[];
    action_label: string;
  };
  model_results: Record<string, VerifiedDemoSequenceResult>;
}


// Fetch helper with error handling
async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${url}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });
  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(errorBody.detail || `Request failed with status ${response.status}`);
  }
  return response.json();
}

export const api = {
  // System
  getSystemStatus: () => fetchJson<SystemStatus>('/system/status'),

  // Dataset
  getDatasetGateStatus: () => fetchJson<DatasetGateStatus>('/dataset/status'),
  getDatasetAudit: () => fetchJson<any>('/dataset/audit'),
  getDatasetSummary: () => fetchJson<DatasetSummary>('/dataset/summary'),
  getDatasetPreview: (limit = 15, fileType = 'ticks') =>
    fetchJson<{ filename: string; total_columns: number; columns: string[]; rows: any[] }>(
      `/dataset/preview?limit=${limit}&file_type=${fileType}`
    ),
  validateDataset: () => fetchJson<any>('/dataset/validate', { method: 'POST' }),

  // Models
  getModels: () => fetchJson<ModelInfo[]>('/models'),
  getTrainingStatus: () => fetchJson<TrainingProgress>('/models/training-status'),
  startTraining: (params: { model_type: string; epochs: number; batch_size: number; learning_rate: number; sequence_length: number }) =>
    fetchJson<{ status: string; model_type: string; epochs: number }>('/models/train', {
      method: 'POST',
      body: JSON.stringify(params),
    }),
  getModelHistory: (modelName: string) => fetchJson<any>(`/models/history/${modelName}`),

  // Prediction
  getPredictionModels: () => fetchJson<PredictionModelInfo[]>('/prediction/models'),
  getSupportedGames: () => fetchJson<{ total_games: number; games: GameInfo[] }>('/prediction/games'),
  getPredictionSequences: (limit = 50, game?: string) =>
    fetchJson<{ total_test_sequences: number; returned_sequences: number; game_filter?: string; sequences: GameSequence[] }>(
      `/prediction/sequences?limit=${limit}${game ? `&game=${encodeURIComponent(game)}` : ''}`
    ),
  predictNextAction: (params: { model_name: string; sequence_index?: number; custom_sequence?: number[][] }) =>
    fetchJson<PredictionTrace>('/prediction/predict', {
      method: 'POST',
      body: JSON.stringify(params),
    }),
  predictNextActionMode: (
    params: {
      model_name: string;
      sequence_index?: number;
      custom_sequence?: number[][];
      condition_overrides?: ConditionOverrides;
    },
    mode: 'dataset' | 'personalized' = 'dataset'
  ) =>
    fetchJson<PredictResponse>(
      `/prediction/predict?mode=${mode}`,
      {
        method: 'POST',
        body: JSON.stringify(params),
      }
    ),
  predictWhatIf: (params: {
    model_name: string;
    sequence_index: number;
    modified_weapon?: string;
    modified_posture?: string;
    modified_scope?: boolean;
    condition_overrides?: ConditionOverrides;
  }) =>
    fetchJson<WhatIfDetailedResponse>('/predict/what-if', {
      method: 'POST',
      body: JSON.stringify(params),
    }),

  // User Gameplay Profile, Dataset Clips & Upload
  getUserProfile: () => fetchJson<UserGameplayProfile>('/gameplay/profile'),
  getGameplayStatistics: () => fetchJson<UserGameplayProfile>('/gameplay/statistics'),
  getDatasetClips: (offset = 0, limit = 30, game?: string) =>
    fetchJson<{ total: number; offset: number; limit: number; clips: Array<{ clip_id: string; game: string; has_raw_parquet: boolean; path: string }> }>(
      `/gameplay/clips?offset=${offset}&limit=${limit}${game ? `&game=${encodeURIComponent(game)}` : ''}`
    ),
  loadDatasetClip: (clipId: string) => fetchJson<UserGameplayProfile>(`/gameplay/clip/${encodeURIComponent(clipId)}`),
  uploadGameplay: async (file: File): Promise<{ status: string; message: string; schema_compatible: boolean; missing_fields: string[]; profile: UserGameplayProfile }> => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch(`${API_BASE}/gameplay/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) {
      const errorBody = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(errorBody.detail || `Upload failed with status ${response.status}`);
    }
    return response.json();
  },

  // Replay
  getReplayInfo: () =>
    fetchJson<{
      total_records: number;
      total_steps: number;
      window_size: number;
      players: string[];
      rounds: string[];
      data_source: string;
      is_synthetic: boolean;
      video_available: boolean;
      video_notice: string;
    }>('/replay/info'),
  getReplayStep: (stepIdx: number, modelName = 'transformer') =>
    fetchJson<ReplayTelemetryStep>(`/replay/step?step_idx=${stepIdx}&model_name=${modelName}`),
  getReplayTimeline: (limit = 60) =>
    fetchJson<{ total_events: number; events: any[]; is_synthetic: boolean }>(`/replay/timeline?limit=${limit}`),

  // Experiments
  getExperimentComparison: () => fetchJson<ModelComparisonItem[]>('/experiments/comparison'),
  getSingleModelReport: (modelName: string) => fetchJson<any>(`/experiments/model/${modelName}`),
  runAllEvaluations: () => fetchJson<{ status: string; evaluated_models: string[] }>('/experiments/evaluate-all', { method: 'POST' }),

  // Signature
  getPlayerSignature: (playerId?: string) =>
    fetchJson<PlayerSignature>(`/signature${playerId ? `?player_id=${encodeURIComponent(playerId)}` : ''}`),

  // Methodology
  getMethodologyInfo: () => fetchJson<any>('/methodology/info'),

  // Tested Presentation Demo Sequences
  getVerifiedDemoSequences: () =>
    fetchJson<{ total_verified: number; evaluation_rule_summary: string; sequences: VerifiedDemoSequence[] }>(
      '/predict/verified-demo-sequences'
    ),
};

