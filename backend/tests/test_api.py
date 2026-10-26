"""
Backend API Unit Tests for GAMER-TWIN FastAPI service.
Tests health, strict dataset gate, model listing, prediction trace, what-if counterfactuals,
gameplay profile upload, and evaluation reports.
"""

import sys
import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "GAMER-TWIN"
    assert data["status"] == "online"


def test_system_status_endpoint():
    response = client.get("/api/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["project_name"] == "GAMER-TWIN"
    assert data["dataset_repo_cloned"] is True
    assert "students" in data
    assert len(data["trained_checkpoints"]) >= 1


def test_dataset_gate_status_and_audit():
    # 1. Gate status
    status_resp = client.get("/api/dataset/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert "gate_status" in status_data
    assert status_data["gate_status"] in [
        "DATASET_READY", "DATASET_PARTIALLY_AVAILABLE", "DATASET_UNAVAILABLE",
        "DATASET_SCHEMA_ERROR", "INSUFFICIENT_TRAINING_DATA"
    ]
    assert "gate_message" in status_data

    # 2. Audit
    audit_resp = client.get("/api/dataset/audit")
    assert audit_resp.status_code == 200
    audit_data = audit_resp.json()
    assert "features_available" in audit_data


def test_dataset_summary_endpoint():
    response = client.get("/api/dataset/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] > 0
    assert len(data["target_classes"]) in (6, 9)
    assert data["features_count"] > 0


def test_models_list_endpoint():
    response = client.get("/api/models")
    assert response.status_code == 200
    models = response.json()
    assert len(models) == 4
    model_ids = [m["id"] for m in models]
    assert "lstm_baseline" in model_ids
    assert "transformer" in model_ids


def test_prediction_sequences_and_predict():
    # 1. Fetch available sequences
    seq_resp = client.get("/api/prediction/sequences?limit=5")
    assert seq_resp.status_code == 200
    seq_data = seq_resp.json()
    assert seq_data["returned_sequences"] > 0

    # 2. Predict using transformer (Mode A)
    pred_resp = client.post(
        "/api/prediction/predict?mode=dataset",
        json={"model_name": "transformer", "sequence_index": 0}
    )
    assert pred_resp.status_code == 200
    pred_data = pred_resp.json()
    assert "mode" in pred_data
    assert "trace" in pred_data
    trace = pred_data["trace"]
    assert "predicted_next_action" in trace
    assert "confidence" in trace
    assert "probability_distribution" in trace
    assert len(trace["ranked_probabilities"]) in (6, 9)

    # 3. Predict Mode B (personalized check without upload)
    mode_b_resp = client.post(
        "/api/prediction/predict?mode=personalized",
        json={"model_name": "transformer", "sequence_index": 0}
    )
    assert mode_b_resp.status_code == 200


def test_what_if_counterfactual_prediction():
    resp = client.post(
        "/api/predict/what-if",
        json={
            "model_name": "transformer",
            "sequence_index": 0,
            "modified_posture": "crouched",
            "modified_scope": True
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "baseline_prediction" in data
    assert "what_if_prediction" in data
    assert "probability_shifts_pct" in data


def test_gameplay_upload_and_profile():
    # Profile default
    p_resp = client.get("/api/gameplay/profile")
    assert p_resp.status_code == 200

    # Upload test CSV
    csv_content = (
        "tick,seconds,player_id,round_id,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,view_pitch,view_yaw,delta_pitch,delta_yaw,"
        "in_forward,in_back,in_moveleft,in_moveright,in_attack,in_reload,in_jump,in_duck,is_scoped,active_weapon,ammo,health,action\n"
        "1,0.015,test_gamer,round_1,10,20,0,5,0,0,0,90,0,0,1,0,0,0,0,0,0,0,0,ak47,30,100,Move\n"
        "2,0.030,test_gamer,round_1,12,20,0,5,0,0,0,90,0,0,0,0,0,0,1,0,0,0,0,ak47,29,100,Fire\n"
    )
    files = {"file": ("test_gameplay.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    upload_resp = client.post("/api/gameplay/upload", files=files)
    assert upload_resp.status_code == 200
    up_data = upload_resp.json()
    assert up_data["status"] == "success"
    assert up_data["schema_compatible"] is True

    # Check updated stats
    stats_resp = client.get("/api/gameplay/statistics")
    assert stats_resp.status_code == 200
    stats = stats_resp.json()
    assert stats["has_profile"] is True
    assert stats["total_records"] == 2


def test_evaluation_endpoints():
    resp = client.get("/api/evaluation/comparison")
    assert resp.status_code == 200
    summaries = resp.json()
    assert len(summaries) >= 1
    for s in summaries:
        assert "accuracy" in s
        assert "data_provenance" in s


def test_prediction_models_and_games():
    # 1. Models endpoint: exactly 3 models
    models_resp = client.get("/api/prediction/models")
    assert models_resp.status_code == 200
    models = models_resp.json()
    assert len(models) == 3
    ids = [m["id"] for m in models]
    assert "transformer" in ids
    assert "lstm_baseline" in ids
    assert "transformer_ssl" in ids

    # Check all 3 models are available and loadable
    for m in models:
        assert m["checkpoint_exists"] is True
        assert m["can_load"] is True
        assert m["is_available"] is True
        assert m["status"] == "Ready for Inference"
        assert "evaluation" in m
        if m["evaluation"]:
            assert "multi_label_accuracy_pct" in m["evaluation"]
            assert m["evaluation"]["test_samples"] == 10859

    # Check transformer_ssl specific attributes
    ssl_model = next(m for m in models if m["id"] == "transformer_ssl")
    assert ssl_model["checkpoint"] == "transformer_ssl_crossfps_best.pt"
    assert ssl_model["is_available"] is True
    assert ssl_model["status"] == "Ready for Inference"
    assert ssl_model["evaluation"]["multi_label_accuracy_pct"] == 91.2
    assert ssl_model["evaluation"]["macro_f1"] == 0.7063

    # 2. Games endpoint: only games with verified sequences > 0
    games_resp = client.get("/api/prediction/games")
    assert games_resp.status_code == 200
    games_data = games_resp.json()
    assert games_data["total_games"] == 4
    game_ids = [g["id"] for g in games_data["games"]]
    assert "halo_infinite" in game_ids
    assert "call_of_duty__warzone" in game_ids
    assert "call_of_duty__modern_warfare_iii" in game_ids
    assert "call_of_duty" in game_ids
    # Unwanted zero-sequence games are filtered out
    assert "worldcam_xonotic" not in game_ids
    assert "mw" not in game_ids
    assert "halo" not in game_ids


def test_prediction_with_ssl_transformer():
    # Test predict using the genuine trained Self-Supervised Transformer
    pred_resp = client.post(
        "/api/prediction/predict",
        json={
            "model_name": "transformer_ssl",
            "sequence_index": 0
        }
    )
    assert pred_resp.status_code == 200
    p_data = pred_resp.json()
    assert p_data["trace"]["model_name"] == "Self-Supervised Transformer"
    assert p_data["trace"]["checkpoint_file"] == "transformer_ssl_crossfps_best.pt"
    assert "continuous_prediction" in p_data
    assert "strafe_lx" in p_data["continuous_prediction"]
    assert "turn_yaw_rx" in p_data["continuous_prediction"]
    assert len(p_data["trace"]["ranked_probabilities"]) == 6


def test_prediction_with_condition_overrides():
    # Test predict with condition overrides
    pred_resp = client.post(
        "/api/prediction/predict",
        json={
            "model_name": "transformer",
            "sequence_index": 0,
            "condition_overrides": {
                "aim_down_sights": True,
                "movement": "forward"
            }
        }
    )
    assert pred_resp.status_code == 200
    p_data = pred_resp.json()
    assert "applied_conditions" in p_data
    assert len(p_data["applied_conditions"]) >= 1

    # Test what-if with condition overrides
    what_if_resp = client.post(
        "/api/predict/what-if",
        json={
            "model_name": "transformer",
            "sequence_index": 0,
            "condition_overrides": {
                "aim_down_sights": True,
                "fire": True
            }
        }
    )
    assert what_if_resp.status_code == 200
    wi_data = what_if_resp.json()
    assert "baseline_prediction" in wi_data
    assert "what_if_prediction" in wi_data
    assert "probability_shifts_pct" in wi_data
    assert "continuous_shifts" in wi_data


def test_verified_demo_sequences():
    resp = client.get("/api/predict/verified-demo-sequences")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_verified" in data
    assert "sequences" in data
    assert data["total_verified"] >= 5
    first_seq = data["sequences"][0]
    assert "clip_id" in first_seq
    assert "target_frame" in first_seq
    assert "ground_truth" in first_seq
    assert "model_results" in first_seq
    assert "transformer" in first_seq["model_results"]
    assert "evaluation_rule" in first_seq["model_results"]["transformer"]
    assert "passed" in first_seq["model_results"]["transformer"]

