# GAMER-TWIN — Synthetic and Placeholder Data Audit & Inventory

## 1. Executive Summary
This audit catalogues every instance of synthetic data, mock fixtures, hardcoded gameplay records, and placeholder logic across the GAMER-TWIN codebase prior to the real-data refactor. Under the strict real-data refactor mandate, every data-driven component will derive exclusively from the 3,443 extracted CrossFPS validation clips (`data/CrossFPS-val/extracted/eval/`) and verified PyTorch model checkpoints (`transformer_crossfps_best.pt` and `lstm_crossfps_best.pt`).

---

## 2. Inventory of Synthetic Data Sources

### A. Frontend (`frontend/src/`)
| File & Location | Synthetic / Placeholder Item | Purpose / Description | Action Required |
|---|---|---|---|
| `frontend/src/pages/MyGameplay.tsx` (Lines 110-136) | `loadSampleProfile` with hardcoded `UserGamer_77` CSV string (`de_dust2`, fake ticks) | Demo button creating mock player telemetry in memory | **Remove entirely**. Replace with genuine CrossFPS clip browser. |
| `frontend/src/pages/MyGameplay.tsx` (Line 154) | Label: `Personalized Behavioural Profile · EgoCS-400K Telemetry` | Stale reference to CS2/EgoCS-400K | Update to real CrossFPS controller telemetry. |
| `frontend/src/pages/PredictionStudio.tsx` (Lines 51-61) | Hardcoded `SUPPORTED_WEAPONS`, `selectedMap: 'de_dust2'`, `selectedWeapon: 'ak47'` | Fabricated game context selectors not present in CrossFPS | Replace with real `source_game` metadata and genuine 10-DoF channels. |
| `frontend/src/pages/PredictionStudio.tsx` (Lines 200-216) | What-If assumptions based on `in_duck` and `is_scoped` CS2 columns | Hypothesized feature shifts | Connect to real 10-DoF channels (`LT` Scope, `A` Jump, continuous yaw). |

### B. Backend API (`backend/app/api/`)
| File & Location | Synthetic / Placeholder Item | Purpose / Description | Action Required |
|---|---|---|---|
| `backend/app/api/routes_gameplay.py` (Lines 26-37) | `_CURRENT_PROFILE` default state with `"player_id": "Guest_Gamer"`, `"map": "de_dust2"` | Fabricated fallback profile | Remove fabricated defaults; require selecting a real dataset clip or valid upload. |
| `backend/app/api/routes_prediction.py` (Line 54) | `load_test_sequences()` loads `outputs/processed/processed_splits.npz` | Old synthetic 23-feature split fixture | Switch to `outputs/processed/crossfps_splits.npz` (real 10-DoF CrossFPS test split). |
| `backend/app/api/routes_prediction.py` (Lines 204-216) | What-If modifies hardcoded column indices 19 and 20 | Old coordinate feature indices | Refactor to modify genuine 10-DoF channels and infer through genuine PyTorch model. |
| `backend/app/api/routes_dataset.py` (Lines 23-31, 102) | `get_active_data_dir()` checks `data/sample_data/ticks.csv` and `data/raw` | Old CS2 sample fixture | Wire to authoritative CrossFPS dataset loader (`data/CrossFPS-val/extracted/eval/`). |
| `backend/app/api/routes_dataset.py` (Lines 129, 141) | `SYNTHETIC_TEST_FIXTURE` strings and checks | References to mock data generator | Remove and derive genuine CrossFPS validation dataset summary. |

### C. Pipeline & Legacy Fixtures (`data/`, `src/`)
| File & Location | Synthetic / Placeholder Item | Purpose / Description | Action Required |
|---|---|---|---|
| `data/sample_data/` | `ticks.csv`, `events.csv`, `player_meta.json` | 2,000 synthetic CS2 telemetry rows created for early prototype | Isolate and bypass. Direct all active routes to `data/CrossFPS-val/`. |
| `outputs/processed/processed_splits.npz` | Synthetic 23-feature splits | Legacy test sequences | Replace in active prediction with `outputs/processed/crossfps_splits.npz`. |

---

## 3. Strict Real-Data Replacement Architecture
1. **Authoritative Dataset Service**: `backend/app/services/crossfps_service.py` dynamically scans `data/CrossFPS-val/extracted/eval/` and reads `metadata_eval.csv`. All counts (clips, sequences, frames) are calculated dynamically at runtime from disk.
2. **Model Verification & Inference**: `backend/app/services/crossfps_predictor.py` loads `outputs/checkpoints/transformer_crossfps_best.pt` and `lstm_crossfps_best.pt`, feeding genuine 10-DoF sequence windows ($T=16$).
3. **My Gameplay (`/gameplay`)**: Replaced mock profile loader with a **Real CrossFPS Clip Explorer** allowing interactive selection of actual clips across Halo Infinite, Modern Warfare, Warzone, etc., displaying genuine 100-frame telemetry timelines and action frequencies.
4. **Prediction Studio (`/predict`)**: Directly connected to real test sequences from `crossfps_splits.npz`. Live PyTorch inference produces genuine continuous analog predictions and discrete action probabilities. What-If counterfactuals evaluate actual model forward passes.
