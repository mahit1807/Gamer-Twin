# GAMER-TWIN: A Personalized Deep Learning Framework for Modeling and Predicting Player Behaviour
### A Transformer-Based Digital Twin for Next-Action Prediction in Interactive Games

**Institution**: Symbiosis Institute of Technology, Pune  
**Department**: Department of Artificial Intelligence and Machine Learning  
**Students**: Mahit Turakhia, Het Shah  
**Faculty Advisor**: Dr. Ananya Bhattacharjee  

---

## 1. Project Purpose & Scientific Scope

GAMER-TWIN models an individual player's gameplay patterns from high-frequency telemetry and discrete inputs to forecast their immediate next action in competitive first-person gameplay (Counter-Strike).

The primary architecture is a **Transformer Encoder** utilizing multi-head self-attention, sinusoidal positional encoding, and temporal pooling over past telemetry windows ($T=16$ frames, $D=23$ features) to predict categorical probability distributions over **9 project target classes**:
`Fire`, `Reload`, `Switch Weapon`, `Inspect`, `Scope`, `Grenade`, `Jump`, `Crouch`, `Move`.

> **Academic Scope Boundary**: GAMER-TWIN is an analytical, predictive digital twin that models and forecasts gameplay actions. It does **not** execute live actions inside the game engine or control game bots via Reinforcement Learning. Autonomous reward-maximizing game-playing agents (e.g., Deep Q-Networks / DQN) are identified strictly as future work.

---

## 2. Full-Stack System Architecture

```text
GAMER-TWIN/
├── backend/                        # FastAPI High-Performance Python Backend
│   ├── app/
│   │   ├── main.py                 # FastAPI application & CORS configuration
│   │   ├── api/                    # REST API routers
│   │   │   ├── routes_system.py    # Health, PyTorch device, checkpoint registry
│   │   │   ├── routes_dataset.py   # Telemetry inspection, validation, record previews
│   │   │   ├── routes_models.py    # Model configurations & background training
│   │   │   ├── routes_prediction.py# Sequence inference & Prediction Trace
│   │   │   ├── routes_replay.py    # Chronological timeline replay engine
│   │   │   ├── routes_experiments.py# Evaluation benchmarks & confusion matrices
│   │   │   ├── routes_signature.py # Novel Player Behaviour Signature
│   │   │   └── routes_methodology.py# Academic methodology documentation
│   │   └── schemas/                # Pydantic validation schemas
│   └── tests/                      # Automated API unit tests (pytest)
├── frontend/                       # React 19 + TypeScript + Vite Application
│   ├── src/
│   │   ├── layouts/AppLayout.tsx   # Warm research lab navigation & status header
│   │   ├── pages/                  # 9 Connected Research Pages
│   │   │   ├── Overview.tsx        # Dashboard, project identity, research workflow
│   │   │   ├── DatasetExplorer.tsx # Schema audit, class distribution, row preview
│   │   │   ├── ModelLab.tsx        # Training controls, live epochs, loss curves
│   │   │   ├── PredictionStudio.tsx# Sequence selector, ranked probabilities, trace
│   │   │   ├── ReplayStudio.tsx    # Chronological timeline scrubber & inputs
│   │   │   ├── ModelComparison.tsx # Quantitative benchmark tables & charts
│   │   │   ├── PlayerSignaturePage.tsx # Novel behavioral fingerprint & transitions
│   │   │   ├── Methodology.tsx     # Formal mathematical formulation & diagrams
│   │   │   └── SettingsStatus.tsx  # Diagnostics, environment info & setup
│   │   ├── services/api.ts         # Strongly-typed HTTP API client
│   │   └── index.css               # Warm research color palette & Tailwind tokens
│   ├── package.json
│   └── vite.config.ts
├── Dataset/                        # Cloned official EgoCS-400K GitHub repository
├── data/
│   ├── raw/                        # Target directory for full EgoCS-400K downloads
│   └── sample_data/                # Synthetic test fixtures for pipeline validation
├── outputs/
│   ├── checkpoints/                # PyTorch saved weights (*.pt) & training histories
│   ├── processed/                  # Normalized splits (*.npz) & pipeline report
│   └── reports/                    # Benchmark JSON summaries & confusion matrices
├── src/                            # Deep Learning & Preprocessing Core Modules
│   ├── models.py                   # 4 Model Architectures (LSTM, Hybrid, Transformer, SSL)
│   ├── training.py                 # Supervised & Self-Supervised training loops
│   ├── evaluation.py               # Top-K accuracy, macro/weighted F1, confusion matrix
│   ├── prediction.py               # NextActionPredictor inference engine
│   ├── preprocessing.py            # Zero-leakage standard scalers & class mappers
│   ├── sequence_builder.py         # Sliding window generator & grouped splits
│   └── data_validation.py          # Telemetry range & completeness checker
└── tests/                          # 16 Core PyTorch pipeline tests
```

---

## 3. Evaluated Model Architectures

1. **Model 1: LSTM Baseline** — 2-layer Recurrent Neural Network classifying final hidden state $h_T$.
2. **Model 2: LSTM + Transformer Hybrid** — Recurrent layers for local micro-momentum followed by Transformer multi-head self-attention for global tactical context.
3. **Model 3: Transformer Encoder (Primary Architecture)** — Pure self-attention with sinusoidal positional encoding and temporal pooling.
4. **Model 4: Transformer + Self-Supervised Learning (SSL)** — Two-stage learning: Stage 1 Masked Autoencoding pretraining on historical telemetry frames; Stage 2 supervised fine-tuning on next-action prediction.

---

## 4. Official Dataset: EgoCS-400K

- **Official Repository**: [https://github.com/EgoCS-400K/Dataset](https://github.com/EgoCS-400K/Dataset)
- **Local Path**: `Dataset/` (cloned into workspace)
- **Contents of Repo**: Demo parsing pipeline (`pipeline/parse_dem`), video overlay renderer (`pipeline/overlay`), timeline & segmentation modules (`pipeline/action_timeline`), and data construction documentation (`docs/data-construction.md`).
- **Data Provenance Handling**: The multi-gigabyte raw video and `.dem` files reside on Hugging Face. The project includes verified synthetic test fixtures in `data/sample_data/` marked clearly as `SYNTHETIC_TEST_FIXTURE` to enable full verification without fabricating data. When real EgoCS-400K CSVs are placed into `data/raw/`, the application automatically switches to `REAL_EGOCS_400K`.

---

## 5. Quickstart & How to Run

### Prerequisites
- Python 3.10+ (Tested on Python 3.12)
- Node.js 18+ and npm (Tested on Node v24.18.0, npm 11.16.0)

### Step 1: Install Backend Dependencies
```bash
pip install -r requirements.txt
pip install fastapi uvicorn pydantic pyarrow httpx
```

### Step 2: Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

### Step 3: Run the FastAPI Backend Server
```bash
# Runs on http://127.0.0.1:8000
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```
Interactive API docs are available at `http://127.0.0.1:8000/docs`.

### Step 4: Run the React Vite Frontend
In a separate terminal:
```bash
# Runs on http://127.0.0.1:3000
cd frontend
npm run dev -- --host 127.0.0.1 --port 3000
```
Open **[http://127.0.0.1:3000](http://127.0.0.1:3000)** in your browser.

---

## 6. How to Use the Research Modules

### A. Prediction Studio (`/predict`)
1. Select one of the 4 model variants from the dropdown.
2. Choose an input sequence window ($T=16$ frames).
3. Click **Forecast Next Action**.
4. Inspect the predicted action, confidence score, ranked softmax distribution bar chart, and the **Prediction Trace** for academic audit.

### B. Replay Studio (`/replay`)
1. Use the chronological timeline scrubber or click **Play** to step through 2,384 ticks.
2. Inspect real-time 3D coordinates ($X, Y, Z$), velocity, camera pitch/yaw, health, ammo, and active keyboard inputs ($W, A, S, D$, Fire, Scope).
3. Compare the Digital Twin's predicted next action with the recorded ground truth action.

### C. Model Lab (`/models`)
1. Select a model architecture (e.g., `transformer` or `transformer_ssl`).
2. Adjust epochs, batch size, learning rate, or sequence length.
3. Click **Start Training Job** to initiate real PyTorch gradient descent in the background.
4. Watch real-time loss and accuracy curves converge.

### D. Model Comparison (`/experiments`)
1. View quantitative benchmarks (Top-1 Accuracy, Top-3 Accuracy, Macro/Weighted F1, Precision, Recall).
2. Inspect per-class precision/recall and the full $9 \times 9$ Confusion Matrix.

### E. Player Behaviour Signature (`/signature`)
1. View empirical behavioral statistics calculated from actual recorded action frequencies.
2. Inspect the macro tactical allocation (Movement vs Combat ratio).
3. Analyze empirical 1st-order Markov action transition probabilities ($A_t \rightarrow A_{t+1}$) and frequent micro-tactic n-grams.

---

## 7. Automated Test Suite

Run all 25 automated tests (16 Core Deep Learning tests + 9 Backend API tests):
```bash
# Run Core ML pipeline tests
pytest tests/ -v

# Run FastAPI backend tests
pytest backend/tests/test_api.py -v
```

---

## 8. Academic Authors & Credits

- **Students**: Mahit Turakhia & Het Shah
- **Faculty Advisor**: Dr. Ananya Bhattacharjee
- **Department**: Artificial Intelligence and Machine Learning
- **Institution**: Symbiosis Institute of Technology, Pune
