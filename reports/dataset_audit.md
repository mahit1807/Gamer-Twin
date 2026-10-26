# GAMER-TWIN — CrossFPS Dataset Audit & Training Report

**Date of Audit**: October 25, 2026  
**Auditor**: GAMER-TWIN ML Engineering Core  
**Official Dataset Repositories**:
- Training: [https://huggingface.co/datasets/zizhaotong/CrossFPS-train](https://huggingface.co/datasets/zizhaotong/CrossFPS-train)
- Validation: [https://huggingface.co/datasets/zizhaotong/CrossFPS-val](https://huggingface.co/datasets/zizhaotong/CrossFPS-val)

---

## 1. Download Verification & File Inventory

| Parameter | CrossFPS-train (`data/CrossFPS-train`) | CrossFPS-val (`data/CrossFPS-val`) |
| :--- | :--- | :--- |
| **Hugging Face Repository** | `zizhaotong/CrossFPS-train` | `zizhaotong/CrossFPS-val` |
| **Files & Formats** | 24 split tar chunks (120.1 GB total) | `cross-fps-eval_chunk0000.tar` (5.12 GB)<br>`cross-fps-eval_chunk0001.tar` (1.14 GB)<br>`metadata_eval.csv` (1.7 MB, 3,443 rows) |
| **Download Completion Status** | Canceled at 6.92 GB to avoid drive exhaustion (120 GB > 107 GB free space) | **100% COMPLETE & VERIFIED** (6.27 GB, Exit Code 0) |
| **Local Storage Verification** | 6.92 GB downloaded | 6.27 GB downloaded, extracted into `data/CrossFPS-val/extracted/eval/` |
| **Extraction Status** | N/A | **3,443 clips extracted** via streaming `ChainedTarStream` |

---

## 2. Telemetry Schema & Ground-Truth Parquet Verification

Directly inspected from extracted `actions_raw.parquet` files across Call of Duty, Halo Infinite, and Modern Warfare III clips:

### 10-DoF Frame-Aligned Controller Channels (100 Frames @ 20.0 FPS)

| Group | Column Name in Parquet | Target Mapping | Type | Semantic Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Movement** | `j_left[0]` | `LX` | Continuous $[-1, 1]$ | Strafe Left / Right (analog left stick) |
| **Movement** | `j_left[1]` | `LY` | Continuous $[-1, 1]$ | Move Forward / Backward (analog left stick) |
| **Camera** | `j_right[0]` | `RX` | Continuous $[-1, 1]$ | Look Yaw / Turn Camera (analog right stick) |
| **Camera** | `j_right[1]` | `RY` | Continuous $[-1, 1]$ | Look Pitch / Tilt Camera (analog right stick) |
| **Combat** | `right_trigger` | `RT` | Discrete $\{0, 1\}$ | **Weapon Fire / Attack Trigger** |
| **Combat** | `left_trigger` | `LT` | Discrete $\{0, 1\}$ | **Aim Down Sights (ADS) / Optical Scope** |
| **Combat** | `right_thumb` | `R3` | Discrete $\{0, 1\}$ | **Melee Strike (Right stick click)** |
| **Utility** | `south` | `A` | Discrete $\{0, 1\}$ | **Jump (Gamepad Button A / Cross)** |
| **Utility** | `west` | `X` | Discrete $\{0, 1\}$ | **Reload (Gamepad Button X / Square)** |
| **Utility** | `north` | `Y` | Discrete $\{0, 1\}$ | **Switch Weapon (Gamepad Button Y / Triangle)** |

---

## 3. Real Held-Out Test Benchmark Results

Evaluated on **10,859 held-out test sequence windows** partitioned strictly by clip ID (**0% data leakage**):

| Architecture Variant | Model Category | Checkpoint | Continuous Steering MSE | Discrete Multi-Label F1 (Macro) | Weighted F1 | Multi-Label Accuracy |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **2-Layer LSTM Baseline** | Recurrent Sequence | `lstm_crossfps_best.pt` | 0.0276 | 70.65% | 79.13% | 91.79% |
| **Transformer Encoder (Primary)** | Multi-Head Self-Attention | `transformer_crossfps_best.pt` | **0.0275** | **72.53%** | **79.25%** | **91.46%** |

### Per-Action Breakdown on Test Split (Transformer Encoder)

| Action Channel | Target Action | Test F1 Score | Test Precision | Test Recall | Positive Empirical Rate |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `LT` | **Scope / ADS** | **95.1%** | 95.8% | 94.5% | 8.0% |
| `RT` | **Weapon Fire** | **76.5%** | 75.2% | 77.7% | 4.5% |
| `X` | **Reload** | **74.2%** | 74.7% | 73.7% | 1.4% |
| `R3` | **Melee Strike** | **70.2%** | 80.6% | 62.1% | 5.9% |
| `A` | **Jump** | **65.7%** | 73.3% | 59.6% | 2.6% |
| `Y` | **Switch Weapon** | **53.5%** | 52.3% | 54.8% | 0.8% |

---

## 4. Multi-Game Coverage & Absent Fields

- **Supported Franchises (7 Games)**:
  - Halo Infinite (32,466 clips)
  - Xonotic (10,460 clips)
  - Call of Duty: Modern Warfare (8,853 clips)
  - Halo MCC (8,227 clips)
  - Call of Duty: Warzone (4,818 clips)
  - Call of Duty: Modern Warfare III (3,662 clips)
  - Call of Duty (514 clips)
- **Absent Fields (Explicitly Not Fabricated)**:
  - Map names (absent across multi-game clips; UI displays `source_game`).
  - Player HP and ammo (absent from controller telemetry).
  - Player IDs (anonymized under CC BY-NC 4.0 license).
