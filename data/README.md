# GAMER-TWIN Dataset Guide: EgoCS-400K & Synthetic Demo Data

## 1. Official Dataset Source: EgoCS-400K
- **Repository**: [https://github.com/EgoCS-400K/Dataset](https://github.com/EgoCS-400K/Dataset)
- **Hugging Face**: [https://huggingface.co/spaces/Cooler-Master/cs2-action-annotation-viewer-preview](https://huggingface.co/spaces/Cooler-Master/cs2-action-annotation-viewer-preview)
- **Description**: EgoCS-400K is a large-scale egocentric Counter-Strike gameplay dataset containing synchronized first-person gameplay video, replay telemetry, action chains, and dynamic programming temporal segments.

## 2. Dataset Structure in EgoCS-400K
The official EgoCS-400K release organizes per-match and per-round data in the following hierarchy:
1. `ticks.csv`: High-frequency tick-level state (coordinates `pos_x, pos_y, pos_z`, velocities `vel_x, vel_y, vel_z`, view angles `view_pitch, view_yaw`, inputs `in_forward, in_back, in_moveleft, in_moveright, in_attack, in_reload, in_jump, in_duck`, ammo, weapon).
2. `events.csv`: Discrete atomic game actions (`fire`, `reload`, `switch_weapon`, `inspect`, `scope`, `grenade`, `jump`, `crouch`, `move`).
3. `action.json` & `protected_action.json`: Action segmentation timelines.
4. `dp_segments.json`: Dynamic programming temporal clip bounds.

## 3. How to Set Up Real EgoCS-400K Data
1. Download match telemetry files from the official repository or Hugging Face.
2. Place the extracted CSV/JSON files into `data/raw/` (e.g., `data/raw/ticks.csv` and `data/raw/events.csv`, or match subfolders like `data/raw/match_01/ticks.csv`).
3. Run the data validation script:
   ```bash
   python -m src.data_validation --input-dir data/raw
   ```
4. Run the data preprocessing and sequence builder:
   ```bash
   python -m src.preprocessing --input-dir data/raw --output-dir outputs/processed
   ```

## 4. Synthetic Demo Data (`data/sample_data/`)
> **IMPORTANT TRANSPARENCY NOTICE (Section 4 & 9 Compliance):**
> Files located in `data/sample_data/` are **SYNTHETIC TEST FIXTURES ONLY**.
> They are generated algorithmically to verify software execution, pipeline integrity, tensor shapes, model forward passes, and UI functionality when the real multi-gigabyte EgoCS-400K dataset is not yet downloaded.
>
> **Under no circumstances are results on synthetic data to be reported as real player behavioral findings or real EgoCS-400K experimental performance.**
