---
license: cc-by-nc-4.0
task_categories:
- image-to-video
tags:
- world-model
- action-conditioned
- fps
- gaming
---

# CrossFPS Dataset

## Overview

CrossFPS is the first multi-game first-person shooter (FPS) dataset with frame-aligned action telemetry.

It comprises **69,000 five-second clips** across **7 FPS titles** at 480×832 resolution and 20 fps, each paired with **10-DoF frame-aligned gamepad controller signals** (4 continuous axes + 6 discrete buttons). The dataset is designed to train interactive world models that generalize across games rather than overfit to single-title patterns.

## Dataset Structure

Each split is organized as a flat directory of clip folders:

```
<split>/
└── <clip_id>/
    ├── video.mp4            # 5-second clip at 480×832, 20fps
    ├── actions_raw.parquet  # Per-frame 10-DoF controller telemetry
    └── metadata.json        # Clip metadata
```

### metadata.json fields

```json
{
  "uuid": "<video_id>_chunk_<chunk_id>_actions",
  "chunk_id": "<chunk_id>",
  "chunk_size": 100,              // number of frames in the clip
  "original_video": {
    "resolution": [1080, 1920],   // original video resolution
    "video_id": "<video_id>",     // source video ID
    "source": "youtube",
    "url": "https://www.youtube.com/watch?v=<video_id>",
    "start_time": 160.0,          // clip start time in seconds
    "end_time": 164.98,           // clip end time in seconds
    "duration": 5.0,
    "start_frame": 9600,          // frame index in original video
    "end_frame": 9899
  },
  "game": "<game_title>",
  "controller_type": "ps4 | keyboard",
  "bbox_controller_overlay": [xtl, ytl, w, h],  // on-screen controller bbox in pixels
  "parent_chunk_id": "<chunk_id>",
  "scene_index": 0,               // scene index within parent chunk
  "total_scenes": 7,
  "window_index": 0,              // 5s window index within scene
  "total_windows": 1,
  "parent_clip": "<video_id>_chunk_<chunk_id>_s<scene_index>",
  "fps": 20.0,
  "original_fps": 60.0,
  "resample_factor": 3,
  "source_game": "<game_title>",
  "split": "train | val | test",
  "prompt": "<two-sentence scene description>",
  "resolution": [480, 832]
}
```

### actions.parquet columns

| Group    | Column | Type       | Description              |
|----------|--------|------------|--------------------------|
| MOVEMENT | LX     | continuous | Move left / right        |
| MOVEMENT | LY     | continuous | Move forward / back      |
| CAMERA   | RX     | continuous | Turn left / right        |
| CAMERA   | RY     | continuous | Look up / down           |
| COMBAT   | RT     | discrete   | Fire                     |
| COMBAT   | LT     | discrete   | Aim down sights (ADS)    |
| COMBAT   | R3     | discrete   | Melee                    |
| UTILITY  | A      | discrete   | Jump                     |
| UTILITY  | X      | discrete   | Reload                   |
| UTILITY  | Y      | discrete   | Switch weapon            |

Continuous signals are normalized to [−1, 1]. Discrete signals are binary (0/1) sampled at each frame.

## Per-Game Statistics

All clips are 5 seconds at 20 fps with 480×832 resolution.

| Game                            | Total  | Train  | Val   | Test  |
|---------------------------------|--------|--------|-------|-------|
| Halo Infinite                   | 32,466 | 30,844 | 973   | 649   |
| Xonotic                         | 10,460 | 9,938  | 313   | 209   |
| Call of Duty: Modern Warfare    | 8,853  | 8,411  | 265   | 177   |
| Halo MCC                        | 8,227  | 7,817  | 246   | 164   |
| Call of Duty: Warzone           | 4,818  | 4,578  | 144   | 96    |
| Call of Duty: Modern Warfare III| 3,662  | 3,480  | 109   | 73    |
| Call of Duty                    | 514    | 489    | 15    | 10    |
| **Total**                       | **69,000** | **65,557** | **2,065** | **1,378** |

Dataset split: **95:3:2** (train/val/test).

## Data Processing Pipeline

Raw gameplay recordings from NitroGen and WorldCam were processed through a nine-stage pipeline to enforce diversity, balance, and cross-game consistency:

1. **Video Clipping** — Raw streams are segmented into fixed-length chunks; game-specific bounding boxes crop out streaming overlays and UI borders.
2. **Scene Cut Detection** — Clips are split at scene transitions (death screens, loading screens) detected via frame-level visual similarity.
3. **Quality Filtering** — Clips shorter than 5 seconds or with left-stick activity below 70% are discarded, yielding ~81,919 clips.
4. **Sliding Window** — Remaining clips are segmented into non-overlapping 5-second windows (~121,845 candidates).
5. **Frame Rate Normalization** — All clips resampled to 20 fps (60 fps sources subsampled 3×; 30 fps sources interpolated).
6. **Action Distribution Balancing** — High-intensity clips (top 15% by peak angular velocity, including rapid 180° turns and fast weapon switches) are oversampled 3× to counteract the long-tail dominance of low-intensity gameplay.
7. **Visual-Action De-biasing** — Clips where action patterns are statistically decoupled from visual context (bottom 20% by scene-action mutual information) are guaranteed inclusion to prevent the model from learning game strategies rather than physical action-visual mappings.
8. **Cross-Game Kinetic Normalization** — Optical flow-based gain calibration aligns action-to-pixel-displacement ratios across titles (post-normalization inter-game gain variance: σ²_gain = 0.034), resolving sensitivity mismatches between game engines. As a fallback, per-title 95th-percentile normalization maps peak camera signals to ±1.
9. **UI Cropping and Resize** — Residual HUD elements are removed via game-specific cropping; all clips resized to 480×832.

Note: Xonotic uses inverted axis conventions; both left-stick axes are negated during preprocessing to unify directional convention across all titles.

## Text Captions

Each clip's first frame is captioned using Gemini in a standardized two-sentence format:
- **Sentence 1**: environment description (setting, lighting, architecture, atmosphere).
- **Sentence 2**: player state and salient visual elements (weapon type, HUD indicators, nearby objects).

These captions are included in `metadata.json` and used as text conditioning during training.

## Dataset Statistics

| Metric                          | Value (Mean ± SD)   |
|---------------------------------|---------------------|
| Linear Velocity (vlin)          | 0.48 ± 0.12         |
| Angular Velocity (ωang)         | 0.26 ± 0.18         |
| Peak Angular Accel. (αpeak)     | 0.78 ± 0.14         |
| Control Smoothness              | 0.82 ± 0.09         |
| Action Entropy (H)              | 2.94 ± 0.31 bits    |
| Gaze Center-bias Index          | 0.42 ± 0.08         |
| Strafe-to-Forward Ratio         | 0.38 : 1.0          |
| Discrete Event Density          | 14.2% ± 3.5%        |
| Optical Flow-Action Corr. (r)   | 0.91 ± 0.03         |

## Intended Use

CrossFPS is intended for training and evaluating:
- **Interactive world models** for FPS and egocentric game environments
- **Action-conditioned video generation** models
- **Cross-game generalization** research in embodied AI

## How to Download

```bash
# Download train split
hf download zizhaotong/CrossFPS-train --repo-type dataset

# Download eval split
hf download zizhaotong/CrossFPS-val --repo-type dataset

# Reassemble and extract train
cat cross-fps-train_chunk*.tar | tar -xf -

# Reassemble and extract eval
cat cross-fps-eval_chunk*.tar | tar -xf -
```

## License and Terms of Use

CC BY-NC 4.0 — This dataset is for research and non-commercial use only.

The dataset contains gameplay videos, frame-aligned action labels, and text captions. It excludes personal identifiers, chat logs, voice recordings, and other private information.

## Citation

```bibtex
@misc{scope2026,
      title={SCOPE: Simulating Cross-game Operations in Playable Environments for FPS World Models},
      author={Zizhao Tong and Hongfeng Lai and Zeqing Wang and Zhaohu Xing and Kexu Cheng and Haoran Xu and Zhao Pu and Shangwen Zhu and Ruili Feng and Jian Zhao and Yan Zhang and Hao Tang and Yeying Jin and Ling Shao},
      year={2026},
      eprint={2605.23345},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.23345}, 
}
```

## Acknowledgements

CrossFPS is built upon gameplay recordings from [NitroGen](https://huggingface.co/datasets/nvidia/NitroGen) and [WorldCam](https://arxiv.org/abs/2603.16871). We thank the respective teams for making their data publicly available.
