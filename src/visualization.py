"""
Visualization routines for GAMER-TWIN using Matplotlib and Seaborn.
Generates academic publication-quality charts for training, evaluation, and predictions.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import PROJECT_CLASSES

# Set clean academic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def plot_training_curves(history: Dict[str, Any]) -> plt.Figure:
    """Plots training and validation loss and accuracy curves."""
    epochs = history.get("epochs", [])
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    train_acc = history.get("train_acc", [])
    val_acc = history.get("val_acc", [])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=100)

    # Loss subplot
    axes[0].plot(epochs, train_loss, "o-", color="#1f77b4", label="Train Loss", linewidth=2)
    axes[0].plot(epochs, val_loss, "s--", color="#d62728", label="Validation Loss", linewidth=2)
    axes[0].set_title(f"Loss Curves - {history.get('model_name', 'Model')}", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Cross Entropy Loss", fontsize=11)
    axes[0].legend(loc="upper right", frameon=True)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Accuracy subplot
    axes[1].plot(epochs, [a * 100 for a in train_acc], "o-", color="#2ca02c", label="Train Acc (%)", linewidth=2)
    axes[1].plot(epochs, [a * 100 for a in val_acc], "s--", color="#ff7f0e", label="Validation Acc (%)", linewidth=2)
    axes[1].set_title("Accuracy Curves", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1].legend(loc="lower right", frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, class_labels: Optional[List[str]] = None) -> plt.Figure:
    """Plots a normalized 9x9 confusion matrix heatmap."""
    labels = class_labels or PROJECT_CLASSES
    cm_arr = np.array(cm)

    # Compute row-normalized percentages
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.where(row_sums > 0, (cm_arr / row_sums) * 100, 0.0)

    fig, ax = plt.subplots(figsize=(8.5, 7.2), dpi=100)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Normalized Recall (%)"},
        ax=ax
    )
    ax.set_title("Held-out Test Confusion Matrix (Row Normalized %)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
    ax.set_ylabel("Ground Truth Class", fontsize=11, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    return fig


def plot_class_distribution(class_counts: Dict[str, int], title: str = "Target Class Distribution") -> plt.Figure:
    """Plots bar chart of class counts and balance."""
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=100)
    bars = ax.bar(classes, counts, color="#3b82f6", edgecolor="#1d4ed8", alpha=0.85)

    # Annotate bar counts
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Action Class", fontsize=11)
    ax.set_ylabel("Sample Count", fontsize=11)
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.tight_layout()
    return fig


def plot_prediction_probabilities(ranked_probs: List[Dict[str, Any]]) -> plt.Figure:
    """Plots horizontal bar chart of ranked prediction probabilities."""
    class_names = [item["class_name"] for item in reversed(ranked_probs)]
    probs = [item["probability"] * 100 for item in reversed(ranked_probs)]

    colors = ["#22c55e" if i == len(probs) - 1 else "#94a3b8" for i in range(len(probs))]

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=100)
    bars = ax.barh(class_names, probs, color=colors, edgecolor="#334155")

    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.1f}%",
                    xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0),
                    textcoords="offset points",
                    ha="left", va="center", fontsize=9, fontweight="bold")

    ax.set_xlim(0, max(100.0, max(probs) * 1.15 if probs else 100))
    ax.set_xlabel("Predicted Probability (%)", fontsize=10, fontweight="bold")
    ax.set_title("Ranked Next-Action Probability Distribution", fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_player_trajectory(df: pd.DataFrame, max_points: int = 1000) -> plt.Figure:
    """Plots 2D spatial trajectory (pos_x vs pos_y) with action markers."""
    sub_df = df.iloc[:max_points].copy()
    fig, ax = plt.subplots(figsize=(7, 6), dpi=100)

    # Plot continuous movement path
    ax.plot(sub_df["pos_x"], sub_df["pos_y"], "-", color="#94a3b8", alpha=0.6, label="Path", linewidth=1.2)

    # Highlight combat events (Fire, Jump, Crouch)
    for act, marker, color in [("Fire", "x", "#ef4444"), ("Jump", "^", "#3b82f6"), ("Crouch", "v", "#8b5cf6")]:
        act_rows = sub_df[sub_df["action"] == act]
        if not act_rows.empty:
            ax.scatter(act_rows["pos_x"], act_rows["pos_y"], marker=marker, color=color, label=act, s=40, zorder=5)

    ax.set_title("2D Spatial Telemetry Trajectory (Top-down)", fontsize=11, fontweight="bold")
    ax.set_xlabel("X Coordinate", fontsize=10)
    ax.set_ylabel("Y Coordinate", fontsize=10)
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    plt.tight_layout()
    return fig
