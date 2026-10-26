"""
Experiment runner for GAMER-TWIN.
Trains and evaluates all four model variants on identical held-out test splits.
Strictly records data provenance and actual measured metrics with zero fabricated numbers.
"""

import os
import argparse
import pandas as pd
import numpy as np

from src.utils import setup_logger, load_config, save_json, PROJECT_CLASSES
from src.data_loader import load_raw_egocs_data, generate_synthetic_gameplay_data
from src.sequence_builder import prepare_train_val_test_splits
from src.training import train_model
from src.evaluation import evaluate_model, save_evaluation_results

logger = setup_logger("RunExperiments")


def run_all_experiments(
    data_dir: str = "data/raw",
    epochs: int = 5,
    output_dir: str = "outputs"
) -> pd.DataFrame:
    """
    Executes training and evaluation across all 4 selectable variants:
    1. LSTM Baseline
    2. LSTM + Transformer Hybrid
    3. Transformer Encoder (Primary)
    4. Transformer + Self-Supervised Learning (SSL)
    """
    config = load_config()
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    reports_dir = os.path.join(output_dir, "reports")
    processed_dir = os.path.join(output_dir, "processed")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # 1. Load Data
    try:
        df, meta = load_raw_egocs_data(data_dir)
        provenance = "REAL_EGOCS_400K"
    except FileNotFoundError:
        logger.warning(
            f"Real EgoCS-400K dataset not found in '{data_dir}'. "
            "Proceeding with SYNTHETIC_TEST_FIXTURE for software pipeline and UI demonstration."
        )
        sample_dir = config.get("data", {}).get("sample_dir", "data/sample_data")
        ticks_path = os.path.join(sample_dir, "ticks.csv")
        if not os.path.exists(ticks_path):
            df = generate_synthetic_gameplay_data(num_rounds=6, ticks_per_round=400, output_dir=sample_dir, seed=42)
        else:
            df = pd.read_csv(ticks_path)
        provenance = "SYNTHETIC_TEST_FIXTURE (Demo / Pipeline Validation)"

    # 2. Leakage-free preparation
    train_ds, val_ds, test_ds, pipe_report = prepare_train_val_test_splits(df, config, processed_dir)
    input_dim = train_ds.sequences.shape[2]

    models_to_train = [
        "lstm_baseline",
        "lstm_transformer",
        "transformer",
        "transformer_ssl"
    ]

    results_list = []

    for model_name in models_to_train:
        logger.info(f"\n==========================================")
        logger.info(f"   Executing Pipeline for: {model_name}")
        logger.info(f"==========================================")

        trained_model, history = train_model(
            model_name=model_name,
            train_dataset=train_ds,
            val_dataset=val_ds,
            input_dim=input_dim,
            config=config,
            checkpoint_dir=checkpoints_dir,
            epochs=epochs
        )

        eval_res = evaluate_model(
            model=trained_model,
            test_dataset=test_ds,
            config=config,
            model_name=model_name,
            data_provenance=provenance,
            top_k=config.get("evaluation", {}).get("top_k", 3)
        )

        save_evaluation_results(eval_res, os.path.join(reports_dir, f"{model_name}_eval.json"))

        m = eval_res["metrics"]
        results_list.append({
            "Model Variant": model_name,
            "Data Provenance": provenance,
            "Accuracy": m["accuracy"],
            "Top-3 Accuracy": m["top_3_accuracy"],
            "F1-Score (Macro)": m["f1_macro"],
            "F1-Score (Weighted)": m["f1_weighted"],
            "Precision (Macro)": m["precision_macro"],
            "Recall (Macro)": m["recall_macro"],
            "Test Samples": eval_res["total_test_samples"],
            "Status": "Evaluated"
        })

    comparison_df = pd.DataFrame(results_list)
    csv_path = os.path.join(reports_dir, "model_comparison.csv")
    json_path = os.path.join(reports_dir, "model_comparison.json")

    comparison_df.to_csv(csv_path, index=False)
    save_json(results_list, json_path)

    logger.info(f"\nModel Comparison Results saved to {csv_path} and {json_path}:\n")
    print(comparison_df.to_string(index=False))

    return comparison_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all GAMER-TWIN experiments")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw telemetry")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs per model")
    args = parser.parse_args()

    run_all_experiments(args.data_dir, epochs=args.epochs)
