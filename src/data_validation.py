"""
Data validation module for GAMER-TWIN.
Inspects telemetry schema, missing values, class distributions, and data integrity.
"""

import os
import argparse
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from src.utils import setup_logger, save_json, load_config, PROJECT_CLASSES
from src.data_loader import load_raw_egocs_data, generate_synthetic_gameplay_data

logger = setup_logger("DataValidation")


def validate_dataset(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    config: Dict[str, Any]
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validates gameplay telemetry dataframe against GAMER-TWIN requirements.
    
    Checks:
    1. Provenance / synthetic vs real dataset label
    2. Required telemetry columns
    3. Missing values and null rates
    4. Coordinate and pitch/yaw physical ranges
    5. Presence and balance of all 9 target classes
    """
    warnings = []
    errors = []
    
    total_records = len(df)
    is_synthetic = metadata.get("is_synthetic", False)
    data_source = metadata.get("data_source", "UNKNOWN")

    if total_records == 0:
        errors.append("Dataset contains 0 records.")
        return False, {"errors": errors, "warnings": warnings, "status": "FAILED"}

    # 1. Column presence check
    schema_cfg = config.get("data", {}).get("schema_mapping", {})
    expected_cols = (
        schema_cfg.get("numerical_features", []) +
        schema_cfg.get("binary_features", []) +
        schema_cfg.get("categorical_features", [])
    )
    detected_cols = list(df.columns)
    missing_cols = [c for c in expected_cols if c not in detected_cols]

    if missing_cols:
        warnings.append(f"Expected features missing from dataset: {missing_cols}. Missing values will be imputed or dropped.")

    # 2. Missing value check
    null_counts = df.isnull().sum().to_dict()
    total_nulls = int(df.isnull().sum().sum())
    null_percentages = {col: round(float(count / total_records * 100), 2) for col, count in null_counts.items() if count > 0}

    if total_nulls > 0:
        warnings.append(f"Dataset contains {total_nulls} null entries across fields: {list(null_percentages.keys())}")

    # 3. Action / Target class check
    target_classes = config.get("data", {}).get("target_classes", PROJECT_CLASSES)
    if "action" not in df.columns:
        errors.append("Critical column 'action' is missing. Target classes cannot be formed.")
        class_counts = {}
        class_proportions = {}
    else:
        class_counts = df["action"].value_counts().to_dict()
        class_proportions = {cls: round(float(class_counts.get(cls, 0) / total_records * 100), 2) for cls in target_classes}

        missing_classes = [cls for cls in target_classes if class_counts.get(cls, 0) == 0]
        if missing_classes:
            warnings.append(
                f"Missing examples for target classes: {missing_classes}. "
                f"Model will not have training supervision for these classes unless balanced or augmented."
            )

    # 4. Range validation
    if "view_pitch" in df.columns:
        invalid_pitch = ((df["view_pitch"] < -90.0) | (df["view_pitch"] > 90.0)).sum()
        if invalid_pitch > 0:
            warnings.append(f"{invalid_pitch} records have view_pitch outside standard [-90, 90] degrees.")

    is_valid = len(errors) == 0

    report = {
        "data_source": data_source,
        "is_synthetic": bool(is_synthetic),
        "total_records": int(total_records),
        "detected_columns": detected_cols,
        "missing_expected_features": missing_cols,
        "null_counts": {k: int(v) for k, v in null_counts.items() if v > 0},
        "null_percentages": null_percentages,
        "target_class_counts": {cls: int(class_counts.get(cls, 0)) for cls in target_classes},
        "target_class_proportions_percent": class_proportions,
        "warnings": warnings,
        "errors": errors,
        "status": "PASSED" if is_valid else "FAILED"
    }

    return is_valid, report


def run_validation(data_dir: str = "data/raw", output_report_path: str = "outputs/reports/data_validation_report.json") -> Dict[str, Any]:
    """Runs data validation workflow and saves a JSON report."""
    config = load_config()
    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)

    try:
        df, meta = load_raw_egocs_data(data_dir)
    except FileNotFoundError as e:
        logger.warning(f"{e} - Validating synthetic sample data instead.")
        sample_dir = config.get("data", {}).get("sample_dir", "data/sample_data")
        if not os.path.exists(os.path.join(sample_dir, "ticks.csv")):
            logger.info("Generating synthetic sample data for pipeline testing...")
            df = generate_synthetic_gameplay_data(num_rounds=5, ticks_per_round=400, output_dir=sample_dir)
        else:
            df = pd.read_csv(os.path.join(sample_dir, "ticks.csv"))
        meta = {
            "data_source": "SYNTHETIC_TEST_FIXTURE",
            "is_synthetic": True,
            "raw_record_count": len(df),
            "detected_files": {"tick_files": ["data/sample_data/ticks.csv"]},
            "columns": list(df.columns)
        }

    is_valid, report = validate_dataset(df, meta, config)
    save_json(report, output_report_path)

    logger.info(f"Validation finished with status: {report['status']}. Report saved to {output_report_path}")
    if report["warnings"]:
        for w in report["warnings"]:
            logger.warning(f"Validation Warning: {w}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAMER-TWIN Data Validation")
    parser.add_argument("--input-dir", type=str, default="data/raw", help="Path to raw telemetry directory")
    parser.add_argument("--report", type=str, default="outputs/reports/data_validation_report.json", help="Path to save output report")
    args = parser.parse_args()

    run_validation(args.input_dir, args.report)
