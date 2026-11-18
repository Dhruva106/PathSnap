"""
model_utils.py - PathSnap Model Loading, Inspection, and Verification Utilities
Provides unified access to all trained baseline models, optimized models, and the champion model.
"""

import os
import json
import joblib
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Canonical registry of all trained models in PathSnap
MODEL_REGISTRY = {
    "champion": {
        "display_name": "Champion Model (Baseline XGBoost)",
        "path": os.path.join(MODELS_DIR, "champion", "best_model.joblib"),
        "fallback_path": os.path.join(MODELS_DIR, "best_model.joblib"),
        "metadata_path": os.path.join(MODELS_DIR, "champion", "best_model_metadata.json"),
        "type": "champion",
        "algorithm": "XGBoost",
    },
    "xgboost": {
        "display_name": "Baseline XGBoost",
        "path": os.path.join(MODELS_DIR, "baseline", "xgboost.joblib"),
        "fallback_path": os.path.join(MODELS_DIR, "best_model.joblib"),
        "metadata_path": os.path.join(MODELS_DIR, "baseline", "xgboost_metadata.json"),
        "type": "baseline",
        "algorithm": "XGBoost",
    },
    "catboost": {
        "display_name": "Baseline CatBoost",
        "path": os.path.join(MODELS_DIR, "baseline", "catboost.joblib"),
        "fallback_path": None,
        "metadata_path": os.path.join(MODELS_DIR, "baseline", "catboost_metadata.json"),
        "type": "baseline",
        "algorithm": "CatBoost",
    },
    "random_forest": {
        "display_name": "Baseline Random Forest",
        "path": os.path.join(MODELS_DIR, "baseline", "random_forest.joblib"),
        "fallback_path": None,
        "metadata_path": os.path.join(MODELS_DIR, "baseline", "random_forest_metadata.json"),
        "type": "baseline",
        "algorithm": "Random Forest",
    },
    "logistic_regression": {
        "display_name": "Baseline Logistic Regression",
        "path": os.path.join(MODELS_DIR, "baseline", "logistic_regression.joblib"),
        "fallback_path": None,
        "metadata_path": os.path.join(MODELS_DIR, "baseline", "logistic_regression_metadata.json"),
        "type": "baseline",
        "algorithm": "Logistic Regression",
    },
    "linear_svm": {
        "display_name": "Baseline Linear SVM (Calibrated)",
        "path": os.path.join(MODELS_DIR, "baseline", "linear_svm.joblib"),
        "fallback_path": None,
        "metadata_path": os.path.join(MODELS_DIR, "baseline", "linear_svm_metadata.json"),
        "type": "baseline",
        "algorithm": "Linear SVM",
    },
    "xgboost_optimized": {
        "display_name": "Bayesian-Optimized XGBoost",
        "path": os.path.join(MODELS_DIR, "optimized", "xgboost_optimized.joblib"),
        "fallback_path": None,
        "metadata_path": os.path.join(MODELS_DIR, "optimized", "xgboost_optimized_metadata.json"),
        "type": "optimized",
        "algorithm": "XGBoost",
    },
}


def get_available_models() -> Dict[str, Dict[str, Any]]:
    """Returns a dictionary of all registered models with their current on-disk availability."""
    available = {}
    for key, info in MODEL_REGISTRY.items():
        exists = os.path.exists(info["path"]) or (info["fallback_path"] and os.path.exists(info["fallback_path"]))
        available[key] = {
            "display_name": info["display_name"],
            "type": info["type"],
            "algorithm": info["algorithm"],
            "path": info["path"] if os.path.exists(info["path"]) else info["fallback_path"],
            "exists_on_disk": exists,
        }
    return available


def load_model(name: str = "champion") -> Any:
    """
    Loads any trained model by registry key name.
    Allowed names: 'champion', 'xgboost', 'catboost', 'random_forest', 'logistic_regression', 'linear_svm', 'xgboost_optimized'
    """
    clean_name = name.lower().strip()
    if clean_name.startswith("baseline/"):
        clean_name = clean_name.replace("baseline/", "")
    elif clean_name.startswith("optimized/"):
        clean_name = clean_name.replace("optimized/", "")
    elif clean_name.startswith("champion/"):
        clean_name = "champion"

    if clean_name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model identifier '{name}'. Available models: {list(MODEL_REGISTRY.keys())}"
        )

    info = MODEL_REGISTRY[clean_name]
    primary_path = info["path"]
    fallback_path = info.get("fallback_path")

    if os.path.exists(primary_path):
        target_path = primary_path
    elif fallback_path and os.path.exists(fallback_path):
        target_path = fallback_path
    else:
        raise FileNotFoundError(
            f"Model artifact for '{name}' not found at {primary_path}. "
            f"Please run 'python src/train.py' to generate and serialize all models."
        )

    return joblib.load(target_path)


def load_champion_model() -> Any:
    """Convenience helper to load the authoritative champion model (Baseline XGBoost)."""
    return load_model("champion")


def get_model_metadata(name: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads metadata for a specific model or the master metadata file if name is None.
    """
    if name is None:
        master_path = os.path.join(MODELS_DIR, "metadata", "model_metadata.json")
        if os.path.exists(master_path):
            with open(master_path, "r") as f:
                return json.load(f)
        alt_path = os.path.join(MODELS_DIR, "best_model_metadata.json")
        if os.path.exists(alt_path):
            with open(alt_path, "r") as f:
                return json.load(f)
        raise FileNotFoundError(f"Master metadata not found at {master_path}")

    clean_name = name.lower().strip()
    if clean_name in MODEL_REGISTRY:
        meta_path = MODEL_REGISTRY[clean_name]["metadata_path"]
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                return json.load(f)

    # Check master metadata fallback
    master_path = os.path.join(MODELS_DIR, "metadata", "model_metadata.json")
    if os.path.exists(master_path):
        with open(master_path, "r") as f:
            master = json.load(f)
            if clean_name in master.get("models", {}).get("baseline", {}):
                return master["models"]["baseline"][clean_name]
            if clean_name in master.get("models", {}).get("optimized", {}):
                return master["models"]["optimized"][clean_name]
            if clean_name == "champion":
                return master["models"]["champion"]

    raise FileNotFoundError(f"Metadata for model '{name}' could not be located.")


def get_feature_metadata() -> Dict[str, Any]:
    """Loads documentation of all engineered and selected features."""
    feat_meta_path = os.path.join(MODELS_DIR, "metadata", "feature_metadata.json")
    if os.path.exists(feat_meta_path):
        with open(feat_meta_path, "r") as f:
            return json.load(f)
    report_path = os.path.join(RESULTS_DIR, "feature_selection", "feature_selection_report.json")
    if not os.path.exists(report_path):
        report_path = os.path.join(RESULTS_DIR, "feature_selection_report.json")
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            return json.load(f)
    raise FileNotFoundError("Feature metadata could not be located.")


def compare_models() -> pd.DataFrame:
    """Returns a pandas DataFrame summarizing all models and their validation metrics."""
    metrics_csv = os.path.join(RESULTS_DIR, "metrics", "model_comparison.csv")
    if not os.path.exists(metrics_csv):
        metrics_csv = os.path.join(RESULTS_DIR, "model_comparison.csv")
    if os.path.exists(metrics_csv):
        return pd.read_csv(metrics_csv)
    raise FileNotFoundError(f"Model comparison metrics CSV not found at {metrics_csv}")


def verify_all_models() -> List[Dict[str, Any]]:
    """
    Verifies that all 7 model artifacts (5 baselines, 1 optimized, 1 champion) can be loaded and inspected.
    """
    verification_results = []
    for key, info in MODEL_REGISTRY.items():
        record = {
            "model_key": key,
            "display_name": info["display_name"],
            "type": info["type"],
            "algorithm": info["algorithm"],
            "path": info["path"],
            "exists": False,
            "can_load": False,
            "estimator_class": None,
            "error": None,
        }
        try:
            target_path = info["path"] if os.path.exists(info["path"]) else info.get("fallback_path")
            if target_path and os.path.exists(target_path):
                record["exists"] = True
                record["path"] = target_path
                estimator = joblib.load(target_path)
                record["can_load"] = True
                record["estimator_class"] = estimator.__class__.__name__
            else:
                record["error"] = "File missing"
        except Exception as e:
            record["error"] = str(e)

        verification_results.append(record)

    return verification_results


if __name__ == "__main__":
    print("=" * 80)
    print("PATHSNAP MODEL UTILITIES - VERIFICATION CHECK")
    print("=" * 80)
    results = verify_all_models()
    for res in results:
        status = "PASSED" if res["can_load"] else "FAILED"
        print(f"[{status:6s}] {res['display_name']:35s} | Class: {str(res['estimator_class']):25s} | Path: {res['path']}")
