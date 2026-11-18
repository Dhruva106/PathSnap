"""
evaluate.py - PathSnap Model Evaluation and Verification Tool
Evaluates any saved model (or all saved models) on the 20 held-out validation trajectories.
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from src.data_loader import load_single_trajectory
from src.preprocessing import preprocess_trajectory
from src.candidate_generation import (
    viterbi_route_align,
    generate_candidates_for_trajectory,
    assign_ground_truth_labels,
)
from src.feature_engineering import extract_engineered_features
from src.model_utils import load_model, get_available_models, get_model_metadata


def evaluate_single_model(
    model_name: str,
    val_trajectories_ids: List[str],
    features: List[str],
    raw_data_dir: str = "data/raw/train"
) -> Dict[str, Any]:
    """Evaluates a single model across the validation trajectories."""
    model = load_model(model_name)

    all_labels = []
    all_preds = []
    all_probs = []
    per_traj_stats = []

    for tid in val_trajectories_ids:
        t_path = os.path.join(raw_data_dir, tid)
        raw_traj = load_single_trajectory(t_path)
        prep = preprocess_trajectory(raw_traj, kinematic_mode="backward")
        _, gt_arcs, _ = viterbi_route_align(prep)
        cands_df = generate_candidates_for_trajectory(
            prep, search_radius_m=50.0, max_candidates_per_point=25, sample_stride=1
        )
        labeled_df = assign_ground_truth_labels(cands_df, gt_arcs)
        labeled_df = extract_engineered_features(labeled_df)

        X = labeled_df[features]
        y = labeled_df["label"].values

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            probs = 1.0 / (1.0 + np.exp(-scores))
        else:
            probs = model.predict(X).astype(float)

        preds = (probs >= 0.5).astype(int)

        t_f1 = f1_score(y, preds, zero_division=0)
        t_prec = precision_score(y, preds, zero_division=0)
        t_rec = recall_score(y, preds, zero_division=0)

        per_traj_stats.append({
            "trajectory_id": tid,
            "candidates_count": len(y),
            "positives_count": int(np.sum(y == 1)),
            "precision": float(t_prec),
            "recall": float(t_rec),
            "f1_score": float(t_f1),
        })

        all_labels.extend(y)
        all_preds.extend(preds)
        all_probs.extend(probs)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)

    per_df = pd.DataFrame(per_traj_stats)

    return {
        "model": model_name,
        "total_instances": len(all_labels),
        "positive_instances": int(np.sum(all_labels == 1)),
        "negative_instances": int(np.sum(all_labels == 0)),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "mean_per_traj_f1": float(per_df["f1_score"].mean()),
        "median_per_traj_f1": float(per_df["f1_score"].median()),
        "confusion_matrix": cm.tolist(),
        "per_trajectory_df": per_df,
    }


def main():
    parser = argparse.ArgumentParser(description="PathSnap Validation Evaluation Tool")
    parser.add_argument("--model", type=str, default="champion", help="Model name ('champion', 'all', 'xgboost', 'catboost', etc.)")
    parser.add_argument("--per_trajectory", action="store_true", help="Print per-trajectory breakdown")
    parser.add_argument("--output", type=str, default=None, help="Save evaluation metrics CSV")
    args = parser.parse_args()

    split_path = "results/data_validation/trajectory_split_report.json"
    if not os.path.exists(split_path):
        split_path = "results/trajectory_split_report.json"

    with open(split_path, "r") as f:
        split_data = json.load(f)

    val_trajs = split_data["val_trajectory_ids"]
    features = [
        "dist_to_segment", "dist_to_u", "dist_to_v", "dist_to_midpoint",
        "arc_length", "proj_ratio", "speed", "dist_rank", "relative_dist",
        "delta_x", "delta_y", "heading_cos", "min_node_dist", "node_asymmetry"
    ]

    print("=" * 80)
    print("PATHSNAP MODEL EVALUATION ON HELD-OUT VALIDATION SET")
    print(f"Validation Trajectories: {len(val_trajs)} trajectories")
    print("=" * 80)

    if args.model.lower() == "all":
        models_to_eval = ["champion", "xgboost", "catboost", "random_forest", "logistic_regression", "linear_svm", "xgboost_optimized"]
    else:
        models_to_eval = [args.model]

    results = []
    for m in models_to_eval:
        print(f"\nEvaluating '{m}'...")
        res = evaluate_single_model(m, val_trajs, features)
        results.append({
            "Model": m,
            "Accuracy": res["accuracy"],
            "Precision": res["precision"],
            "Recall": res["recall"],
            "F1-Score": res["f1_score"],
            "Macro F1": res["macro_f1"],
            "Weighted F1": res["weighted_f1"],
            "ROC-AUC": res["roc_auc"],
            "PR-AUC": res["pr_auc"],
            "Mean Traj F1": res["mean_per_traj_f1"],
            "Median Traj F1": res["median_per_traj_f1"],
        })
        print(f"  F1: {res['f1_score']:.4f} | Prec: {res['precision']:.4f} | Rec: {res['recall']:.4f} | PR-AUC: {res['pr_auc']:.4f} | ROC-AUC: {res['roc_auc']:.4f}")

        if args.per_trajectory:
            print("\n  Per-Trajectory Breakdown:")
            print(res["per_trajectory_df"].to_string(index=False))

    summary_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False).reset_index(drop=True)
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY:")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    if args.output:
        summary_df.to_csv(args.output, index=False)
        print(f"\nSaved metrics to {args.output}")


if __name__ == "__main__":
    main()
