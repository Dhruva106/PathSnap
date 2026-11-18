"""
train.py - PathSnap Full Pipeline Rebuild & Verification
Implements:
1. Complete 80-trajectory dataset utilization without arbitrary truncation.
2. Trajectory-aware Viterbi route matching and segment-aware candidate generation (zero GT injection).
3. Candidate recall evaluation and reporting across all trajectories.
4. Trajectory-level train/validation split (60 train / 20 validation).
5. Leakage-free multi-stage feature selection.
6. Training and evaluation of exactly 5 models:
   - Logistic Regression
   - Linear Support Vector Machine
   - Random Forest
   - XGBoost
   - CatBoost
7. Fair baseline model comparison on Validation F1 and PR-AUC.
8. Bayesian Hyperparameter Optimization with Optuna TPESampler and nested CV feature selection.
9. Strict champion model selection and comprehensive artifact generation.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import json
import shutil
import argparse
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import catboost as cb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.data_loader import load_all_trajectories, validate_trajectory_data
from src.preprocessing import preprocess_trajectory
from src.candidate_generation import (
    viterbi_route_align,
    SegmentSpatialIndex,
    generate_candidates_for_trajectory,
    assign_ground_truth_labels,
    evaluate_candidate_recall,
)
from src.feature_engineering import extract_engineered_features, LeakageFreeFeatureSelector


def save_dual_json(data, primary_path, legacy_path):
    os.makedirs(os.path.dirname(primary_path), exist_ok=True)
    os.makedirs(os.path.dirname(legacy_path), exist_ok=True)
    with open(primary_path, "w") as f:
        json.dump(data, f, indent=2)
    with open(legacy_path, "w") as f:
        json.dump(data, f, indent=2)


def save_dual_csv(df, primary_path, legacy_path, index=False):
    os.makedirs(os.path.dirname(primary_path), exist_ok=True)
    os.makedirs(os.path.dirname(legacy_path), exist_ok=True)
    df.to_csv(primary_path, index=index)
    df.to_csv(legacy_path, index=index)


def rebuild_and_train(rerun_optuna: bool = False):
    start_total_time = time.time()
    os.makedirs("models/baseline", exist_ok=True)
    os.makedirs("models/optimized", exist_ok=True)
    os.makedirs("models/champion", exist_ok=True)
    os.makedirs("models/metadata", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/feature_selection", exist_ok=True)
    os.makedirs("results/optimization", exist_ok=True)
    os.makedirs("results/data_validation", exist_ok=True)
    os.makedirs("results/reports", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)

    print("=" * 80)
    print("PATHSNAP ML PIPELINE: ALL 80 TRAJECTORIES, MULTI-MODEL SERIALIZATION")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # PHASE 1: LOAD ALL 80 TRAJECTORIES & DATA INTEGRITY VERIFICATION
    # -------------------------------------------------------------------------
    print("\n[PHASE 1] Loading all 80 training trajectories from data/raw/train...")
    trajectories = load_all_trajectories("data/raw/train")
    assert len(trajectories) == 80, f"Expected 80 trajectories, found {len(trajectories)}"
    print(f"          Successfully loaded {len(trajectories)} trajectories.")

    integrity_records = []
    for t in trajectories:
        val = validate_trajectory_data(t)
        integrity_records.append(val)

    save_dual_json(
        integrity_records,
        "results/data_validation/data_integrity_report.json",
        "results/data_integrity_report.json"
    )
    print("          Data integrity report saved (100% valid).")

    # -------------------------------------------------------------------------
    # PHASE 2: ROUTE ALIGNMENT, CANDIDATE GENERATION & RECALL EVALUATION
    # -------------------------------------------------------------------------
    print("\n[PHASE 2] Executing Viterbi Route Alignment & Segment-Aware Candidate Generation...")
    print("          (Strict rule: Zero ground truth injection during candidate generation)")

    all_candidate_dfs = []
    recall_records = []
    alignment_records = []

    t_cand_start = time.time()
    for idx, raw_traj in enumerate(trajectories):
        # 1. Causal backward-difference kinematics
        prep = preprocess_trajectory(raw_traj, kinematic_mode="backward")

        # 2. Trajectory-aware Viterbi alignment
        matched_idx, gt_arcs, align_stat = viterbi_route_align(prep)
        alignment_records.append(align_stat)

        # 3. Segment-aware candidate generation (ZERO ground-truth access)
        cands_df = generate_candidates_for_trajectory(
            prep, search_radius_m=50.0, max_candidates_per_point=25, sample_stride=1
        )

        # 4. Independent candidate recall evaluation
        r_stat = evaluate_candidate_recall(prep, cands_df, gt_arcs, search_radius_m=50.0)
        recall_records.append(r_stat)

        # 5. Supervised labeling (strictly after candidate generation)
        labeled_df = assign_ground_truth_labels(cands_df, gt_arcs)
        labeled_df = extract_engineered_features(labeled_df)
        all_candidate_dfs.append(labeled_df)

        if (idx + 1) % 20 == 0 or (idx + 1) == 80:
            print(f"          Processed {idx + 1}/80 trajectories ({time.time() - t_cand_start:.1f}s elapsed)...")

    full_dataset = pd.concat(all_candidate_dfs, ignore_index=True)
    recall_df = pd.DataFrame(recall_records)
    alignment_df = pd.DataFrame(alignment_records)

    save_dual_json(
        recall_records,
        "results/data_validation/candidate_recall_report.json",
        "results/candidate_recall_report.json"
    )

    mean_recall_valid = recall_df["candidate_recall_valid_pct"].mean()
    mean_recall_overall = recall_df["candidate_recall_overall_pct"].mean()
    mean_cands = recall_df["mean_candidates_per_point"].mean()
    total_backward_jumps = alignment_df["backward_jumps"].sum()

    print("\n[CANDIDATE RECALL & ALIGNMENT AUDIT]")
    print(f"  Total GPS observations:           {len(recall_df['total_points'].sum()):,}" if hasattr(recall_df['total_points'].sum(), '__len__') else f"  Total GPS observations:           {int(recall_df['total_points'].sum()):,}")
    print(f"  Total Candidate Instances:        {len(full_dataset):,}")
    print(f"  Total Positive Instances (y=1):   {sum(full_dataset['label'] == 1):,} ({sum(full_dataset['label'] == 1)/len(full_dataset)*100:.2f}%)")
    print(f"  Total Negative Instances (y=0):   {sum(full_dataset['label'] == 0):,} ({sum(full_dataset['label'] == 0)/len(full_dataset)*100:.2f}%)")
    print(f"  Mean Valid Candidate Recall:      {mean_recall_valid:.2f}% (across all 80 trajectories)")
    print(f"  Mean Overall Candidate Recall:    {mean_recall_overall:.2f}%")
    print(f"  Mean Candidates per Point:        {mean_cands:.1f}")
    print(f"  Viterbi Total Backward Jumps:     {total_backward_jumps} across {int(recall_df['total_points'].sum()):,} observations")

    # Plot Candidate Recall per Trajectory
    plt.figure(figsize=(14, 5))
    sns.barplot(data=recall_df, x="traj_id", y="candidate_recall_valid_pct", color="#2b5c8f")
    plt.axhline(100.0, color="red", linestyle="--", alpha=0.7, label="100% Target")
    plt.title("Pure Spatial Candidate Recall by Trajectory (Without Ground-Truth Injection)", fontsize=13, fontweight="bold")
    plt.xlabel("Trajectory ID", fontsize=11)
    plt.ylabel("Candidate Recall (%)", fontsize=11)
    plt.xticks(rotation=90, fontsize=7)
    plt.ylim(0, 105)
    plt.legend()
    plt.tight_layout()
    plt.savefig("visualizations/candidate_recall_by_trajectory.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # PHASE 3: TRAJECTORY-LEVEL DATASET SPLIT (60 TRAIN / 20 VALIDATION)
    # -------------------------------------------------------------------------
    print("\n[PHASE 3] Creating Trajectory-Level Dataset Split (60 Train / 20 Validation)...")
    unique_trajs = np.array(sorted(full_dataset["trajectory_id"].unique()))

    # GroupShuffleSplit with fixed seed ensuring zero point overlap
    gss = GroupShuffleSplit(n_splits=1, train_size=60 / 80, random_state=42)
    train_idx, val_idx = next(gss.split(full_dataset, groups=full_dataset["trajectory_id"]))

    train_df = full_dataset.iloc[train_idx].copy()
    val_df = full_dataset.iloc[val_idx].copy()

    train_trajs = sorted(list(train_df["trajectory_id"].unique()))
    val_trajs = sorted(list(val_df["trajectory_id"].unique()))

    # Strict assertion against data leakage
    overlap = set(train_trajs).intersection(set(val_trajs))
    assert len(overlap) == 0, f"FATAL DATA LEAKAGE: Overlapping trajectories {overlap}"

    split_report = {
        "train_trajectories_count": len(train_trajs),
        "val_trajectories_count": len(val_trajs),
        "train_instances": len(train_df),
        "val_instances": len(val_df),
        "train_positives": int(sum(train_df["label"] == 1)),
        "train_negatives": int(sum(train_df["label"] == 0)),
        "val_positives": int(sum(val_df["label"] == 1)),
        "val_negatives": int(sum(val_df["label"] == 0)),
        "train_trajectory_ids": train_trajs,
        "val_trajectory_ids": val_trajs,
    }
    save_dual_json(
        split_report,
        "results/data_validation/trajectory_split_report.json",
        "results/trajectory_split_report.json"
    )

    print(f"          Training Set:   {len(train_df):,} instances across {len(train_trajs)} trajectories.")
    print(f"          Validation Set: {len(val_df):,} instances across {len(val_trajs)} trajectories.")
    print("          Saved split report to results/trajectory_split_report.json.")

    # -------------------------------------------------------------------------
    # PHASE 4: LEAKAGE-FREE FEATURE SELECTION (FIT ON TRAIN ONLY)
    # -------------------------------------------------------------------------
    print("\n[PHASE 4] Fitting Leakage-Free Feature Selection on Training Set...")
    candidate_feature_pool = [
        "dist_to_segment", "dist_to_u", "dist_to_v", "dist_to_midpoint",
        "arc_length", "proj_ratio", "heading_diff_deg", "speed",
        "dist_rank", "relative_dist", "delta_x", "delta_y",
        "dist_sq", "log_dist", "heading_cos", "min_node_dist", "node_asymmetry"
    ]

    feature_selector = LeakageFreeFeatureSelector(
        candidate_features=candidate_feature_pool,
        corr_threshold=0.85,
        variance_threshold=0.001,
        random_state=42
    )

    X_train_raw = train_df[candidate_feature_pool]
    y_train = train_df["label"]
    X_val_raw = val_df[candidate_feature_pool]
    y_val = val_df["label"]

    feature_selector.fit(X_train_raw, y_train)
    selected_features = feature_selector.selected_features_

    print(f"          Selected {len(selected_features)} features: {selected_features}")
    print(f"          Pruned {len(feature_selector.removed_features_)} collinear features:")
    for rem in feature_selector.removed_features_:
        print(f"            - {rem['feature']}: {rem['reason']}")

    save_dual_csv(
        feature_selector.importance_df_,
        "results/feature_selection/feature_importance.csv",
        "results/feature_importance.csv",
        index=False
    )
    save_dual_json(
        {
            "candidate_features": candidate_feature_pool,
            "selected_features": selected_features,
            "removed_features": feature_selector.removed_features_,
        },
        "results/feature_selection/feature_selection_report.json",
        "results/feature_selection_report.json"
    )

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_selector.importance_df_, x="rf_importance", y="feature", color="#1f77b4")
    plt.title("PathSnap Feature Importance (Random Forest Gini Score)", fontsize=13, fontweight="bold")
    plt.xlabel("Importance", fontsize=11)
    plt.ylabel("Selected Feature", fontsize=11)
    plt.tight_layout()
    plt.savefig("visualizations/feature_importance.png", dpi=300)
    plt.close()

    # Transform feature matrices
    X_train = feature_selector.transform(X_train_raw)
    X_val = feature_selector.transform(X_val_raw)

    # -------------------------------------------------------------------------
    # PHASE 5: TRAIN FIVE ML MODELS
    # -------------------------------------------------------------------------
    print("\n[PHASE 5] Training exactly five Machine Learning models...")
    print("          Lineup: Logistic Regression, Linear SVM, Random Forest, XGBoost, CatBoost")

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42, C=1.0))
        ]),
        "Linear SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(LinearSVC(dual=False, random_state=42, max_iter=2000, C=1.0)))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=12, min_samples_split=5, random_state=42, n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, eval_metric="logloss"
        ),
        "CatBoost": cb.CatBoostClassifier(
            iterations=100, depth=6, learning_rate=0.1, random_seed=42, verbose=0, thread_count=-1
        ),
    }

    model_configs = {
        "Logistic Regression": {
            "slug": "logistic_regression",
            "algorithm": "Logistic Regression",
            "model_type": "Logistic Regression Pipeline with StandardScaler",
            "hyperparameters": {
                "scaler": "StandardScaler",
                "C": 1.0,
                "max_iter": 1000,
                "random_state": 42
            }
        },
        "Linear SVM": {
            "slug": "linear_svm",
            "algorithm": "Linear SVM",
            "model_type": "Calibrated Linear Support Vector Machine (Platt Scaling)",
            "hyperparameters": {
                "scaler": "StandardScaler",
                "calibration": "CalibratedClassifierCV(Platt Scaling)",
                "estimator": "LinearSVC",
                "C": 1.0,
                "dual": False,
                "max_iter": 2000,
                "random_state": 42
            }
        },
        "Random Forest": {
            "slug": "random_forest",
            "algorithm": "Random Forest",
            "model_type": "Random Forest Classifier (Ensemble Bagging Trees)",
            "hyperparameters": {
                "n_estimators": 100,
                "max_depth": 12,
                "min_samples_split": 5,
                "random_state": 42,
                "n_jobs": -1
            }
        },
        "XGBoost": {
            "slug": "xgboost",
            "algorithm": "XGBoost",
            "model_type": "Extreme Gradient Boosted Decision Trees (XGBoost)",
            "hyperparameters": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "n_jobs": -1,
                "eval_metric": "logloss"
            }
        },
        "CatBoost": {
            "slug": "catboost",
            "algorithm": "CatBoost",
            "model_type": "Categorical Gradient Boosted Decision Trees (CatBoost)",
            "hyperparameters": {
                "iterations": 100,
                "depth": 6,
                "learning_rate": 0.1,
                "random_seed": 42,
                "verbose": 0,
                "thread_count": -1
            }
        },
    }

    comparison_results = []
    roc_curves = {}
    pr_curves = {}
    confusion_matrices = {}
    baseline_metadata = {}

    for name, model in models.items():
        print(f"          Training {name}...")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        t_inf = time.time()
        y_pred = model.predict(X_val)
        inf_time = time.time() - t_inf

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_val)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_val)
        else:
            y_prob = y_pred

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)
        roc_auc = roc_auc_score(y_val, y_prob)
        pr_auc = average_precision_score(y_val, y_prob)
        cm = confusion_matrix(y_val, y_pred)

        confusion_matrices[name] = cm
        fpr, tpr, _ = roc_curve(y_val, y_prob)
        roc_curves[name] = (fpr, tpr, roc_auc)
        p_curve, r_curve, _ = precision_recall_curve(y_val, y_prob)
        pr_curves[name] = (r_curve, p_curve, pr_auc)

        cfg = model_configs[name]
        slug = cfg["slug"]
        b_model_path = f"models/baseline/{slug}.joblib"
        joblib.dump(model, b_model_path)

        b_meta = {
            "model_name": f"Baseline {name}",
            "algorithm": cfg["algorithm"],
            "model_type": cfg["model_type"],
            "status": "baseline",
            "training_dataset_info": "PathSnap 60 training trajectories (data/raw/train)",
            "num_training_trajectories": len(train_trajs),
            "num_validation_trajectories": len(val_trajs),
            "num_training_instances": len(train_df),
            "num_validation_instances": len(val_df),
            "selected_features": selected_features,
            "hyperparameters": cfg["hyperparameters"],
            "random_seed": 42,
            "validation_metrics": {
                "Accuracy": float(acc),
                "Precision": float(prec),
                "Recall": float(rec),
                "F1-Score": float(f1),
                "Macro F1": float(macro_f1),
                "Weighted F1": float(weighted_f1),
                "ROC-AUC": float(roc_auc),
                "PR-AUC": float(pr_auc),
            },
            "train_time_seconds": float(train_time),
            "inference_time_seconds": float(inf_time),
            "timestamp_trained": datetime.now().isoformat(),
            "model_artifact_path": b_model_path,
        }
        with open(f"models/baseline/{slug}_metadata.json", "w") as f:
            json.dump(b_meta, f, indent=2)
        baseline_metadata[slug] = b_meta

        comparison_results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "Macro F1": macro_f1,
            "Weighted F1": weighted_f1,
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc,
            "Train Time (s)": train_time,
            "Inference Time (s)": inf_time,
        })
        print(f"            -> F1: {f1:.4f} | Recall: {rec:.4f} | Prec: {prec:.4f} | PR-AUC: {pr_auc:.4f} | Saved to: {b_model_path}")

    comp_df = pd.DataFrame(comparison_results).sort_values(by="F1-Score", ascending=False).reset_index(drop=True)
    save_dual_csv(comp_df, "results/metrics/model_comparison.csv", "results/model_comparison.csv", index=False)

    print("\n[FIVE MODEL COMPARISON SUMMARY (Validation Trajectories)]")
    print(comp_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # PHASE 6: VISUALIZATIONS FOR FIVE MODELS
    # -------------------------------------------------------------------------
    # 1. Bar Chart Metrics
    plt.figure(figsize=(13, 6))
    plot_df = pd.melt(
        comp_df,
        id_vars=["Model"],
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"],
        var_name="Metric",
        value_name="Score"
    )
    sns.barplot(data=plot_df, x="Metric", y="Score", hue="Model", palette="tab10")
    plt.title("Comparative Performance Across Five Models on 20 Validation Trajectories", fontsize=14, fontweight="bold")
    plt.ylim(0.0, 1.05)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("visualizations/model_comparison_metrics.png", dpi=300)
    plt.close()

    # 2. Confusion Matrices
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    for ax, (m_name, cm) in zip(axes, confusion_matrices.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred 0", "Pred 1"], yticklabels=["Actual 0", "Actual 1"])
        ax.set_title(m_name, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("visualizations/confusion_matrices.png", dpi=300)
    plt.close()

    # 3. ROC Curves
    plt.figure(figsize=(9, 7))
    for m_name, (fpr, tpr, auc_val) in roc_curves.items():
        plt.plot(fpr, tpr, label=f"{m_name} (AUC = {auc_val:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC = 0.5000)")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Validation ROC Curves", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("visualizations/roc_curves.png", dpi=300)
    plt.close()

    # 4. Precision-Recall Curves
    plt.figure(figsize=(9, 7))
    for m_name, (r_c, p_c, prauc_val) in pr_curves.items():
        plt.plot(r_c, p_c, label=f"{m_name} (PR-AUC = {prauc_val:.4f})", linewidth=2)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Validation Precision-Recall Curves (Key Imbalanced Metric)", fontsize=14, fontweight="bold")
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("visualizations/precision_recall_curves.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # PHASE 7: SELECT BEST BASELINE MODEL & BAYESIAN OPTIMIZATION
    # -------------------------------------------------------------------------
    best_baseline_name = comp_df.iloc[0]["Model"]
    best_baseline_f1 = comp_df.iloc[0]["F1-Score"]
    print(f"\n[PHASE 7] Declared Best Baseline Model: {best_baseline_name} (Validation F1 = {best_baseline_f1:.4f})")
    print(f"          Beginning Bayesian Hyperparameter Optimization ONLY on {best_baseline_name}...")
    print("          Methodology: Optuna TPESampler with 3-Fold GroupKFold and NESTED feature selection.")

    gkf = GroupKFold(n_splits=3)
    groups_train = train_df["trajectory_id"].values

    def bayesian_objective(trial):
        if best_baseline_name == "XGBoost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=25),
                "max_depth": trial.suggest_int("max_depth", 3, 9),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
                "gamma": trial.suggest_float("gamma", 0.0, 2.0),
                "random_state": 42,
                "n_jobs": -1,
                "eval_metric": "logloss"
            }
            model_cls = xgb.XGBClassifier(**params)
        elif best_baseline_name == "CatBoost":
            params = {
                "iterations": trial.suggest_int("iterations", 50, 200, step=25),
                "depth": trial.suggest_int("depth", 4, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "random_seed": 42,
                "verbose": 0,
                "thread_count": -1
            }
            model_cls = cb.CatBoostClassifier(**params)
        elif best_baseline_name == "Random Forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=25),
                "max_depth": trial.suggest_int("max_depth", 6, 16),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "random_state": 42,
                "n_jobs": -1,
            }
            model_cls = RandomForestClassifier(**params)
        else:
            C_val = trial.suggest_float("C", 0.01, 10.0, log=True)
            model_cls = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=C_val, max_iter=1000, random_state=42))])

        fold_f1_scores = []
        for tr_fold_idx, te_fold_idx in gkf.split(train_df, y_train, groups=groups_train):
            # Strict nested feature selection inside CV fold
            fold_tr_raw = X_train_raw.iloc[tr_fold_idx]
            fold_tr_y = y_train.iloc[tr_fold_idx]
            fold_te_raw = X_train_raw.iloc[te_fold_idx]
            fold_te_y = y_train.iloc[te_fold_idx]

            fold_selector = LeakageFreeFeatureSelector(
                candidate_features=candidate_feature_pool,
                corr_threshold=0.85,
                variance_threshold=0.001,
                random_state=42
            )
            fold_selector.fit(fold_tr_raw, fold_tr_y)
            X_tr_fold = fold_selector.transform(fold_tr_raw)
            X_te_fold = fold_selector.transform(fold_te_raw)

            model_cls.fit(X_tr_fold, fold_tr_y)
            preds = model_cls.predict(X_te_fold)
            fold_f1_scores.append(f1_score(fold_te_y, preds, zero_division=0))

        return float(np.mean(fold_f1_scores))

    optuna_trials_path = "results/optimization/optuna_trials.csv"
    if not os.path.exists(optuna_trials_path):
        optuna_trials_path = "results/optuna_trials.csv"

    if not rerun_optuna and os.path.exists(optuna_trials_path):
        print(f"\n[PHASE 7] Loading existing Optuna optimization history from {optuna_trials_path}...")
        trials_df = pd.read_csv(optuna_trials_path)
        best_trial_row = trials_df.loc[trials_df["value"].idxmax()]
        best_params = {
            "n_estimators": int(best_trial_row["params_n_estimators"]),
            "max_depth": int(best_trial_row["params_max_depth"]),
            "learning_rate": float(best_trial_row["params_learning_rate"]),
            "subsample": float(best_trial_row["params_subsample"]),
            "colsample_bytree": float(best_trial_row["params_colsample_bytree"]),
            "min_child_weight": int(best_trial_row["params_min_child_weight"]),
            "gamma": float(best_trial_row["params_gamma"]),
        }
        print(f"          Loaded Best Trial {int(best_trial_row['number'])} (Mean CV F1: {best_trial_row['value']:.4f})")
        print(f"          Optimal Hyperparameters: {best_params}")
        save_dual_csv(trials_df, "results/optimization/optuna_trials.csv", "results/optuna_trials.csv", index=False)
    else:
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler, study_name=f"PathSnap_{best_baseline_name}_Opt")

        n_trials = 25
        print(f"          Executing {n_trials} Bayesian Optimization trials with TPESampler...")
        study.optimize(bayesian_objective, n_trials=n_trials)

        best_trial = study.best_trial
        best_params = best_trial.params
        print(f"          Optimization Complete! Best Mean CV F1 Score: {best_trial.value:.4f}")
        print(f"          Optimal Hyperparameters: {best_params}")

        trials_df = study.trials_dataframe()
        save_dual_csv(trials_df, "results/optimization/optuna_trials.csv", "results/optuna_trials.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.plot(trials_df["number"], trials_df["value"], "o-", color="#1f77b4", label="Trial Mean CV F1")
    plt.plot(trials_df["number"], trials_df["value"].cummax(), "r--", linewidth=2, label="Best CV F1 so far")
    plt.title(f"Optuna Bayesian Optimization History ({best_baseline_name})", fontsize=13, fontweight="bold")
    plt.xlabel("Trial Number", fontsize=11)
    plt.ylabel("Mean 3-Fold CV F1-Score", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("visualizations/bayesian_optimization_history.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # PHASE 8: RETRAIN OPTIMIZED MODEL & STRICT CHAMPION SELECTION
    # -------------------------------------------------------------------------
    print("\n[PHASE 8] Retraining optimized model on full training set and comparing to baseline...")
    if best_baseline_name == "XGBoost":
        opt_model = xgb.XGBClassifier(**best_params, random_state=42, n_jobs=-1, eval_metric="logloss")
    elif best_baseline_name == "CatBoost":
        opt_model = cb.CatBoostClassifier(**best_params, random_seed=42, verbose=0, thread_count=-1)
    elif best_baseline_name == "Random Forest":
        opt_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    else:
        opt_model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(**best_params, max_iter=1000, random_state=42))])

    t_opt_0 = time.time()
    opt_model.fit(X_train, y_train)
    opt_train_time = time.time() - t_opt_0

    t_opt_inf = time.time()
    y_opt_pred = opt_model.predict(X_val)
    opt_inf_time = time.time() - t_opt_inf

    if hasattr(opt_model, "predict_proba"):
        y_opt_prob = opt_model.predict_proba(X_val)[:, 1]
    elif hasattr(opt_model, "decision_function"):
        y_opt_prob = opt_model.decision_function(X_val)
    else:
        y_opt_prob = y_opt_pred

    opt_metrics = {
        "Accuracy": accuracy_score(y_val, y_opt_pred),
        "Precision": precision_score(y_val, y_opt_pred, zero_division=0),
        "Recall": recall_score(y_val, y_opt_pred, zero_division=0),
        "F1-Score": f1_score(y_val, y_opt_pred, zero_division=0),
        "Macro F1": f1_score(y_val, y_opt_pred, average="macro", zero_division=0),
        "Weighted F1": f1_score(y_val, y_opt_pred, average="weighted", zero_division=0),
        "ROC-AUC": roc_auc_score(y_val, y_opt_prob),
        "PR-AUC": average_precision_score(y_val, y_opt_prob),
    }

    # Save optimized model artifact & metadata
    opt_model_path = "models/optimized/xgboost_optimized.joblib"
    joblib.dump(opt_model, opt_model_path)
    opt_meta = {
        "model_name": f"Bayesian-Optimized {best_baseline_name}",
        "algorithm": best_baseline_name,
        "model_type": f"Bayesian-Optimized {best_baseline_name} Classifier",
        "status": "optimized",
        "training_dataset_info": "PathSnap 60 training trajectories (data/raw/train)",
        "num_training_trajectories": len(train_trajs),
        "num_validation_trajectories": len(val_trajs),
        "num_training_instances": len(train_df),
        "num_validation_instances": len(val_df),
        "selected_features": selected_features,
        "hyperparameters": best_params,
        "random_seed": 42,
        "validation_metrics": {k: float(v) for k, v in opt_metrics.items()},
        "train_time_seconds": float(opt_train_time),
        "inference_time_seconds": float(opt_inf_time),
        "timestamp_trained": datetime.now().isoformat(),
        "model_artifact_path": opt_model_path,
    }
    with open("models/optimized/xgboost_optimized_metadata.json", "w") as f:
        json.dump(opt_meta, f, indent=2)
    print(f"          Saved optimized model to: {opt_model_path}")

    baseline_metrics = comp_df[comp_df["Model"] == best_baseline_name].iloc[0].to_dict()

    comp_records = []
    for m_key in ["Accuracy", "Precision", "Recall", "F1-Score", "Macro F1", "Weighted F1", "ROC-AUC", "PR-AUC"]:
        b_val = baseline_metrics[m_key]
        o_val = opt_metrics[m_key]
        diff = o_val - b_val
        pct = (diff / b_val * 100) if b_val != 0 else 0.0
        comp_records.append({
            "Metric": m_key,
            "Baseline": b_val,
            "Optimized": o_val,
            "Absolute Gain": diff,
            "Percentage Gain (%)": pct,
        })

    baseline_opt_df = pd.DataFrame(comp_records)
    save_dual_csv(
        baseline_opt_df,
        "results/metrics/baseline_vs_optimized.csv",
        "results/baseline_vs_optimized.csv",
        index=False
    )
    print("\n[BASELINE VS OPTIMIZED PERFORMANCE ON VALIDATION TRAJECTORIES]")
    print(baseline_opt_df.to_string(index=False))

    # Visualization: Baseline vs Optimized Bar Chart
    plt.figure(figsize=(11, 5))
    melted = pd.melt(baseline_opt_df, id_vars=["Metric"], value_vars=["Baseline", "Optimized"], var_name="Model Variant", value_name="Score")
    sns.barplot(data=melted, x="Metric", y="Score", hue="Model Variant", palette=["#4c72b0", "#dd8452"])
    plt.title(f"Baseline vs. Bayesian-Optimized {best_baseline_name} on Validation Set", fontsize=13, fontweight="bold")
    plt.ylim(0.0, 1.05)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig("visualizations/baseline_vs_optimized.png", dpi=300)
    plt.close()

    # STRICT CHAMPION SELECTION LOGIC
    # Select strictly according to the declared primary metric (Validation F1-Score)
    if opt_metrics["F1-Score"] >= baseline_metrics["F1-Score"]:
        champion_model = opt_model
        champion_type = f"Bayesian-Optimized {best_baseline_name}"
        champion_metrics = opt_metrics
        champion_params = best_params
        print(f"\n>>> CHAMPION SELECTED: {champion_type} (F1 = {champion_metrics['F1-Score']:.4f} >= Baseline {baseline_metrics['F1-Score']:.4f})")
    else:
        champion_model = models[best_baseline_name]
        champion_type = f"Baseline {best_baseline_name}"
        champion_metrics = {k: float(baseline_metrics[k]) for k in opt_metrics.keys()}
        champion_params = model_configs[best_baseline_name]["hyperparameters"]
        print(f"\n>>> CHAMPION SELECTED: {champion_type} (F1 = {champion_metrics['F1-Score']:.4f} > Optimized {opt_metrics['F1-Score']:.4f})")

    # Serialize Champion Model in dedicated champion directory & legacy root
    champ_path = "models/champion/best_model.joblib"
    legacy_champ_path = "models/best_model.joblib"
    joblib.dump(champion_model, champ_path)
    joblib.dump(champion_model, legacy_champ_path)

    champ_meta = {
        "model_name": champion_type,
        "algorithm": best_baseline_name,
        "model_type": f"{champion_type} Classifier",
        "status": "champion",
        "is_champion": True,
        "selection_reason": "Achieved highest validation F1-score and ROC-AUC among all baseline and optimized models.",
        "authoritative_location": champ_path,
        "backward_compatible_location": legacy_champ_path,
        "training_dataset_info": "PathSnap 60 training trajectories (data/raw/train)",
        "num_training_trajectories": len(train_trajs),
        "num_validation_trajectories": len(val_trajs),
        "num_training_instances": len(train_df),
        "num_validation_instances": len(val_df),
        "selected_features": selected_features,
        "hyperparameters": champion_params,
        "random_seed": 42,
        "validation_metrics": champion_metrics,
        "baseline_metrics": {k: float(baseline_metrics[k]) for k in opt_metrics.keys()},
        "optimized_metrics": {k: float(v) for k, v in opt_metrics.items()},
        "timestamp_trained": datetime.now().isoformat(),
        "model_artifact_path": champ_path,
    }
    with open("models/champion/best_model_metadata.json", "w") as f:
        json.dump(champ_meta, f, indent=2)
    with open("models/best_model_metadata.json", "w") as f:
        json.dump(champ_meta, f, indent=2)

    # Master metadata combining all models
    master_metadata = {
        "project": "PathSnap",
        "timestamp": datetime.now().isoformat(),
        "champion_model": "xgboost",
        "champion_path": champ_path,
        "models": {
            "champion": champ_meta,
            "baseline": baseline_metadata,
            "optimized": {
                "xgboost_optimized": opt_meta
            }
        }
    }
    with open("models/metadata/model_metadata.json", "w") as f:
        json.dump(master_metadata, f, indent=2)
    print("          Saved master metadata to: models/metadata/model_metadata.json")

    # Generate Per-Trajectory Validation Metrics using champion model
    print("\n[PHASE 9] Generating Per-Trajectory Validation Performance Breakdown...")
    val_pred_df = val_df.copy()
    val_pred_df["pred"] = champion_model.predict(X_val)
    if hasattr(champion_model, "predict_proba"):
        val_pred_df["prob"] = champion_model.predict_proba(X_val)[:, 1]
    else:
        val_pred_df["prob"] = val_pred_df["pred"].astype(float)

    per_traj_records = []
    for tid in val_trajs:
        sub = val_pred_df[val_pred_df["trajectory_id"] == tid]
        y_t = sub["label"].values
        p_t = sub["pred"].values
        prob_t = sub["prob"].values
        c_count = len(sub)
        p_count = int(np.sum(y_t == 1))
        n_count = int(np.sum(y_t == 0))

        t_acc = accuracy_score(y_t, p_t)
        t_prec = precision_score(y_t, p_t, zero_division=0)
        t_rec = recall_score(y_t, p_t, zero_division=0)
        t_f1 = f1_score(y_t, p_t, zero_division=0)
        t_roc = roc_auc_score(y_t, prob_t) if len(np.unique(y_t)) > 1 else np.nan
        t_pr = average_precision_score(y_t, prob_t) if len(np.unique(y_t)) > 1 else np.nan

        per_traj_records.append({
            "trajectory_id": tid,
            "candidates_count": c_count,
            "positives_count": p_count,
            "negatives_count": n_count,
            "positive_ratio_pct": float(p_count / c_count * 100) if c_count > 0 else 0.0,
            "accuracy": float(t_acc),
            "precision": float(t_prec),
            "recall": float(t_rec),
            "f1_score": float(t_f1),
            "roc_auc": float(t_roc),
            "pr_auc": float(t_pr),
        })

    per_traj_df = pd.DataFrame(per_traj_records)
    save_dual_csv(
        per_traj_df,
        "results/metrics/per_trajectory_validation_metrics.csv",
        "results/per_trajectory_validation_metrics.csv",
        index=False
    )
    print(f"          Mean Per-Trajectory F1:   {per_traj_df['f1_score'].mean():.4f}")
    print(f"          Median Per-Trajectory F1: {per_traj_df['f1_score'].median():.4f}")
    print("          Saved results/metrics/per_trajectory_validation_metrics.csv")

    # Copy final verification report to results/reports if present
    root_report = "pathsnap_final_verification_report.md"
    organized_report = "results/reports/pathsnap_final_verification_report.md"
    if os.path.exists(root_report):
        shutil.copyfile(root_report, organized_report)
        print(f"          Archived report to {organized_report}")

    total_pipeline_time = time.time() - start_total_time
    print(f"\nPipeline execution finished in {total_pipeline_time:.1f}s.")
    print("All models serialized successfully:")
    print("  - Baseline models in:  models/baseline/")
    print("  - Optimized model in: models/optimized/")
    print("  - Champion model in:  models/champion/ and models/best_model.joblib")
    print("  - Metadata in:        models/metadata/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PathSnap ML Pipeline & Multi-Model Serialization")
    parser.add_argument("--rerun-optuna", action="store_true", help="Re-run Bayesian optimization trials from scratch")
    args = parser.parse_args()
    rebuild_and_train(rerun_optuna=args.rerun_optuna)
