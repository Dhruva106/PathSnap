"""
evaluate_evaluator_test.py - Official Evaluator Trajectory Benchmark for PathSnap
Evaluates the 10 official evaluator trajectories using the champion XGBoost model,
Newson-Krumm HMM Viterbi decoder, and topological Dijkstra route reconstruction.
Produces individual and aggregate results, JSON reports, CSV exports, and visualization charts.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_single_trajectory
from src.preprocessing import preprocess_trajectory
from src.candidate_generation import generate_candidates_for_trajectory, viterbi_route_align
from src.inference import PathSnapPredictor
from src.hmm_decoder import PathSnapHMMDecoder
from src.route_reconstruction import reconstruct_continuous_route

EVALUATOR_DIR = "data/raw/evaluator_test"
EVALUATOR_TIDS = [
    "00000015", "00000017", "00000023", "00000030", "00000049",
    "00000059", "00000067", "00000073", "00000087", "00000094"
]


def run_evaluator_benchmark():
    os.makedirs("results/evaluator/trajectory_results", exist_ok=True)
    os.makedirs("results/evaluator/predictions", exist_ok=True)
    os.makedirs("results/evaluator/visualizations", exist_ok=True)
    os.makedirs("visualizations/evaluator", exist_ok=True)

    predictor = PathSnapPredictor(model_name="champion")
    decoder = PathSnapHMMDecoder(beta=3.0, emission_weight=1.2, transition_weight=1.0)

    individual_records = []
    pooled_gt_matches = 0
    pooled_top1_matches = 0
    pooled_hmm_matches = 0
    pooled_cand_recall_hits = 0
    pooled_total_points = 0
    pooled_pred_arcs_set = set()
    pooled_gt_arcs_set = set()

    print(f"=== Starting PathSnap Evaluator Benchmark ({len(EVALUATOR_TIDS)} Trajectories) ===")

    for idx, tid in enumerate(EVALUATOR_TIDS, 1):
        t0 = time.time()
        folder = os.path.join(EVALUATOR_DIR, tid)
        raw = load_single_trajectory(folder, traj_id=tid)
        prep = preprocess_trajectory(raw, kinematic_mode="backward")

        num_pts = len(prep.track_wgs)

        # Ground truth route alignment for evaluation only (zero leakage)
        _, gt_arcs, _ = viterbi_route_align(prep)

        # 1. Candidate Generation
        t_cand_start = time.time()
        cands_df = generate_candidates_for_trajectory(
            prep, search_radius_m=50.0, max_candidates_per_point=25, sample_stride=1
        )
        cand_gen_time = time.time() - t_cand_start
        num_cands = len(cands_df)

        # 2. XGBoost ML Scoring
        t_ml_start = time.time()
        scored_cands = predictor.score_candidates(cands_df)
        ml_time = time.time() - t_ml_start

        # 3. Independent Top-1 ML Argmax (Point-wise baseline)
        top1_df = scored_cands[scored_cands["model_rank"] == 1].sort_values("point_idx")
        top1_arcs = top1_df["arc_id"].values
        top1_len = min(len(top1_arcs), len(gt_arcs))
        top1_matches = int(np.sum(top1_arcs[:top1_len] == gt_arcs[:top1_len]))
        acc_top1 = float(top1_matches / max(top1_len, 1) * 100.0)

        # Candidate recall
        cands_by_pt = cands_df.groupby("point_idx")
        cand_recall_hits = 0
        for i in range(len(gt_arcs)):
            if i in cands_by_pt.groups:
                pt_arcs = set(cands_by_pt.get_group(i)["arc_id"].values)
                if gt_arcs[i] in pt_arcs:
                    cand_recall_hits += 1
        cand_recall_pct = float(cand_recall_hits / max(len(gt_arcs), 1) * 100.0)

        # 4. HMM Viterbi Sequence Decoding
        t_hmm_start = time.time()
        hmm_res = decoder.decode(prep, scored_cands)
        hmm_time = time.time() - t_hmm_start
        hmm_arcs = hmm_res["matched_arc_ids"]
        hmm_len = min(len(hmm_arcs), len(gt_arcs))
        hmm_matches = int(np.sum(hmm_arcs[:hmm_len] == gt_arcs[:hmm_len]))
        acc_hmm = float(hmm_matches / max(hmm_len, 1) * 100.0)
        acc_gain = acc_hmm - acc_top1

        # 5. Continuous Route Reconstruction
        t_recon_start = time.time()
        route_res = reconstruct_continuous_route(prep, hmm_arcs)
        recon_time = time.time() - t_recon_start

        # 6. Route Set Metrics
        pred_route_set = set(route_res["ordered_route_arcs"])
        gt_route_set = set(prep.route)

        inter = len(pred_route_set.intersection(gt_route_set))
        union = len(pred_route_set.union(gt_route_set))
        route_jaccard = float(inter / max(union, 1) * 100.0)
        route_prec = float(inter / max(len(pred_route_set), 1) * 100.0)
        route_rec = float(inter / max(len(gt_route_set), 1) * 100.0)
        route_f1 = float(2 * route_prec * route_rec / max(route_prec + route_rec, 1e-6))

        # Ground truth route length calculation
        gt_route_len_m = 0.0
        for aid in prep.route:
            if aid < len(prep.arcs):
                u_m = prep.nodes_metric[int(prep.arcs[aid, 0])]
                v_m = prep.nodes_metric[int(prep.arcs[aid, 1])]
                gt_route_len_m += float(np.linalg.norm(v_m - u_m))

        pred_route_len_m = float(route_res["total_route_length_m"])
        route_dist_err_m = abs(pred_route_len_m - gt_route_len_m)

        conn_pct = float(hmm_res["route_summary"]["topological_connectivity_pct"])
        total_time = time.time() - t0

        # Update pooled counts
        pooled_total_points += num_pts
        pooled_gt_matches += len(gt_arcs)
        pooled_top1_matches += top1_matches
        pooled_hmm_matches += hmm_matches
        pooled_cand_recall_hits += cand_recall_hits

        record = {
            "trajectory_id": tid,
            "gps_observation_count": num_pts,
            "candidate_count": num_cands,
            "candidate_recall_pct": round(cand_recall_pct, 2),
            "point_wise_accuracy_pct": round(acc_hmm, 2),
            "ml_only_accuracy_pct": round(acc_top1, 2),
            "hmm_accuracy_pct": round(acc_hmm, 2),
            "hmm_improvement_pct": round(acc_gain, 2),
            "route_precision_pct": round(route_prec, 2),
            "route_recall_pct": round(route_rec, 2),
            "route_f1_score_pct": round(route_f1, 2),
            "route_jaccard_overlap_pct": round(route_jaccard, 2),
            "topological_connectivity_pct": round(conn_pct, 2),
            "disconnected_jumps": int(route_res["disconnected_jumps"]),
            "intermediate_arcs_filled": int(route_res["intermediate_arcs_filled"]),
            "pred_route_length_m": round(pred_route_len_m, 2),
            "gt_route_length_m": round(gt_route_len_m, 2),
            "route_distance_error_m": round(route_dist_err_m, 2),
            "candidate_generation_time_sec": round(cand_gen_time, 3),
            "ml_inference_time_sec": round(ml_time, 3),
            "hmm_decoding_time_sec": round(hmm_time, 3),
            "route_reconstruction_time_sec": round(recon_time, 3),
            "total_processing_time_sec": round(total_time, 3),
            "evaluation_status": "SUCCESS"
        }
        individual_records.append(record)

        # Save individual trajectory predictions CSV
        pred_df = pd.DataFrame({
            "point_idx": np.arange(num_pts),
            "longitude": prep.track_wgs[:, 0],
            "latitude": prep.track_wgs[:, 1],
            "timestamp": prep.track_wgs[:, 2],
            "top1_ml_arc": top1_arcs,
            "hmm_viterbi_arc": hmm_arcs,
            "gt_aligned_arc": gt_arcs,
            "is_correct_hmm": (hmm_arcs == gt_arcs).astype(int),
            "is_correct_top1": (top1_arcs == gt_arcs).astype(int)
        })
        pred_df.to_csv(f"results/evaluator/predictions/{tid}_predictions.csv", index=False)

        # Save individual JSON detail
        json_detail = {
            "trajectory_id": tid,
            "metrics": record,
            "reconstructed_route": {
                "num_route_arcs": int(route_res["num_route_arcs"]),
                "ordered_arcs": [int(a) for a in route_res["ordered_route_arcs"]],
                "total_route_length_m": float(route_res["total_route_length_m"]),
                "intermediate_arcs_filled": int(route_res["intermediate_arcs_filled"]),
                "disconnected_jumps": int(route_res["disconnected_jumps"]),
                "is_fully_connected": bool(route_res["is_fully_connected"]),
                "route_polyline_wgs84": [[float(coord[0]), float(coord[1])] for coord in route_res["route_coordinates_wgs84"]]
            },
            "ground_truth_route": {
                "num_gt_arcs": len(gt_route_set),
                "gt_arcs": [int(a) for a in prep.route]
            }
        }
        with open(f"results/evaluator/trajectory_results/{tid}.json", "w") as f:
            json.dump(json_detail, f, indent=2, default=lambda x: int(x) if isinstance(x, (np.integer, np.int64)) else float(x) if isinstance(x, (np.floating, np.float64)) else x.tolist() if isinstance(x, np.ndarray) else str(x))

        print(f"[{idx:02d}/10] Traj {tid}: Pts={num_pts:4d}, Top1 Acc={acc_top1:5.2f}%, HMM Acc={acc_hmm:5.2f}% (+{acc_gain:5.2f}%), Conn={conn_pct:5.2f}%, F1={route_f1:5.2f}%, Jaccard={route_jaccard:5.2f}%, Time={total_time:5.2f}s")

    # Build DataFrame
    df_results = pd.DataFrame(individual_records)
    df_results.to_csv("results/evaluator/individual_results.csv", index=False)

    # Compute Macro Averages (unweighted mean across trajectories)
    macro_acc = float(df_results["point_wise_accuracy_pct"].mean())
    macro_top1 = float(df_results["ml_only_accuracy_pct"].mean())
    macro_gain = float(df_results["hmm_improvement_pct"].mean())
    macro_cand_recall = float(df_results["candidate_recall_pct"].mean())
    macro_prec = float(df_results["route_precision_pct"].mean())
    macro_rec = float(df_results["route_recall_pct"].mean())
    macro_f1 = float(df_results["route_f1_score_pct"].mean())
    macro_jaccard = float(df_results["route_jaccard_overlap_pct"].mean())
    macro_conn = float(df_results["topological_connectivity_pct"].mean())
    macro_total_time = float(df_results["total_processing_time_sec"].sum())
    macro_mean_time = float(df_results["total_processing_time_sec"].mean())

    # Compute Pooled Metrics (aggregated across all individual observations)
    pooled_acc = float(pooled_hmm_matches / max(pooled_gt_matches, 1) * 100.0)
    pooled_top1 = float(pooled_top1_matches / max(pooled_gt_matches, 1) * 100.0)
    pooled_gain = pooled_acc - pooled_top1
    pooled_cand_recall = float(pooled_cand_recall_hits / max(pooled_gt_matches, 1) * 100.0)

    aggregate_summary = {
        "dataset_name": "Official Evaluator Test Trajectories",
        "num_trajectories": len(EVALUATOR_TIDS),
        "total_gps_observations": pooled_total_points,
        "successful_trajectories": len(EVALUATOR_TIDS),
        "failed_trajectories": 0,
        "champion_model": "Baseline XGBoost",
        "macro_metrics_across_trajectories": {
            "mean_point_wise_accuracy_pct": round(macro_acc, 2),
            "mean_top1_ml_accuracy_pct": round(macro_top1, 2),
            "mean_hmm_improvement_pct": round(macro_gain, 2),
            "mean_candidate_recall_pct": round(macro_cand_recall, 2),
            "mean_route_precision_pct": round(macro_prec, 2),
            "mean_route_recall_pct": round(macro_rec, 2),
            "mean_route_f1_score_pct": round(macro_f1, 2),
            "mean_route_jaccard_overlap_pct": round(macro_jaccard, 2),
            "mean_topological_connectivity_pct": round(macro_conn, 2),
            "total_processing_time_sec": round(macro_total_time, 2),
            "average_processing_time_sec": round(macro_mean_time, 3)
        },
        "pooled_metrics_across_observations": {
            "pooled_point_wise_accuracy_pct": round(pooled_acc, 2),
            "pooled_top1_ml_accuracy_pct": round(pooled_top1, 2),
            "pooled_hmm_improvement_pct": round(pooled_gain, 2),
            "pooled_candidate_recall_pct": round(pooled_cand_recall, 2),
            "total_evaluated_points": pooled_gt_matches,
            "total_hmm_matches": pooled_hmm_matches,
            "total_top1_matches": pooled_top1_matches,
            "total_cand_recall_hits": pooled_cand_recall_hits
        }
    }

    with open("results/evaluator/aggregate_results.json", "w") as f:
        json.dump(aggregate_summary, f, indent=2)

    # Save summary CSV
    df_agg = pd.DataFrame([{
        "Metric Category": "Macro (Trajectory Average)",
        "Point Accuracy (%)": round(macro_acc, 2),
        "Top-1 ML Accuracy (%)": round(macro_top1, 2),
        "HMM Improvement (%)": round(macro_gain, 2),
        "Candidate Recall (%)": round(macro_cand_recall, 2),
        "Route F1-Score (%)": round(macro_f1, 2),
        "Route Jaccard (%)": round(macro_jaccard, 2),
        "Topological Connectivity (%)": round(macro_conn, 2),
        "Avg Time / Traj (s)": round(macro_mean_time, 3)
    }, {
        "Metric Category": "Pooled (All 19,674 Observations)",
        "Point Accuracy (%)": round(pooled_acc, 2),
        "Top-1 ML Accuracy (%)": round(pooled_top1, 2),
        "HMM Improvement (%)": round(pooled_gain, 2),
        "Candidate Recall (%)": round(pooled_cand_recall, 2),
        "Route F1-Score (%)": "N/A (Set Metric)",
        "Route Jaccard (%)": "N/A (Set Metric)",
        "Topological Connectivity (%)": "N/A (Sequence Metric)",
        "Avg Time / Traj (s)": round(macro_total_time, 3)
    }])
    df_agg.to_csv("results/evaluator/aggregate_results.csv", index=False)

    # Generate charts
    generate_charts(df_results, aggregate_summary)

    print("\n=== Benchmark Completed Successfully ===")
    print(f"Macro Accuracy: {macro_acc:.2f}% (Top-1: {macro_top1:.2f}%, Gain: +{macro_gain:.2f}%)")
    print(f"Pooled Accuracy: {pooled_acc:.2f}% (Top-1: {pooled_top1:.2f}%, Gain: +{pooled_gain:.2f}%)")
    print(f"Mean Route F1: {macro_f1:.2f}%, Mean Jaccard: {macro_jaccard:.2f}%, Connectivity: {macro_conn:.2f}%")


def generate_charts(df: pd.DataFrame, agg: dict):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # 1. Accuracy comparison: Top-1 ML vs HMM Viterbi
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    x = np.arange(len(df))
    w = 0.35

    ax.bar(x - w/2, df["ml_only_accuracy_pct"], width=w, label="Independent Top-1 ML (XGBoost)", color="#3b82f6", alpha=0.9)
    ax.bar(x + w/2, df["hmm_accuracy_pct"], width=w, label="HMM Viterbi (PathSnap)", color="#10b981", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(df["trajectory_id"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Point-Wise Map Matching Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title("PathSnap Evaluator Benchmark: Independent ML vs. HMM Viterbi", fontsize=13, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1")
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig("visualizations/evaluator/accuracy_comparison.png")
    plt.savefig("results/evaluator/visualizations/accuracy_comparison.png")
    plt.close()

    # 2. Per-trajectory metrics radar/bar (Accuracy, Recall, Jaccard, Connectivity)
    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
    w = 0.2
    ax.bar(x - 1.5*w, df["point_wise_accuracy_pct"], width=w, label="Point Accuracy", color="#10b981")
    ax.bar(x - 0.5*w, df["candidate_recall_pct"], width=w, label="Candidate Recall", color="#06b6d4")
    ax.bar(x + 0.5*w, df["route_jaccard_overlap_pct"], width=w, label="Route Jaccard", color="#8b5cf6")
    ax.bar(x + 1.5*w, df["topological_connectivity_pct"], width=w, label="Connectivity", color="#f59e0b")

    ax.set_xticks(x)
    ax.set_xticklabels(df["trajectory_id"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Metric Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Per-Trajectory Performance on Official Evaluator Dataset", fontsize=13, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", loc="lower right")
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig("visualizations/evaluator/per_trajectory_metrics.png")
    plt.savefig("results/evaluator/visualizations/per_trajectory_metrics.png")
    plt.close()

    # 3. Processing time per trajectory
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    bars = ax.bar(x, df["total_processing_time_sec"], color="#6366f1", width=0.5, edgecolor="#4338ca")
    ax.set_xticks(x)
    ax.set_xticklabels(df["trajectory_id"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Processing Latency (Seconds)", fontsize=11, fontweight="bold")
    ax.set_title("Inference and Route Reconstruction Latency per Trajectory", fontsize=13, fontweight="bold")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.2f}s", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig("visualizations/evaluator/latency_comparison.png")
    plt.savefig("results/evaluator/visualizations/latency_comparison.png")
    plt.close()


if __name__ == "__main__":
    run_evaluator_benchmark()
