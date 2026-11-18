"""
evaluate_student_test.py - Independent Evaluation Benchmark on 10 Student Test Trajectories
Evaluates:
- Baseline XGBoost Independent ML Accuracy
- HMM Viterbi Sequence Decoding Accuracy
- Accuracy Gain (%)
- Topological Route Connectivity (%)
- Route Jaccard Agreement (%)
- Intermediate Arcs Filled & Disconnected Jumps
- Computational Latency
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
from src.candidate_generation import viterbi_route_align
from src.inference import PathSnapPredictor
from src.hmm_decoder import PathSnapHMMDecoder
from src.route_reconstruction import reconstruct_continuous_route

STUDENT_TEST_DIR = "data/raw/student_test"
TEST_IDS = [
    "00000000", "00000007", "00000009", "00000040", "00000052",
    "00000056", "00000065", "00000076", "00000077", "00000091"
]


def run_student_test_benchmark():
    predictor = PathSnapPredictor(model_name="champion")
    decoder = PathSnapHMMDecoder(beta=3.0, emission_weight=1.2, transition_weight=1.0)

    results = []
    print(f"=== PathSnap Student Test Evaluation ({len(TEST_IDS)} Trajectories) ===")

    for idx, tid in enumerate(TEST_IDS, 1):
        t_start = time.time()
        folder = os.path.join(STUDENT_TEST_DIR, tid)
        raw = load_single_trajectory(folder, traj_id=tid)
        prep = preprocess_trajectory(raw, kinematic_mode="backward")

        # Ground truth route alignment for evaluation only
        _, gt_arcs, _ = viterbi_route_align(prep)

        # 1. Candidate Generation & XGBoost Prediction
        scored_cands = predictor.predict_trajectory(prep)

        # 2. Independent Top-1 ML Argmax
        top1_df = scored_cands[scored_cands["model_rank"] == 1].sort_values("point_idx")
        top1_arcs = top1_df["arc_id"].values
        acc_indep = float(np.mean(top1_arcs == gt_arcs) * 100.0)

        # 3. HMM Viterbi Sequence Decoding
        t_hmm_start = time.time()
        hmm_res = decoder.decode(prep, scored_cands)
        hmm_time = time.time() - t_hmm_start
        hmm_arcs = hmm_res["matched_arc_ids"]
        acc_hmm = float(np.mean(hmm_arcs == gt_arcs) * 100.0)

        # 4. Continuous Route Reconstruction
        route_res = reconstruct_continuous_route(prep, hmm_arcs)

        # 5. Route Set Jaccard Overlap
        gt_route_set = set(prep.route)
        pred_route_set = set(route_res["ordered_route_arcs"])
        route_jaccard = float(
            len(gt_route_set.intersection(pred_route_set)) /
            max(len(gt_route_set.union(pred_route_set)), 1) * 100.0
        )

        acc_gain = acc_hmm - acc_indep
        total_time = time.time() - t_start
        conn_pct = float(hmm_res["route_summary"]["topological_connectivity_pct"])

        row = {
            "trajectory_id": tid,
            "num_gps_points": len(prep.track_wgs),
            "num_gt_route_arcs": len(gt_route_set),
            "independent_accuracy_pct": acc_indep,
            "hmm_accuracy_pct": acc_hmm,
            "accuracy_gain_pct": acc_gain,
            "route_jaccard_overlap_pct": route_jaccard,
            "topological_connectivity_pct": conn_pct,
            "reconstructed_route_arcs": route_res["num_route_arcs"],
            "intermediate_arcs_filled": route_res["intermediate_arcs_filled"],
            "disconnected_jumps": route_res["disconnected_jumps"],
            "hmm_decoding_time_sec": hmm_time,
            "total_time_sec": total_time,
        }
        results.append(row)

        print(
            f"[{idx:02d}/10] Traj {tid}: Points={len(prep.track_wgs):5d} | "
            f"Indep={acc_indep:5.2f}% | HMM={acc_hmm:5.2f}% | Gain={acc_gain:+5.2f}% | "
            f"Conn={conn_pct:5.1f}% | Time={total_time:5.2f}s"
        )

    res_df = pd.DataFrame(results)

    # Save to metrics directory and root results directory
    os.makedirs("results/metrics", exist_ok=True)
    res_df.to_csv("results/metrics/student_test_evaluation.csv", index=False)
    res_df.to_csv("results/student_test_evaluation.csv", index=False)
    print("\nSaved benchmark metrics to:")
    print("  - results/metrics/student_test_evaluation.csv")
    print("  - results/student_test_evaluation.csv")

    # Summary Statistics
    total_pts = res_df["num_gps_points"].sum()
    mean_indep = res_df["independent_accuracy_pct"].mean()
    mean_hmm = res_df["hmm_accuracy_pct"].mean()
    mean_gain = res_df["accuracy_gain_pct"].mean()
    mean_conn = res_df["topological_connectivity_pct"].mean()
    mean_jaccard = res_df["route_jaccard_overlap_pct"].mean()
    total_jumps = res_df["disconnected_jumps"].sum()

    print("\n" + "=" * 65)
    print("           PATHSNAP STUDENT TEST BENCHMARK SUMMARY")
    print("=" * 65)
    print(f"Total Trajectories Evaluated:   {len(res_df)}")
    print(f"Total GPS Observations:         {total_pts:,}")
    print(f"Mean Independent ML Accuracy:   {mean_indep:.2f}%")
    print(f"Mean HMM Viterbi Accuracy:      {mean_hmm:.2f}%")
    print(f"Mean Accuracy Improvement:      {mean_gain:+.2f}%")
    print(f"Mean Topological Connectivity:  {mean_conn:.2f}%")
    print(f"Mean Route Jaccard Agreement:   {mean_jaccard:.2f}%")
    print(f"Total Disconnected Jumps:       {total_jumps} (across all 10 trajectories)")
    print("=" * 65)

    # Plot Comparison Bar Chart
    os.makedirs("visualizations", exist_ok=True)
    plt.figure(figsize=(12, 6))
    x = np.arange(len(res_df))
    width = 0.35

    plt.bar(x - width/2, res_df["independent_accuracy_pct"], width, label="Independent XGBoost (Top-1 Argmax)", color="#4c72b0")
    plt.bar(x + width/2, res_df["hmm_accuracy_pct"], width, label="HMM Viterbi Sequence Decoding", color="#55a868")

    plt.xlabel("Student Test Trajectory ID", fontsize=11, fontweight="bold")
    plt.ylabel("Point-wise Matching Accuracy (%)", fontsize=11, fontweight="bold")
    plt.title("PathSnap Performance on Student Test Set: Independent ML vs. HMM Viterbi", fontsize=13, fontweight="bold")
    plt.xticks(x, res_df["trajectory_id"], rotation=45, fontsize=10)
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3, axis="y")
    plt.legend(fontsize=10, loc="lower right")
    plt.tight_layout()

    plot_path = "visualizations/student_test_accuracy_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved visualization: {plot_path}")


if __name__ == "__main__":
    run_student_test_benchmark()
