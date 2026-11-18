"""
test_evaluator_comparison.py - Automated Test Suite for Official Evaluator Dataset & Teammate Benchmarks
"""

import os
import sys
import json
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.main import app
from src.data_loader import load_single_trajectory, validate_trajectory_data
from src.preprocessing import preprocess_trajectory
from src.candidate_generation import generate_candidates_for_trajectory
from src.inference import PathSnapPredictor
from src.hmm_decoder import PathSnapHMMDecoder
from src.route_reconstruction import reconstruct_continuous_route

client = TestClient(app)

EVALUATOR_TIDS = [
    "00000015", "00000017", "00000023", "00000030", "00000049",
    "00000059", "00000067", "00000073", "00000087", "00000094"
]


class TestEvaluatorDataset:
    """Verifies that the evaluator dataset is complete, uncorrupted, and valid."""

    def test_all_10_trajectories_exist(self):
        eval_dir = "data/raw/evaluator_test"
        assert os.path.exists(eval_dir), f"Directory {eval_dir} missing."
        for tid in EVALUATOR_TIDS:
            traj_folder = os.path.join(eval_dir, tid)
            assert os.path.exists(traj_folder), f"Trajectory folder {tid} missing."

    def test_trajectory_files_and_integrity(self):
        eval_dir = "data/raw/evaluator_test"
        total_points = 0
        for tid in EVALUATOR_TIDS:
            folder = os.path.join(eval_dir, tid)
            data = load_single_trajectory(folder, tid)
            val = validate_trajectory_data(data)
            assert val["valid"], f"Trajectory {tid} failed integrity validation: {val['issues']}"
            assert val["num_track_points"] > 0
            assert val["num_nodes"] > 0
            assert val["num_arcs"] > 0
            assert val["num_route_arcs"] > 0
            total_points += val["num_track_points"]

        assert total_points == 19674, f"Expected 19674 total GPS points, got {total_points}"


class TestEvaluatorInferencePipeline:
    """Verifies inference and reconstruction on evaluator data."""

    def test_candidate_generation_and_scoring(self):
        folder = "data/raw/evaluator_test/00000023"
        raw = load_single_trajectory(folder, "00000023")
        prep = preprocess_trajectory(raw, kinematic_mode="backward")

        cands_df = generate_candidates_for_trajectory(prep, search_radius_m=50.0, max_candidates_per_point=25)
        assert not cands_df.empty
        assert "point_idx" in cands_df.columns
        assert "arc_id" in cands_df.columns

        predictor = PathSnapPredictor(model_name="champion")
        scored_df = predictor.score_candidates(cands_df)
        assert "match_prob" in scored_df.columns
        assert "model_rank" in scored_df.columns

    def test_hmm_viterbi_and_route_reconstruction(self):
        folder = "data/raw/evaluator_test/00000023"
        raw = load_single_trajectory(folder, "00000023")
        prep = preprocess_trajectory(raw, kinematic_mode="backward")

        cands_df = generate_candidates_for_trajectory(prep, search_radius_m=50.0, max_candidates_per_point=25)
        predictor = PathSnapPredictor(model_name="champion")
        scored_df = predictor.score_candidates(cands_df)

        decoder = PathSnapHMMDecoder(beta=3.0, emission_weight=1.2)
        hmm_res = decoder.decode(prep, scored_df)
        assert len(hmm_res["matched_arc_ids"]) == len(prep.track_wgs)

        route_res = reconstruct_continuous_route(prep, hmm_res["matched_arc_ids"])
        assert route_res["num_route_arcs"] > 0
        assert route_res["total_route_length_m"] > 0
        assert len(route_res["route_coordinates_wgs84"]) >= 2


class TestEvaluatorResultsPersistence:
    """Verifies that precomputed evaluator results are valid and comprehensive."""

    def test_individual_results_csv(self):
        csv_path = "results/evaluator/individual_results.csv"
        assert os.path.exists(csv_path), "results/evaluator/individual_results.csv missing."
        import pandas as pd
        df = pd.read_csv(csv_path)
        assert len(df) == 10
        assert set(df["trajectory_id"].astype(str).str.zfill(8)) == set(EVALUATOR_TIDS)
        assert (df["evaluation_status"] == "SUCCESS").all()
        assert df["point_wise_accuracy_pct"].mean() >= 85.0
        assert df["candidate_recall_pct"].mean() >= 99.5

    def test_aggregate_results_json(self):
        json_path = "results/evaluator/aggregate_results.json"
        assert os.path.exists(json_path), "results/evaluator/aggregate_results.json missing."
        with open(json_path, "r") as f:
            data = json.load(f)
        assert data["num_trajectories"] == 10
        assert data["total_gps_observations"] == 19674
        assert data["successful_trajectories"] == 10
        assert data["macro_metrics_across_trajectories"]["mean_point_wise_accuracy_pct"] >= 85.0
        assert data["macro_metrics_across_trajectories"]["mean_route_f1_score_pct"] >= 85.0


class TestComparisonAPIEndpoints:
    """Verifies all evaluator and comparison REST endpoints."""

    def test_get_evaluator_trajectories(self):
        res = client.get("/evaluator/trajectories")
        assert res.status_code == 200
        data = res.json()
        assert len(data) == 10
        assert data[0]["trajectory_id"] in EVALUATOR_TIDS

    def test_get_evaluator_results(self):
        res = client.get("/evaluator/results")
        assert res.status_code == 200
        data = res.json()
        assert len(data) == 10

    def test_get_evaluator_aggregate(self):
        res = client.get("/evaluator/aggregate")
        assert res.status_code == 200
        data = res.json()
        assert data["num_trajectories"] == 10
        assert "macro_metrics_across_trajectories" in data
        assert "pooled_metrics_across_observations" in data

    def test_get_comparison_implementations(self):
        res = client.get("/comparison/implementations")
        assert res.status_code == 200
        data = res.json()
        assert "PathSnap (Champion XGBoost + HMM)" in data["implementations"]

    def test_get_comparison_results(self):
        res = client.get("/comparison/results")
        assert res.status_code == 200
        data = res.json()
        assert data["total_evaluator_trajectories"] == 10
        assert len(data["entries"]) >= 10

    def test_get_comparison_template_csv_and_json(self):
        res_csv = client.get("/comparison/template?format=csv")
        assert res_csv.status_code == 200
        assert "text/csv" in res_csv.headers["content-type"]
        assert "implementation_name,trajectory_id,accuracy" in res_csv.text

        res_json = client.get("/comparison/template?format=json")
        assert res_json.status_code == 200
        data = res_json.json()
        assert "records" in data
        assert len(data["records"]) == 10

    def test_import_metrics_validation_rejection(self):
        # Invalid trajectory ID
        bad_payload = {
            "records": [
                {
                    "implementation_name": "Bad Model",
                    "trajectory_id": "99999999",
                    "accuracy": 85.0
                }
            ]
        }
        res = client.post("/comparison/import", json=bad_payload)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "error"
        assert len(data["validation_errors"]) > 0

    def test_evaluate_teammate_route_prediction_endpoint(self):
        payload = {
            "implementation_name": "Test Teammate Model",
            "trajectory_id": "00000023",
            "predicted_route": [2330, 1590, 2799, 1588],
            "processing_time_seconds": 0.8
        }
        res = client.post("/comparison/evaluate", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert "route_precision" in data["metrics"]
        assert "route_polyline_wgs84" in data["metrics"]
