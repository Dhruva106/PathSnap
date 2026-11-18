"""
test_integration.py - Integration Test Suite for PathSnap FastAPI Endpoints
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


class TestAPIEndpoints:
    """End-to-end integration tests for PathSnap REST API."""

    def test_root_endpoint(self, client):
        res = client.get("/", headers={"accept": "application/json"})
        assert res.status_code == 200
        data = res.json()
        assert data["application"] == "PathSnap"
        assert data["status"] == "online"
        assert "endpoints" in data

    def test_health_endpoint(self, client):
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "Baseline XGBoost" in data["champion_model"]

    def test_model_info_endpoint(self, client):
        res = client.get("/model-info")
        assert res.status_code == 200
        data = res.json()
        assert data["model_name"] == "Baseline XGBoost"
        assert len(data["selected_features"]) == 14
        assert "F1-Score" in data["validation_metrics"]
        assert data["validation_metrics"]["F1-Score"] > 0.78

    def test_features_endpoint(self, client):
        res = client.get("/features")
        assert res.status_code == 200
        data = res.json()
        assert "selected_features" in data
        assert len(data["selected_features"]) == 14

    def test_models_list_endpoint(self, client):
        res = client.get("/models")
        assert res.status_code == 200
        data = res.json()
        assert "champion" in data
        assert "xgboost" in data

    def test_trajectories_list_endpoint(self, client):
        res = client.get("/trajectories")
        assert res.status_code == 200
        data = res.json()
        assert data["total_count"] >= 80
        assert len(data["trajectories"]) >= 80

    def test_trajectory_detail_endpoint(self, client):
        res = client.get("/trajectories/00000001")
        assert res.status_code == 200
        data = res.json()
        assert data["trajectory_id"] == "00000001"
        assert data["split"] == "val"
        assert data["point_count"] == 1315
        assert "bounding_box" in data

    def test_map_match_with_hmm(self, client):
        payload = {
            "trajectory_id": "00000001",
            "use_hmm": True,
            "search_radius_m": 50.0,
            "max_candidates": 25
        }
        res = client.post("/map-match", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["total_gps_points"] == 1315
        assert len(data["matched_points"]) == 1315
        assert data["reconstructed_route"]["num_arcs"] > 0
        assert data["reconstructed_route"]["is_fully_connected"] is True
        assert "HMM" in data["method"]

    def test_map_match_without_hmm(self, client):
        payload = {
            "trajectory_id": "00000001",
            "use_hmm": False
        }
        res = client.post("/map-match", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert "Top-1" in data["method"]

    def test_map_match_custom_points(self, client):
        detail = client.get("/trajectories/00000001").json()
        sample_pts = detail["preview_points"][:5]

        payload = {
            "trajectory_id": "00000001",
            "points": sample_pts,
            "use_hmm": True
        }
        res = client.post("/map-match", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert len(data["matched_points"]) == 5

    def test_map_match_not_found(self, client):
        payload = {"trajectory_id": "99999999"}
        res = client.post("/map-match", json=payload)
        assert res.status_code == 404

    def test_map_match_invalid_points(self, client):
        # Sending only 1 point must fail validation (requires >= 2 points)
        payload = {
            "points": [{"latitude": 39.9, "longitude": 116.4, "timestamp": 100.0}]
        }
        res = client.post("/map-match", json=payload)
        assert res.status_code == 422

    def test_evaluate_endpoint(self, client):
        # Map match first
        match_res = client.post("/map-match", json={"trajectory_id": "00000001", "use_hmm": True})
        assert match_res.status_code == 200
        match_data = match_res.json()
        matched_arcs = [p["matched_arc_id"] for p in match_data["matched_points"]]
        route_arcs = match_data["reconstructed_route"]["ordered_arc_ids"]

        # Now evaluate
        eval_payload = {
            "trajectory_id": "00000001",
            "matched_arc_ids": matched_arcs,
            "reconstructed_route_arcs": route_arcs,
        }
        eval_res = client.post("/evaluate", json=eval_payload)
        assert eval_res.status_code == 200
        eval_data = eval_res.json()
        assert eval_data["status"] == "success"
        assert eval_data["point_wise_accuracy_pct"] > 95.0
        assert eval_data["candidate_recall_pct"] > 98.0
        assert eval_data["route_f1_score_pct"] > 90.0
        assert eval_data["route_jaccard_overlap_pct"] > 90.0
        assert len(eval_data["ground_truth_polyline_wgs84"]) > 0

    def test_evaluate_not_found(self, client):
        eval_payload = {
            "trajectory_id": "99999999",
            "matched_arc_ids": [1, 2, 3],
        }
        eval_res = client.post("/evaluate", json=eval_payload)
        assert eval_res.status_code == 404
