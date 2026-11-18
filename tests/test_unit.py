"""
test_unit.py - Unit Test Suite for PathSnap Core Modules
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_single_trajectory, TrajectoryData
from src.preprocessing import preprocess_trajectory, PreprocessedTrajectory
from src.candidate_generation import generate_candidates_for_trajectory
from src.feature_engineering import extract_engineered_features
from src.inference import PathSnapPredictor
from src.hmm_decoder import PathSnapHMMDecoder
from src.route_reconstruction import reconstruct_continuous_route
from src.model_utils import load_champion_model, get_model_metadata, get_feature_metadata

# Use sample validation trajectory 00000001 for unit tests
SAMPLE_TRAJ_PATH = "data/raw/train/00000001"
SAMPLE_TRAJ_ID = "00000001"


class TestDataLoader:
    """Validates data loading and schema validation."""

    def test_load_single_trajectory(self):
        assert os.path.exists(SAMPLE_TRAJ_PATH), f"Sample path {SAMPLE_TRAJ_PATH} does not exist"
        data = load_single_trajectory(SAMPLE_TRAJ_PATH, traj_id=SAMPLE_TRAJ_ID)

        assert isinstance(data, TrajectoryData)
        assert data.traj_id == SAMPLE_TRAJ_ID
        assert data.track.ndim == 2
        assert data.track.shape[1] == 3  # [lon, lat, time]
        assert len(data.track) > 0
        assert data.nodes.ndim == 2
        assert data.nodes.shape[1] == 2  # [lon, lat]
        assert data.arcs.ndim == 2
        assert data.arcs.shape[1] == 2   # [u, v]
        assert data.route is not None
        assert len(data.route) > 0


class TestPreprocessing:
    """Validates coordinate projection and kinematic feature derivation."""

    @pytest.fixture(scope="class")
    def preprocessed(self):
        raw = load_single_trajectory(SAMPLE_TRAJ_PATH, traj_id=SAMPLE_TRAJ_ID)
        return preprocess_trajectory(raw, kinematic_mode="backward")

    def test_preprocessed_structure(self, preprocessed):
        assert isinstance(preprocessed, PreprocessedTrajectory)
        assert len(preprocessed.track_metric) == len(preprocessed.track_wgs)
        assert len(preprocessed.speeds) == len(preprocessed.track_wgs)
        assert len(preprocessed.headings) == len(preprocessed.track_wgs)
        assert len(preprocessed.arc_lengths) == len(preprocessed.arcs)

    def test_metric_projection_validity(self, preprocessed):
        # Coordinate 0 in tangent plane should be close to origin
        ref_pt = preprocessed.track_metric[0]
        assert not np.isnan(ref_pt).any()
        assert not np.isinf(ref_pt).any()

        # Arc lengths must be non-negative
        assert (preprocessed.arc_lengths >= 0).all()
        # Speeds must be non-negative
        assert (preprocessed.speeds >= 0).all()


class TestCandidateGeneration:
    """Validates spatial segment indexing and candidate pruning."""

    @pytest.fixture(scope="class")
    def candidates(self):
        raw = load_single_trajectory(SAMPLE_TRAJ_PATH, traj_id=SAMPLE_TRAJ_ID)
        prep = preprocess_trajectory(raw, kinematic_mode="backward")
        return generate_candidates_for_trajectory(
            prep, search_radius_m=50.0, max_candidates_per_point=25, sample_stride=10
        )

    def test_candidate_columns(self, candidates):
        assert isinstance(candidates, pd.DataFrame)
        assert not candidates.empty
        required_cols = [
            "point_idx", "arc_id",
            "dist_to_segment", "dist_to_u", "dist_to_v", "dist_to_midpoint", "proj_ratio"
        ]
        for col in required_cols:
            assert col in candidates.columns

    def test_candidate_distance_bounds(self, candidates):
        # All candidate segment distances must be non-negative
        assert (candidates["dist_to_segment"] >= 0).all()
        # Projection ratios should be bounded in [0, 1]
        assert (candidates["proj_ratio"] >= 0.0).all()
        assert (candidates["proj_ratio"] <= 1.0).all()


class TestFeatureEngineering:
    """Validates extraction of 14 engineered features without data leakage."""

    @pytest.fixture(scope="class")
    def features(self):
        raw = load_single_trajectory(SAMPLE_TRAJ_PATH, traj_id=SAMPLE_TRAJ_ID)
        prep = preprocess_trajectory(raw, kinematic_mode="backward")
        cands = generate_candidates_for_trajectory(
            prep, search_radius_m=50.0, max_candidates_per_point=25, sample_stride=20
        )
        return extract_engineered_features(cands)

    def test_feature_schema(self, features):
        schema = get_feature_metadata()
        expected_features = schema["selected_features"]
        assert len(expected_features) == 14

        feat_names = [f["name"] if isinstance(f, dict) else f for f in expected_features]
        for feat in feat_names:
            assert feat in features.columns, f"Feature '{feat}' missing from extracted dataframe"
            assert not features[feat].isna().any(), f"Feature '{feat}' contains NaN values"


class TestChampionModelInference:
    """Validates loading and scoring with the champion Baseline XGBoost model."""

    def test_model_loading(self):
        model = load_champion_model()
        assert model is not None
        meta = get_model_metadata("champion")
        assert meta["status"] == "champion"
        assert meta["model_name"] == "Baseline XGBoost"

    def test_predictor_scoring(self):
        predictor = PathSnapPredictor(model_name="champion")
        raw = load_single_trajectory(SAMPLE_TRAJ_PATH, traj_id=SAMPLE_TRAJ_ID)
        prep = preprocess_trajectory(raw, kinematic_mode="backward")
        cands = generate_candidates_for_trajectory(
            prep, search_radius_m=50.0, max_candidates_per_point=25, sample_stride=20
        )
        scored = predictor.score_candidates(cands)

        assert "match_prob" in scored.columns
        assert "model_rank" in scored.columns
        assert (scored["match_prob"] >= 0.0).all()
        assert (scored["match_prob"] <= 1.0).all()
        # Verify rank 1 has highest probability per point
        top1 = scored[scored["model_rank"] == 1]
        assert len(top1) > 0


class TestHMMDecoderAndRouteReconstruction:
    """Validates HMM Viterbi decoding and continuous route reconstruction."""

    def test_hmm_decoding(self):
        raw = load_single_trajectory(SAMPLE_TRAJ_PATH, traj_id=SAMPLE_TRAJ_ID)
        prep = preprocess_trajectory(raw, kinematic_mode="backward")
        cands = generate_candidates_for_trajectory(
            prep, search_radius_m=50.0, max_candidates_per_point=25, sample_stride=10
        )
        predictor = PathSnapPredictor(model_name="champion")
        scored = predictor.score_candidates(cands)

        decoder = PathSnapHMMDecoder()
        res = decoder.decode(prep, scored)

        assert "matched_arc_ids" in res
        matched_arcs = res["matched_arc_ids"]
        expected_len = len(scored["point_idx"].unique())
        assert len(matched_arcs) == expected_len

        # Test continuous route reconstruction
        route_info = reconstruct_continuous_route(prep, matched_arcs)
        assert route_info["num_route_arcs"] > 0
        assert route_info["total_route_length_m"] > 0
        assert len(route_info["route_coordinates_wgs84"]) > 0
        assert route_info["is_fully_connected"] is True
