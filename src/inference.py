"""
inference.py - PathSnap Machine Learning Inference Pipeline
Provides end-to-end candidate road scoring for GPS trajectories using the trained champion model.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List

# Ensure local imports work cleanly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_single_trajectory, TrajectoryData
from src.preprocessing import preprocess_trajectory, PreprocessedTrajectory
from src.candidate_generation import generate_candidates_for_trajectory
from src.feature_engineering import extract_engineered_features
from src.model_utils import load_champion_model, load_model, get_model_metadata


class PathSnapPredictor:
    """
    Inference scoring engine that takes raw GPS trajectories and predicts candidate road segment probabilities.
    """
    def __init__(self, model_name: str = "champion", model_path: Optional[str] = None):
        self.model_name = model_name
        if model_path is not None:
            import joblib
            self.model = joblib.load(model_path)
            meta = get_model_metadata(model_name) if model_name else {}
            self.selected_features = meta.get("selected_features", [
                "dist_to_segment", "dist_to_u", "dist_to_v", "dist_to_midpoint",
                "arc_length", "proj_ratio", "speed", "dist_rank", "relative_dist",
                "delta_x", "delta_y", "heading_cos", "min_node_dist", "node_asymmetry"
            ])
        else:
            self.model = load_model(model_name)
            meta = get_model_metadata(model_name)
            self.selected_features = meta.get("selected_features", [
                "dist_to_segment", "dist_to_u", "dist_to_v", "dist_to_midpoint",
                "arc_length", "proj_ratio", "speed", "dist_rank", "relative_dist",
                "delta_x", "delta_y", "heading_cos", "min_node_dist", "node_asymmetry"
            ])

    def score_candidates(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes a candidate DataFrame with engineered features and returns it annotated with match probabilities.
        """
        if candidates_df.empty:
            df = candidates_df.copy()
            df["match_prob"] = []
            df["predicted_label"] = []
            return df

        df = candidates_df.copy()
        missing_feats = [f for f in self.selected_features if f not in df.columns]
        if missing_feats:
            df = extract_engineered_features(df)

        X = df[self.selected_features]

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[:, 1]
        elif hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X)
            probs = 1.0 / (1.0 + np.exp(-scores))
        else:
            probs = self.model.predict(X).astype(float)

        preds = (probs >= 0.5).astype(int)

        df["match_prob"] = probs
        df["predicted_label"] = preds

        # Rank candidates per GPS point by predicted probability descending
        df["model_rank"] = df.groupby("point_idx")["match_prob"].rank(ascending=False, method="first").astype(int)
        df = df.sort_values(by=["point_idx", "model_rank"]).reset_index(drop=True)
        return df

    def predict_trajectory(
        self,
        trajectory: Union[str, TrajectoryData, PreprocessedTrajectory],
        search_radius_m: float = 50.0,
        max_candidates_per_point: int = 25,
        sample_stride: int = 1
    ) -> pd.DataFrame:
        """
        Executes the full inference pipeline for a trajectory:
        1. Preprocessing & coordinate projection (if needed)
        2. Segment-aware spatial candidate generation (zero GT access)
        3. Feature engineering
        4. ML model probability prediction
        """
        if isinstance(trajectory, str):
            raw_traj = load_single_trajectory(trajectory)
            prep = preprocess_trajectory(raw_traj, kinematic_mode="backward")
        elif isinstance(trajectory, TrajectoryData):
            prep = preprocess_trajectory(trajectory, kinematic_mode="backward")
        elif isinstance(trajectory, PreprocessedTrajectory):
            prep = trajectory
        else:
            raise TypeError(f"Unsupported trajectory type: {type(trajectory)}")

        # Candidate generation
        cands_df = generate_candidates_for_trajectory(
            prep,
            search_radius_m=search_radius_m,
            max_candidates_per_point=max_candidates_per_point,
            sample_stride=sample_stride
        )

        # Feature engineering
        feat_df = extract_engineered_features(cands_df)

        # ML scoring
        scored_df = self.score_candidates(feat_df)
        return scored_df


def main():
    parser = argparse.ArgumentParser(description="PathSnap ML Model Inference Engine")
    parser.add_argument("--traj_id", type=str, default="00000001", help="Trajectory ID in data/raw/train to score")
    parser.add_argument("--model", type=str, default="champion", help="Model to use (champion, xgboost, catboost, etc.)")
    parser.add_argument("--output", type=str, default=None, help="Optional output CSV path for scored candidates")
    args = parser.parse_args()

    traj_path = os.path.join("data/raw/train", args.traj_id)
    if not os.path.exists(traj_path):
        print(f"Error: Trajectory folder {traj_path} not found.")
        sys.exit(1)

    print(f"Loading model '{args.model}'...")
    predictor = PathSnapPredictor(model_name=args.model)

    print(f"Scoring trajectory {args.traj_id}...")
    scored = predictor.predict_trajectory(traj_path)

    print("\n--- Inference Results Preview ---")
    print(f"Total scored candidate segments: {len(scored):,}")
    print(f"Total GPS points: {scored['point_idx'].nunique():,}")
    top1 = scored[scored["model_rank"] == 1]
    print(f"Mean top-1 predicted probability: {top1['match_prob'].mean():.4f}")
    print("\nSample top-ranked candidates:")
    print(top1[["point_idx", "arc_id", "dist_to_segment", "match_prob", "predicted_label"]].head(10).to_string(index=False))

    if args.output:
        scored.to_csv(args.output, index=False)
        print(f"\nSaved scored candidates to {args.output}")


if __name__ == "__main__":
    main()
