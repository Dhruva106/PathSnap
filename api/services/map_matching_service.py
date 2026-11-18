"""
map_matching_service.py - Core Service Encapsulating PathSnap ML and HMM Inference
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data_loader import load_single_trajectory, TrajectoryData
from src.preprocessing import preprocess_trajectory, PreprocessedTrajectory
from src.candidate_generation import generate_candidates_for_trajectory, viterbi_route_align
from src.inference import PathSnapPredictor
from src.hmm_decoder import PathSnapHMMDecoder
from src.route_reconstruction import reconstruct_continuous_route
from src.model_utils import load_champion_model, get_model_metadata

from api.schemas.map_match import (
    MapMatchRequest,
    MapMatchResponse,
    MatchedPointResponse,
    ReconstructedRouteResponse,
    EvaluateRequest,
    EvaluateResponse,
    GPSPoint,
)


class MapMatchingService:
    """
    Singleton service class initialized on FastAPI startup.
    Handles data validation, candidate generation, XGBoost ML scoring, HMM Viterbi decoding,
    and route reconstruction.
    """

    def __init__(self):
        self.predictor = PathSnapPredictor(model_name="champion")
        self.decoder = PathSnapHMMDecoder()
        self.metadata = get_model_metadata("champion")
        self.champion_model = self.predictor.model
        print(f"[MapMatchingService] Initialized with champion model: {self.metadata.get('model_name', 'Baseline XGBoost')}")

    def resolve_trajectory_bundle(
        self,
        trajectory_id: Optional[str],
        points: Optional[List[GPSPoint]]
    ) -> Tuple[PreprocessedTrajectory, str]:
        """
        Locates the trajectory data and road network bundle.
        Checks data/raw/train and data/raw/student_test.
        """
        tid = trajectory_id
        if not tid:
            tid = "00000001"  # Default sample trajectory

        folder_train = os.path.join("data/raw/train", tid)
        folder_test = os.path.join("data/raw/student_test", tid)
        folder_eval = os.path.join("data/raw/evaluator_test", tid)

        if os.path.exists(folder_train):
            raw = load_single_trajectory(folder_train, traj_id=tid)
        elif os.path.exists(folder_test):
            raw = load_single_trajectory(folder_test, traj_id=tid)
        elif os.path.exists(folder_eval):
            raw = load_single_trajectory(folder_eval, traj_id=tid)
        else:
            raise FileNotFoundError(f"Trajectory '{tid}' not found in data/raw/train, student_test, or evaluator_test.")

        # If custom points are supplied, replace raw.track with custom points
        if points is not None and len(points) > 0:
            custom_track = np.array(
                [[p.longitude, p.latitude, p.timestamp] for p in points],
                dtype=np.float64
            )
            raw = TrajectoryData(
                traj_id=tid,
                track=custom_track,
                nodes=raw.nodes,
                arcs=raw.arcs,
                route=raw.route
            )

        prep = preprocess_trajectory(raw, kinematic_mode="backward")
        return prep, tid

    def map_match(self, request: MapMatchRequest) -> MapMatchResponse:
        """
        Executes end-to-end map matching:
        1. Resolve trajectory & road network
        2. Preprocessing & Local Tangent coordinate projection
        3. Segment-aware candidate generation (r = search_radius_m, max = max_candidates)
        4. XGBoost candidate probability prediction
        5. Sequence decoding: HMM Viterbi (if use_hmm=True) or Top-1 Argmax (if use_hmm=False)
        6. Continuous route reconstruction along road graph G
        7. Response packaging
        """
        t_start = time.time()

        # 1. Resolve & Preprocess
        prep, resolved_tid = self.resolve_trajectory_bundle(
            request.trajectory_id, request.points
        )

        # 2. Candidate Generation
        cands_df = generate_candidates_for_trajectory(
            prep,
            search_radius_m=request.search_radius_m,
            max_candidates_per_point=request.max_candidates,
            sample_stride=1
        )

        # 3. XGBoost Scoring
        scored_df = self.predictor.score_candidates(cands_df)

        num_pts = len(prep.track_metric)
        R = 6371000.0
        cos_ref = np.cos(np.radians(prep.ref_lat))

        # 4. Decoding Selection
        if request.use_hmm:
            method_str = "XGBoost ML Classifier + HMM Viterbi Sequence Decoding"
            hmm_res = self.decoder.decode(prep, scored_df)
            matched_arcs = hmm_res["matched_arc_ids"]
        else:
            method_str = "Independent XGBoost ML Classifier (Top-1 Argmax)"
            top1_df = scored_df[scored_df["model_rank"] == 1].sort_values("point_idx")
            matched_arcs = top1_df["arc_id"].values

        # 5. Continuous Route Reconstruction
        route_info = reconstruct_continuous_route(prep, matched_arcs)

        # 6. Build MatchedPointResponse objects
        matched_points_list: List[MatchedPointResponse] = []
        cands_by_pt = scored_df.groupby("point_idx")

        for i in range(num_pts):
            orig_lon, orig_lat, orig_time = prep.track_wgs[i]
            orig_pt_m = prep.track_metric[i]

            pt_cands = cands_by_pt.get_group(i) if i in cands_by_pt.groups else pd.DataFrame()
            top1_arc = int(pt_cands.iloc[0]["arc_id"]) if not pt_cands.empty else int(matched_arcs[i])

            chosen_arc = int(matched_arcs[i])
            chosen_row = pt_cands[pt_cands["arc_id"] == chosen_arc] if not pt_cands.empty else pd.DataFrame()

            if not chosen_row.empty:
                conf = float(chosen_row.iloc[0]["match_prob"])
                dist_m = float(chosen_row.iloc[0]["dist_to_segment"])
                proj_t = float(chosen_row.iloc[0]["proj_ratio"])
            else:
                conf = 0.5
                u_m = prep.arc_u_metric[chosen_arc]
                v_m = prep.arc_v_metric[chosen_arc]
                cab = v_m - u_m
                cap = orig_pt_m - u_m
                proj_t = float(np.clip(np.dot(cap, cab) / max(np.dot(cab, cab), 1e-12), 0.0, 1.0))
                proj_m = u_m + proj_t * cab
                dist_m = float(np.linalg.norm(orig_pt_m - proj_m))

            # Compute matched point lat/lon on arc segment
            u_m = prep.arc_u_metric[chosen_arc]
            v_m = prep.arc_v_metric[chosen_arc]
            matched_m = u_m + proj_t * (v_m - u_m)

            matched_lat = float(prep.ref_lat + (matched_m[1] / R) * (180.0 / np.pi))
            matched_lon = float(prep.ref_lon + (matched_m[0] / (R * cos_ref)) * (180.0 / np.pi))

            matched_points_list.append(MatchedPointResponse(
                point_idx=i,
                original_latitude=float(orig_lat),
                original_longitude=float(orig_lon),
                timestamp=float(orig_time),
                matched_arc_id=chosen_arc,
                matched_latitude=matched_lat,
                matched_longitude=matched_lon,
                orthogonal_distance_m=dist_m,
                confidence=conf,
                candidate_count=len(pt_cands),
                top1_ml_arc_id=top1_arc,
            ))

        processing_time = time.time() - t_start

        return MapMatchResponse(
            status="success",
            trajectory_id=resolved_tid,
            total_gps_points=num_pts,
            matched_points=matched_points_list,
            reconstructed_route=ReconstructedRouteResponse(
                ordered_arc_ids=route_info["ordered_route_arcs"],
                num_arcs=route_info["num_route_arcs"],
                total_length_meters=route_info["total_route_length_m"],
                intermediate_arcs_filled=route_info["intermediate_arcs_filled"],
                disconnected_jumps=route_info["disconnected_jumps"],
                is_fully_connected=route_info["is_fully_connected"],
                route_polyline_wgs84=route_info["route_coordinates_wgs84"],
            ),
            processing_time_seconds=float(processing_time),
            model_used=self.metadata.get("model_name", "Baseline XGBoost"),
            method=method_str,
        )

    def evaluate_trajectory(self, request: EvaluateRequest) -> EvaluateResponse:
        """
        Evaluates predicted road arc sequence against reference ground truth (.route).
        Ground truth is accessed exclusively during this evaluation phase, guaranteeing
        zero data leakage into inference.
        """
        t_start = time.time()
        tid = request.trajectory_id

        folder_train = os.path.join("data/raw/train", tid)
        folder_test = os.path.join("data/raw/student_test", tid)
        folder_eval = os.path.join("data/raw/evaluator_test", tid)

        if os.path.exists(folder_train):
            raw = load_single_trajectory(folder_train, traj_id=tid)
        elif os.path.exists(folder_test):
            raw = load_single_trajectory(folder_test, traj_id=tid)
        elif os.path.exists(folder_eval):
            raw = load_single_trajectory(folder_eval, traj_id=tid)
        else:
            raise FileNotFoundError(f"Trajectory '{tid}' not found for evaluation.")

        prep = preprocess_trajectory(raw, kinematic_mode="backward")
        _, gt_arcs, _ = viterbi_route_align(prep)

        matched_arcs = np.array(request.matched_arc_ids, dtype=int)
        min_len = min(len(matched_arcs), len(gt_arcs))
        if min_len > 0:
            pt_acc = float(np.mean(matched_arcs[:min_len] == gt_arcs[:min_len]) * 100.0)
        else:
            pt_acc = 0.0

        # Compute candidate recall & top1 baseline
        cands_df = generate_candidates_for_trajectory(
            prep, search_radius_m=50.0, max_candidates_per_point=25, sample_stride=1
        )
        scored_df = self.predictor.score_candidates(cands_df)
        top1_df = scored_df[scored_df["model_rank"] == 1].sort_values("point_idx")
        top1_arcs = top1_df["arc_id"].values
        top1_len = min(len(top1_arcs), len(gt_arcs))
        top1_acc = float(np.mean(top1_arcs[:top1_len] == gt_arcs[:top1_len]) * 100.0) if top1_len > 0 else 0.0

        cands_by_pt = cands_df.groupby("point_idx")
        recall_hits = 0
        for i in range(len(gt_arcs)):
            if i in cands_by_pt.groups:
                pt_arc_set = set(cands_by_pt.get_group(i)["arc_id"].values)
                if gt_arcs[i] in pt_arc_set:
                    recall_hits += 1
        cand_recall = float(recall_hits / max(len(gt_arcs), 1) * 100.0)

        # Route set metrics
        pred_route_set = set(request.reconstructed_route_arcs if request.reconstructed_route_arcs else matched_arcs)
        gt_route_set = set(prep.route)

        inter = len(pred_route_set.intersection(gt_route_set))
        union = len(pred_route_set.union(gt_route_set))
        jaccard = float(inter / max(union, 1) * 100.0)
        route_prec = float(inter / max(len(pred_route_set), 1) * 100.0)
        route_rec = float(inter / max(len(gt_route_set), 1) * 100.0)
        route_f1 = float(2 * route_prec * route_rec / max(route_prec + route_rec, 1e-6))

        # Check disconnected jumps in reconstructed route
        route_arcs = request.reconstructed_route_arcs if request.reconstructed_route_arcs else list(matched_arcs)
        jumps = 0
        filled = 0
        for idx in range(1, len(route_arcs)):
            p_arc = route_arcs[idx - 1]
            c_arc = route_arcs[idx]
            if p_arc != c_arc:
                if p_arc < len(prep.arcs) and c_arc < len(prep.arcs):
                    if int(prep.arcs[p_arc, 1]) != int(prep.arcs[c_arc, 0]):
                        jumps += 1
        conn_pct = float(max(0.0, 100.0 * (1.0 - jumps / max(len(matched_arcs) - 1, 1))))

        # Convert ground truth route arcs into WGS84 coordinates
        ref_lat = prep.ref_lat
        ref_lon = prep.ref_lon
        cos_ref = np.cos(np.radians(ref_lat))
        R = 6371000.0

        gt_polyline_wgs84 = []
        for aid in prep.route:
            if aid < len(prep.arcs):
                u_node = int(prep.arcs[aid, 0])
                v_node = int(prep.arcs[aid, 1])
                u_m = prep.nodes_metric[u_node]
                v_m = prep.nodes_metric[v_node]

                u_lat = float(ref_lat + (u_m[1] / R) * (180.0 / np.pi))
                u_lon = float(ref_lon + (u_m[0] / (R * cos_ref)) * (180.0 / np.pi))
                v_lat = float(ref_lat + (v_m[1] / R) * (180.0 / np.pi))
                v_lon = float(ref_lon + (v_m[0] / (R * cos_ref)) * (180.0 / np.pi))

                if not gt_polyline_wgs84:
                    gt_polyline_wgs84.append([u_lat, u_lon])
                gt_polyline_wgs84.append([v_lat, v_lon])

        eval_time = float(time.time() - t_start)

        return EvaluateResponse(
            status="success",
            trajectory_id=tid,
            total_gps_points=len(prep.track_wgs),
            point_wise_accuracy_pct=round(pt_acc, 2),
            top1_baseline_accuracy_pct=round(top1_acc, 2),
            accuracy_improvement_pct=round(pt_acc - top1_acc, 2),
            candidate_recall_pct=round(cand_recall, 2),
            route_precision_pct=round(route_prec, 2),
            route_recall_pct=round(route_rec, 2),
            route_f1_score_pct=round(route_f1, 2),
            route_jaccard_overlap_pct=round(jaccard, 2),
            topological_connectivity_pct=round(conn_pct, 2),
            disconnected_jumps=jumps,
            intermediate_arcs_filled=filled,
            num_ground_truth_route_arcs=len(gt_route_set),
            num_predicted_route_arcs=len(pred_route_set),
            ground_truth_polyline_wgs84=gt_polyline_wgs84,
            evaluation_time_seconds=round(eval_time, 3),
        )


# Global service singleton
_service_instance: Optional[MapMatchingService] = None


def get_map_matching_service() -> MapMatchingService:
    global _service_instance
    if _service_instance is None:
        _service_instance = MapMatchingService()
    return _service_instance
