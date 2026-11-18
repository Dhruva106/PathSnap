"""
candidate_generation.py - PathSnap Spatial Candidate Generation & Route Matching
Implements:
1. Trajectory-aware Viterbi route matching for ground-truth label generation.
2. Segment-aware spatial candidate generation with ZERO ground-truth injection.
3. Candidate recall evaluation and diagnostic analytics.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from typing import List, Dict, Tuple, Optional
from src.preprocessing import PreprocessedTrajectory


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """
    Computes Euclidean distance from point p to segment [a, b], projection parameter t, and projected point.
    p: [x, y], a: [x, y], b: [x, y]
    Returns: (distance, t_clamped, projected_point)
    """
    ab = b - a
    ap = p - a
    ab_len2 = np.dot(ab, ab)
    if ab_len2 <= 1e-12:
        return float(np.linalg.norm(ap)), 0.5, a
    t = np.clip(np.dot(ap, ab) / ab_len2, 0.0, 1.0)
    proj = a + t * ab
    dist = float(np.linalg.norm(p - proj))
    return dist, float(t), proj


def viterbi_route_align(
    prep: PreprocessedTrajectory,
    sigma: float = 10.0,
    max_forward_jump: int = 15,
    max_backward_jump: int = 1
) -> Tuple[np.ndarray, np.ndarray, Dict[str, any]]:
    """
    Trajectory-aware Dynamic Programming (Viterbi) alignment of GPS observations to the ground-truth route.
    Maintains progressive forward progression along the route sequence, prevents unconstrained backward jumps,
    and supports legitimate repeated traversals or loops.

    Returns:
    - matched_indices: Index into prep.route for each GPS point
    - gt_arc_ids: The ground-truth arc ID for each GPS point
    - alignment_stats: Diagnostic report on backward jumps and alignment quality
    """
    route_arcs = prep.route
    num_route = len(route_arcs)
    num_pts = len(prep.track_metric)

    ru = prep.arc_u_metric[route_arcs]
    rv = prep.arc_v_metric[route_arcs]
    track = prep.track_metric

    rab = rv - ru
    rab_len2 = np.sum(rab**2, axis=1)
    rab_len2[rab_len2 <= 1e-12] = 1e-12

    # Compute Euclidean distance from each point to every route arc
    dists = np.zeros((num_pts, num_route), dtype=np.float32)
    for i in range(num_pts):
        pt = track[i]
        rap = pt - ru
        dot_prod = np.sum(rap * rab, axis=1)
        t = np.clip(dot_prod / rab_len2, 0.0, 1.0)
        proj = ru + t[:, None] * rab
        dists[i] = np.linalg.norm(pt - proj, axis=1)

    # Emission cost = (distance / sigma)^2
    emission_cost = (dists / sigma)**2

    # Prior: slight penalty for starting deep in the route
    dp = emission_cost[0].copy() + np.arange(num_route) * 1.0
    parent = np.zeros((num_pts, num_route), dtype=np.int32)

    for t in range(1, num_pts):
        dt = max(prep.timestamps[t] - prep.timestamps[t-1], 0.1)
        speed = prep.speeds[t-1]
        max_fwd = max(4, int(np.ceil((speed * dt) / 15.0)))
        max_fwd = min(max_fwd, max_forward_jump)

        new_dp = np.full(num_route, 1e9, dtype=np.float32)
        for k in range(num_route):
            prev_min = max(0, k - max_fwd)
            prev_max = min(num_route, k + max_backward_jump + 1)
            diffs = k - np.arange(prev_min, prev_max)
            # Transition penalty: 0 for staying, linear penalty for advancing, steep penalty for backward jumps
            trans_pen = np.where(diffs == 0, 0.0, np.where(diffs > 0, diffs * 1.5, -diffs * 25.0))
            costs = dp[prev_min:prev_max] + trans_pen
            best_prev_idx = np.argmin(costs)
            parent[t, k] = prev_min + best_prev_idx
            new_dp[k] = costs[best_prev_idx] + emission_cost[t, k]
        dp = new_dp

    best_last = np.argmin(dp)
    matched_indices = np.zeros(num_pts, dtype=np.int32)
    curr = best_last
    for t in range(num_pts - 1, -1, -1):
        matched_indices[t] = curr
        curr = parent[t, curr]

    gt_arc_ids = route_arcs[matched_indices]

    diffs = np.diff(matched_indices)
    backward_jumps = int(np.sum(diffs < 0))
    forward_jumps_large = int(np.sum(diffs > 5))

    # Detect ambiguity: points where 2nd closest route arc is within 2 meters of closest
    sorted_dists = np.sort(dists, axis=1)
    ambiguous_count = int(np.sum((sorted_dists[:, 1] - sorted_dists[:, 0]) < 2.0))

    alignment_stats = {
        "traj_id": prep.traj_id,
        "total_points": num_pts,
        "num_route_arcs": num_route,
        "backward_jumps": backward_jumps,
        "backward_jump_pct": float(backward_jumps / max(1, num_pts - 1) * 100),
        "forward_jumps_large": forward_jumps_large,
        "ambiguous_points": ambiguous_count,
        "ambiguous_pct": float(ambiguous_count / num_pts * 100),
        "route_start_idx": int(matched_indices[0]),
        "route_end_idx": int(matched_indices[-1]),
    }

    return matched_indices, gt_arc_ids, alignment_stats


class SegmentSpatialIndex:
    """Segment-aware spatial index using arc discretization and cKDTree."""
    def __init__(self, prep: PreprocessedTrajectory, step_size: float = 35.0):
        self.prep = prep
        self.step_size = step_size

        u_pts = prep.arc_u_metric
        v_pts = prep.arc_v_metric
        lengths = prep.arc_lengths
        num_arcs = prep.num_arcs

        sample_coords = []
        sample_arc_ids = []

        for arc_id in range(num_arcs):
            L = lengths[arc_id]
            u = u_pts[arc_id]
            v = v_pts[arc_id]

            if L <= step_size:
                sample_coords.append(0.5 * (u + v))
                sample_arc_ids.append(arc_id)
            else:
                n_samples = int(np.ceil(L / step_size)) + 1
                t_vals = np.linspace(0.0, 1.0, n_samples)
                pts = u + t_vals[:, None] * (v - u)
                sample_coords.extend(pts)
                sample_arc_ids.extend([arc_id] * n_samples)

        self.sample_coords = np.array(sample_coords, dtype=np.float32)
        self.sample_arc_ids = np.array(sample_arc_ids, dtype=np.int32)
        self.tree = cKDTree(self.sample_coords)


def generate_candidates_for_trajectory(
    prep: PreprocessedTrajectory,
    search_radius_m: float = 50.0,
    max_candidates_per_point: int = 25,
    sample_stride: int = 1,
    spatial_index: Optional[SegmentSpatialIndex] = None
) -> pd.DataFrame:
    """
    Generates candidate road segments for points along a trajectory using ONLY GPS coordinates
    and road network geometry. ZERO ground truth injection or knowledge.
    Vectorized across track points for high computational efficiency.
    """
    if spatial_index is None:
        spatial_index = SegmentSpatialIndex(prep, step_size=35.0)

    point_indices = list(range(0, len(prep.track_metric), sample_stride))
    track_metric = prep.track_metric[point_indices]
    speeds = prep.speeds[point_indices]
    headings = prep.headings[point_indices]
    timestamps = prep.timestamps[point_indices]
    num_query_pts = len(point_indices)

    u_pts = prep.arc_u_metric
    v_pts = prep.arc_v_metric
    sample_arc_ids = spatial_index.sample_arc_ids
    tree = spatial_index.tree

    # Vectorized batch spatial query
    r_query = search_radius_m + (spatial_index.step_size / 2.0)
    all_sample_neighbors = tree.query_ball_point(track_metric, r=r_query)

    records = []
    for i in range(num_query_pts):
        pt_idx = point_indices[i]
        raw_ids = all_sample_neighbors[i]

        if len(raw_ids) == 0:
            # Fallback to nearest 3 samples in case of severe GPS noise spike
            _, raw_ids = tree.query(track_metric[i], k=min(3, len(spatial_index.sample_coords)))
            if isinstance(raw_ids, int):
                raw_ids = [raw_ids]
            else:
                raw_ids = list(raw_ids)

        cand_arcs = np.unique(sample_arc_ids[raw_ids])

        pt = track_metric[i]
        cu = u_pts[cand_arcs]
        cv = v_pts[cand_arcs]
        cab = cv - cu
        cab_len2 = np.sum(cab**2, axis=1)
        cab_len2[cab_len2 <= 1e-12] = 1e-12

        cap = pt - cu
        ct = np.clip(np.sum(cap * cab, axis=1) / cab_len2, 0.0, 1.0)
        cproj = cu + ct[:, None] * cab
        cdists = np.linalg.norm(pt - cproj, axis=1)

        valid = cdists <= search_radius_m
        if not np.any(valid):
            keep_idx = np.argsort(cdists)[:min(3, len(cand_arcs))]
        else:
            keep_idx = np.where(valid)[0]
            if len(keep_idx) > max_candidates_per_point:
                sub_order = np.argsort(cdists[keep_idx])[:max_candidates_per_point]
                keep_idx = keep_idx[sub_order]

        sorted_order = np.argsort(cdists[keep_idx])
        keep_idx = keep_idx[sorted_order]

        pt_speed = speeds[i]
        pt_heading = headings[i]
        pt_time = timestamps[i]
        min_d = max(cdists[keep_idx[0]], 0.1)

        for rank, kid in enumerate(keep_idx):
            arc_id = cand_arcs[kid]
            dist = cdists[kid]
            proj_t = ct[kid]
            proj_pt = cproj[kid]

            u_pt = cu[kid]
            v_pt = cv[kid]
            dist_u = float(np.linalg.norm(pt - u_pt))
            dist_v = float(np.linalg.norm(pt - v_pt))
            dist_mid = float(np.linalg.norm(pt - prep.arc_mids_metric[arc_id]))
            arc_len = float(prep.arc_lengths[arc_id])
            arc_heading = float(prep.arc_headings[arc_id])

            angle_diff = np.abs(pt_heading - arc_heading)
            angle_diff = np.mod(angle_diff, 2 * np.pi)
            if angle_diff > np.pi:
                angle_diff = 2 * np.pi - angle_diff
            heading_diff_deg = float(np.degrees(angle_diff))

            records.append({
                "trajectory_id": prep.traj_id,
                "point_idx": pt_idx,
                "timestamp": pt_time,
                "speed": pt_speed,
                "arc_id": arc_id,
                "dist_to_segment": dist,
                "dist_to_u": dist_u,
                "dist_to_v": dist_v,
                "dist_to_midpoint": dist_mid,
                "arc_length": arc_len,
                "proj_ratio": proj_t,
                "heading_diff_deg": heading_diff_deg,
                "delta_x": float(pt[0] - proj_pt[0]),
                "delta_y": float(pt[1] - proj_pt[1]),
                "dist_rank": rank + 1,
                "relative_dist": dist / min_d,
            })

    return pd.DataFrame(records)


def assign_ground_truth_labels(
    candidates_df: pd.DataFrame,
    gt_arc_ids: np.ndarray
) -> pd.DataFrame:
    """
    Explicitly assigns supervised labels (1 = true matching arc, 0 = distractor)
    AFTER candidate generation has concluded.
    """
    df = candidates_df.copy()
    point_indices = df["point_idx"].values
    arc_ids = df["arc_id"].values
    true_arcs_for_points = gt_arc_ids[point_indices]
    df["label"] = (arc_ids == true_arcs_for_points).astype(np.int32)
    return df


def evaluate_candidate_recall(
    prep: PreprocessedTrajectory,
    candidates_df: pd.DataFrame,
    gt_arc_ids: np.ndarray,
    search_radius_m: float = 50.0
) -> Dict[str, any]:
    """
    Evaluates Candidate Recall independently without ground-truth injection.
    """
    num_pts = len(prep.track_metric)

    # Compute distance to ground-truth arc
    ru = prep.arc_u_metric[gt_arc_ids]
    rv = prep.arc_v_metric[gt_arc_ids]
    track = prep.track_metric
    rab = rv - ru
    rab_len2 = np.sum(rab**2, axis=1)
    rab_len2[rab_len2 <= 1e-12] = 1e-12
    rap = track - ru
    t = np.clip(np.sum(rap * rab, axis=1) / rab_len2, 0.0, 1.0)
    proj = ru + t[:, None] * rab
    gt_dists = np.linalg.norm(track - proj, axis=1)

    cand_groups = candidates_df.groupby("point_idx")["arc_id"].apply(set).to_dict()

    valid_gt_pts = 0
    recalled_valid_pts = 0
    recalled_total_pts = 0
    zero_cands = 0
    counts = []

    for i in range(num_pts):
        cands = cand_groups.get(i, set())
        counts.append(len(cands))
        if len(cands) == 0:
            zero_cands += 1

        gt_arc = gt_arc_ids[i]
        is_recalled = gt_arc in cands

        if is_recalled:
            recalled_total_pts += 1

        if gt_dists[i] <= search_radius_m:
            valid_gt_pts += 1
            if is_recalled:
                recalled_valid_pts += 1

    recall_valid = (recalled_valid_pts / valid_gt_pts * 100.0) if valid_gt_pts > 0 else 0.0
    recall_overall = (recalled_total_pts / num_pts * 100.0) if num_pts > 0 else 0.0

    return {
        "traj_id": prep.traj_id,
        "total_points": num_pts,
        "valid_gt_points_within_radius": valid_gt_pts,
        "recalled_valid_points": recalled_valid_pts,
        "recalled_total_points": recalled_total_pts,
        "candidate_recall_valid_pct": float(recall_valid),
        "candidate_recall_overall_pct": float(recall_overall),
        "zero_candidate_points": zero_cands,
        "mean_candidates_per_point": float(np.mean(counts)) if counts else 0.0,
        "median_candidates_per_point": float(np.median(counts)) if counts else 0.0,
        "min_candidates_per_point": int(np.min(counts)) if counts else 0,
        "max_candidates_per_point": int(np.max(counts)) if counts else 0,
    }
