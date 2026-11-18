"""
preprocessing.py - PathSnap Coordinate Transformation and Kinematic Preprocessing
Transforms geographic coordinates (WGS84) to local metric coordinate frames,
computes causal trajectory kinematics (speed, heading, delta time), and prepares road network geometry.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from src.data_loader import TrajectoryData


class PreprocessedTrajectory:
    """Encapsulates converted metric coordinates and kinematic attributes."""
    def __init__(
        self,
        traj_id: str,
        track_wgs: np.ndarray,
        track_metric: np.ndarray,
        timestamps: np.ndarray,
        speeds: np.ndarray,
        headings: np.ndarray,
        nodes_metric: np.ndarray,
        arcs: np.ndarray,
        arc_u_metric: np.ndarray,
        arc_v_metric: np.ndarray,
        arc_mids_metric: np.ndarray,
        arc_lengths: np.ndarray,
        arc_headings: np.ndarray,
        route: np.ndarray,
        ref_lat: float,
        ref_lon: float,
        kinematic_mode: str = "backward"
    ):
        self.traj_id = traj_id
        self.track_wgs = track_wgs
        self.track_metric = track_metric
        self.timestamps = timestamps
        self.speeds = speeds
        self.headings = headings
        self.nodes_metric = nodes_metric
        self.arcs = arcs
        self.arc_u_metric = arc_u_metric
        self.arc_v_metric = arc_v_metric
        self.arc_mids_metric = arc_mids_metric
        self.arc_lengths = arc_lengths
        self.arc_headings = arc_headings
        self.route = route
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self.kinematic_mode = kinematic_mode

    @property
    def num_arcs(self) -> int:
        return len(self.arcs)

    @property
    def num_points(self) -> int:
        return len(self.track_metric)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes_metric)


def wgs84_to_metric_factors(lat_center_deg: float) -> Tuple[float, float]:
    """
    Returns meters per degree of longitude and latitude at a reference latitude.
    Based on WGS84 ellipsoid approximations.
    """
    lat_rad = np.radians(lat_center_deg)
    # WGS-84 meters per degree
    kx = 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3 * lat_rad) + 0.118 * np.cos(5 * lat_rad)
    ky = 111132.95 - 559.82 * np.cos(2 * lat_rad) + 1.175 * np.cos(4 * lat_rad)
    return kx, ky


def preprocess_trajectory(
    data: TrajectoryData,
    kinematic_mode: str = "backward"
) -> PreprocessedTrajectory:
    """
    Preprocesses raw trajectory data:
    1. Computes local Cartesian metric coordinates (easting/northing meters relative to trajectory center)
    2. Calculates causal instantaneous speeds and headings for GPS track points
       (default 'backward' difference ensures no future-observation leakage)
    3. Precomputes arc lengths, headings, and midpoints for rapid spatial candidate generation
    """
    # 1. Reference origin: mean track position
    ref_lon = float(np.mean(data.track[:, 0]))
    ref_lat = float(np.mean(data.track[:, 1]))
    kx, ky = wgs84_to_metric_factors(ref_lat)

    # 2. Metric track coordinates: shape (N, 2) [x_m, y_m]
    track_x = (data.track[:, 0] - ref_lon) * kx
    track_y = (data.track[:, 1] - ref_lat) * ky
    track_metric = np.column_stack([track_x, track_y])
    timestamps = data.track[:, 2]

    # 3. Kinematic computations: speed and heading
    num_pts = len(track_metric)
    speeds = np.zeros(num_pts, dtype=np.float64)
    headings = np.zeros(num_pts, dtype=np.float64)

    if num_pts > 1:
        if kinematic_mode == "backward":
            # Strictly causal backward difference: p_t - p_{t-1}
            dx = np.diff(track_x)
            dy = np.diff(track_y)
            dt = np.diff(timestamps)
            dt_safe = np.where(dt <= 0.0, 1.0, dt)
            step_dists = np.hypot(dx, dy)
            step_speeds = step_dists / dt_safe
            step_headings = np.arctan2(dy, dx)

            # Point 0 has no prior observation: assign step 0->1 values
            speeds[0] = step_speeds[0]
            headings[0] = step_headings[0]
            speeds[1:] = step_speeds
            headings[1:] = step_headings
        else:
            # Forward difference: p_{t+1} - p_t
            dx = np.diff(track_x)
            dy = np.diff(track_y)
            dt = np.diff(timestamps)
            dt_safe = np.where(dt <= 0.0, 1.0, dt)
            step_dists = np.hypot(dx, dy)
            step_speeds = step_dists / dt_safe
            step_headings = np.arctan2(dy, dx)

            speeds[:-1] = step_speeds
            speeds[-1] = step_speeds[-1]
            headings[:-1] = step_headings
            headings[-1] = step_headings[-1]

    # 4. Metric road network nodes: shape (M, 2)
    nodes_x = (data.nodes[:, 0] - ref_lon) * kx
    nodes_y = (data.nodes[:, 1] - ref_lat) * ky
    nodes_metric = np.column_stack([nodes_x, nodes_y])

    # 5. Road network arcs: vectors, lengths, headings, midpoints
    u_idx = data.arcs[:, 0]
    v_idx = data.arcs[:, 1]
    arc_u_metric = nodes_metric[u_idx]
    arc_v_metric = nodes_metric[v_idx]

    arc_dx = arc_v_metric[:, 0] - arc_u_metric[:, 0]
    arc_dy = arc_v_metric[:, 1] - arc_u_metric[:, 1]
    arc_lengths = np.hypot(arc_dx, arc_dy)
    arc_headings = np.arctan2(arc_dy, arc_dx)
    arc_mids_metric = 0.5 * (arc_u_metric + arc_v_metric)

    return PreprocessedTrajectory(
        traj_id=data.traj_id,
        track_wgs=data.track,
        track_metric=track_metric,
        timestamps=timestamps,
        speeds=speeds,
        headings=headings,
        nodes_metric=nodes_metric,
        arcs=data.arcs,
        arc_u_metric=arc_u_metric,
        arc_v_metric=arc_v_metric,
        arc_mids_metric=arc_mids_metric,
        arc_lengths=arc_lengths,
        arc_headings=arc_headings,
        route=data.route,
        ref_lat=ref_lat,
        ref_lon=ref_lon,
        kinematic_mode=kinematic_mode
    )
