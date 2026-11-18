"""
data_loader.py - PathSnap Dataset Loader and Validator
Loads and verifies GPS trajectories, road network nodes, arcs, and ground-truth routes.
"""

import os
import glob
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


class TrajectoryData:
    """Container for a single trajectory and its associated road network."""
    def __init__(self, traj_id: str, track: np.ndarray, nodes: np.ndarray, arcs: np.ndarray, route: np.ndarray):
        self.traj_id = traj_id
        # track: [lon, lat, timestamp]
        self.track = track
        # nodes: [lon, lat] indexed 0 to N-1
        self.nodes = nodes
        # arcs: [u, v] indexed 0 to M-1
        self.arcs = arcs
        # route: array of arc_ids
        self.route = route

    @property
    def num_points(self) -> int:
        return len(self.track)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_arcs(self) -> int:
        return len(self.arcs)

    @property
    def num_route_arcs(self) -> int:
        return len(self.route)


def load_single_trajectory(folder_path: str, traj_id: Optional[str] = None) -> TrajectoryData:
    """
    Loads all 4 raw files for a given trajectory folder.
    Files: <traj_id>.track, <traj_id>.nodes, <traj_id>.arcs, <traj_id>.route
    """
    if traj_id is None:
        traj_id = os.path.basename(os.path.normpath(folder_path))

    track_path = os.path.join(folder_path, f"{traj_id}.track")
    nodes_path = os.path.join(folder_path, f"{traj_id}.nodes")
    arcs_path = os.path.join(folder_path, f"{traj_id}.arcs")
    route_path = os.path.join(folder_path, f"{traj_id}.route")

    for p in [track_path, nodes_path, arcs_path, route_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required data file: {p}")

    # Load track: lon, lat, timestamp
    track_pts = []
    with open(track_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                track_pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
    track = np.array(track_pts, dtype=np.float64)

    # Load nodes: lon, lat
    node_pts = []
    with open(nodes_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                node_pts.append([float(parts[0]), float(parts[1])])
    nodes = np.array(node_pts, dtype=np.float64)

    # Load arcs: u, v (node indices)
    arc_list = []
    with open(arcs_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                arc_list.append([int(parts[0]), int(parts[1])])
    arcs = np.array(arc_list, dtype=np.int64)

    # Load route: sequence of arc indices
    route_list = []
    with open(route_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip()
            if parts:
                route_list.append(int(parts))
    route = np.array(route_list, dtype=np.int64)

    return TrajectoryData(traj_id=traj_id, track=track, nodes=nodes, arcs=arcs, route=route)


def validate_trajectory_data(data: TrajectoryData) -> Dict[str, any]:
    """
    Performs data integrity verification on loaded trajectory data.
    Checks coordinate bounds, node references, arc references, and timestamps.
    """
    issues = []

    # 1. Check coordinates
    if np.any(data.track[:, 0] < -180.0) or np.any(data.track[:, 0] > 180.0):
        issues.append("Track longitude out of bounds [-180, 180]")
    if np.any(data.track[:, 1] < -90.0) or np.any(data.track[:, 1] > 90.0):
        issues.append("Track latitude out of bounds [-90, 90]")
    if np.any(data.nodes[:, 0] < -180.0) or np.any(data.nodes[:, 0] > 180.0):
        issues.append("Nodes longitude out of bounds [-180, 180]")
    if np.any(data.nodes[:, 1] < -90.0) or np.any(data.nodes[:, 1] > 90.0):
        issues.append("Nodes latitude out of bounds [-90, 90]")

    # 2. Check timestamps
    if len(data.track) > 1:
        dts = np.diff(data.track[:, 2])
        if np.any(dts < 0):
            issues.append(f"Non-monotonic timestamps detected in {data.traj_id}")

    # 3. Check arc node bounds
    if len(data.arcs) > 0:
        if np.any(data.arcs < 0) or np.any(data.arcs >= data.num_nodes):
            issues.append(f"Arc references invalid node indices in {data.traj_id}")

    # 4. Check route arc bounds
    if len(data.route) > 0:
        if np.any(data.route < 0) or np.any(data.route >= data.num_arcs):
            issues.append(f"Route references invalid arc indices in {data.traj_id}")

    return {
        "traj_id": data.traj_id,
        "valid": len(issues) == 0,
        "issues": issues,
        "num_track_points": data.num_points,
        "num_nodes": data.num_nodes,
        "num_arcs": data.num_arcs,
        "num_route_arcs": data.num_route_arcs,
        "time_span_sec": float(data.track[-1, 2] - data.track[0, 2]) if len(data.track) > 1 else 0.0,
    }


def load_all_trajectories(base_dir: str, limit: Optional[int] = None) -> List[TrajectoryData]:
    """
    Loads all trajectories found in subdirectories of base_dir.
    """
    dirs = sorted([d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)])
    if limit is not None:
        dirs = dirs[:limit]

    trajectories = []
    for d in dirs:
        traj_id = os.path.basename(d)
        data = load_single_trajectory(d, traj_id)
        trajectories.append(data)
    return trajectories
