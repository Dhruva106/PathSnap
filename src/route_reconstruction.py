"""
route_reconstruction.py - PathSnap Full Road Network Route Reconstruction
Interpolates intermediate road graph arcs between point-wise Viterbi matches to produce
a continuous, topologically valid driving route with exact WGS84 and metric geometry.
"""

import os
import sys
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import PreprocessedTrajectory


def reconstruct_continuous_route(
    prep: PreprocessedTrajectory,
    matched_arc_ids: np.ndarray,
    road_graph: Optional[nx.DiGraph] = None
) -> Dict[str, Any]:
    """
    Takes a sequence of point-wise matched arc IDs and fills in any intermediate graph edges
    to form a continuous, topologically connected road arc route sequence.
    """
    if len(matched_arc_ids) == 0:
        return {
            "ordered_route_arcs": [],
            "total_route_length_m": 0.0,
            "intermediate_arcs_filled": 0,
            "route_coordinates_wgs84": [],
            "is_fully_connected": True,
        }

    arcs = prep.arcs
    lengths = prep.arc_lengths
    num_arcs = len(arcs)

    # Build or reuse graph with arc_id mapping
    if road_graph is None:
        G = nx.DiGraph()
        for aid in range(num_arcs):
            u = int(arcs[aid, 0])
            v = int(arcs[aid, 1])
            l = float(lengths[aid])
            if not G.has_edge(u, v) or l < G[u][v]["weight"]:
                G.add_edge(u, v, weight=max(l, 0.1), arc_id=aid)
    else:
        G = road_graph

    # Map node pairs to arc_ids
    edge_to_arc = {}
    for aid in range(num_arcs):
        u = int(arcs[aid, 0])
        v = int(arcs[aid, 1])
        if (u, v) not in edge_to_arc or lengths[aid] < lengths[edge_to_arc[(u, v)]]:
            edge_to_arc[(u, v)] = aid

    # Reconstruct route sequence
    complete_route_arcs: List[int] = [int(matched_arc_ids[0])]
    intermediate_filled = 0
    disconnected_jumps = 0

    num_pts = len(matched_arc_ids)
    for t in range(1, num_pts):
        prev_arc = int(matched_arc_ids[t - 1])
        curr_arc = int(matched_arc_ids[t])

        if curr_arc == prev_arc:
            continue

        prev_v = int(arcs[prev_arc, 1])
        curr_u = int(arcs[curr_arc, 0])

        if prev_v == curr_u:
            # Directly adjacent connected segments
            complete_route_arcs.append(curr_arc)
        else:
            # Need to find shortest path between prev_v and curr_u in G
            try:
                path_nodes = nx.shortest_path(G, source=prev_v, target=curr_u, weight="weight")
                # Insert intermediate arcs
                for i in range(len(path_nodes) - 1):
                    n_from = path_nodes[i]
                    n_to = path_nodes[i + 1]
                    intermediate_arc = edge_to_arc.get((n_from, n_to), G[n_from][n_to].get("arc_id"))
                    if intermediate_arc is not None and intermediate_arc != complete_route_arcs[-1]:
                        complete_route_arcs.append(intermediate_arc)
                        intermediate_filled += 1
                complete_route_arcs.append(curr_arc)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # Disconnected jump
                complete_route_arcs.append(curr_arc)
                disconnected_jumps += 1

    # Calculate total length
    total_length = float(np.sum(lengths[complete_route_arcs]))

    # Generate WGS84 coordinate centerline polyline [[lat, lon], ...]
    # using prep.ref_lat, prep.ref_lon to convert back from metric nodes
    ref_lat = prep.ref_lat
    ref_lon = prep.ref_lon
    cos_ref = np.cos(np.radians(ref_lat))
    R = 6371000.0

    route_coords_wgs84 = []
    for aid in complete_route_arcs:
        u_node = int(arcs[aid, 0])
        v_node = int(arcs[aid, 1])
        u_m = prep.nodes_metric[u_node]
        v_m = prep.nodes_metric[v_node]

        # Invert Equirectangular projection
        u_lat = float(ref_lat + (u_m[1] / R) * (180.0 / np.pi))
        u_lon = float(ref_lon + (u_m[0] / (R * cos_ref)) * (180.0 / np.pi))
        v_lat = float(ref_lat + (v_m[1] / R) * (180.0 / np.pi))
        v_lon = float(ref_lon + (v_m[0] / (R * cos_ref)) * (180.0 / np.pi))

        if not route_coords_wgs84:
            route_coords_wgs84.append([u_lat, u_lon])
        route_coords_wgs84.append([v_lat, v_lon])

    return {
        "ordered_route_arcs": complete_route_arcs,
        "num_route_arcs": len(complete_route_arcs),
        "total_route_length_m": total_length,
        "intermediate_arcs_filled": intermediate_filled,
        "disconnected_jumps": disconnected_jumps,
        "is_fully_connected": (disconnected_jumps == 0),
        "route_coordinates_wgs84": route_coords_wgs84,
    }
