"""
hmm_decoder.py - PathSnap Hidden Markov Model and Viterbi Sequence Decoder
Reconstructs temporally and topologically consistent road sequences from ML candidate predictions.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Union

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import PreprocessedTrajectory


class PathSnapHMMDecoder:
    """
    Hidden Markov Model with Viterbi decoding for map matching.
    - Hidden States: Candidate road arcs at each GPS observation point
    - Emission Probabilities: Derived from the trained champion ML model posterior probabilities
      and orthogonal distance to road segment
    - Transition Probabilities: Newson-Krumm network-to-Euclidean routing distance consistency model
    """

    def __init__(
        self,
        beta: float = 3.0,                  # Transition distance scale parameter (meters)
        emission_weight: float = 1.0,        # Weight for log ML emission probability
        transition_weight: float = 1.0,      # Weight for log transition probability
        sigma_z: float = 10.0,               # Spatial GPS measurement error std dev (meters)
        max_dist_cutoff_factor: float = 2.5, # Multiplier for GPS step distance cutoff in Dijkstra
        min_cutoff_m: float = 150.0,         # Minimum search cutoff in Dijkstra (meters)
        disconnect_penalty: float = -20.0,   # Log-prob penalty when no directed path connects candidates
        turn_penalty_rad: float = 0.5,       # Penalty for sharp heading changes between connected arcs
    ):
        self.beta = beta
        self.emission_weight = emission_weight
        self.transition_weight = transition_weight
        self.sigma_z = sigma_z
        self.max_dist_cutoff_factor = max_dist_cutoff_factor
        self.min_cutoff_m = min_cutoff_m
        self.disconnect_penalty = disconnect_penalty
        self.turn_penalty_rad = turn_penalty_rad

    def build_road_graph(self, prep: PreprocessedTrajectory) -> nx.DiGraph:
        """
        Builds a directed NetworkX graph representing the road network for shortest path calculations.
        Nodes: integer node IDs
        Edges: (u, v) with weight = metric arc length
        """
        G = nx.DiGraph()
        arcs = prep.arcs
        lengths = prep.arc_lengths
        num_arcs = len(arcs)

        for arc_id in range(num_arcs):
            u = int(arcs[arc_id, 0])
            v = int(arcs[arc_id, 1])
            length = float(lengths[arc_id])
            # Keep the minimum length if parallel edges exist
            if G.has_edge(u, v):
                if length < G[u][v]["weight"]:
                    G[u][v]["weight"] = length
                    G[u][v]["arc_id"] = arc_id
            else:
                G.add_edge(u, v, weight=max(length, 0.1), arc_id=arc_id)

        return G

    def compute_emission_log_prob(
        self,
        ml_prob: float,
        dist_to_segment: float
    ) -> float:
        """
        Computes the log emission probability combining ML classifier posterior and spatial Gaussian noise.
        ln P(z | r) = w_ml * ln(max(ml_prob, eps)) - (dist^2 / (2 * sigma_z^2))
        """
        clamped_ml = max(float(ml_prob), 1e-6)
        log_ml = np.log(clamped_ml)
        gaussian_term = -0.5 * (float(dist_to_segment) / max(self.sigma_z, 0.1)) ** 2
        return float(self.emission_weight * log_ml + 0.2 * gaussian_term)

    def compute_network_distance(
        self,
        G: nx.DiGraph,
        arc_a: int,
        arc_b: int,
        proj_t_a: float,
        proj_t_b: float,
        prep: PreprocessedTrajectory,
        cached_dijkstra: Dict[int, Dict[int, float]],
        d_gps: float
    ) -> Tuple[float, bool]:
        """
        Computes network shortest path distance between projected point on arc_a to projected point on arc_b.
        Returns: (distance_meters, is_connected_bool)
        """
        if arc_a == arc_b:
            length_a = float(prep.arc_lengths[arc_a])
            delta_t = proj_t_b - proj_t_a
            if delta_t >= -0.05:
                # Normal forward progression on same segment
                return max(delta_t * length_a, 0.0), True
            else:
                # Apparent backward movement along directed one-way segment
                backward_dist = abs(delta_t) * length_a
                return backward_dist * 2.0, False

        # Different arcs: Path is (remaining on arc_a) + (shortest path v_a -> u_b) + (traversed on arc_b)
        u_a, v_a = int(prep.arcs[arc_a, 0]), int(prep.arcs[arc_a, 1])
        u_b, v_b = int(prep.arcs[arc_b, 0]), int(prep.arcs[arc_b, 1])

        length_a = float(prep.arc_lengths[arc_a])
        length_b = float(prep.arc_lengths[arc_b])

        rem_a = max((1.0 - proj_t_a) * length_a, 0.0)
        traversed_b = max(proj_t_b * length_b, 0.0)

        if v_a == u_b:
            # Directly adjacent connected segments
            return rem_a + traversed_b, True

        # Query Dijkstra cache
        if v_a not in cached_dijkstra:
            cutoff = max(d_gps * self.max_dist_cutoff_factor + 50.0, self.min_cutoff_m)
            try:
                cached_dijkstra[v_a] = nx.single_source_dijkstra_path_length(
                    G, v_a, cutoff=cutoff, weight="weight"
                )
            except Exception:
                cached_dijkstra[v_a] = {}

        node_paths = cached_dijkstra[v_a]
        if u_b in node_paths:
            path_len = node_paths[u_b]
            return rem_a + path_len + traversed_b, True

        # If disconnected within cutoff, estimate via Euclidean gap between nodes plus penalty
        coord_va = prep.nodes_metric[v_a]
        coord_ub = prep.nodes_metric[u_b]
        euclid_gap = float(np.linalg.norm(coord_va - coord_ub))
        fallback_dist = rem_a + euclid_gap * 1.5 + traversed_b
        return fallback_dist, False

    def compute_transition_log_prob(
        self,
        net_dist: float,
        d_gps: float,
        is_connected: bool
    ) -> float:
        """
        Newson-Krumm exponential transition model:
        ln P(r_a -> r_b) = -|d_net - d_gps| / beta + (disconnect_penalty if not connected)
        """
        diff = abs(net_dist - d_gps)
        log_prob = - (diff / max(self.beta, 0.1))
        if not is_connected:
            log_prob += self.disconnect_penalty
        return float(self.transition_weight * log_prob)

    def decode(
        self,
        prep: PreprocessedTrajectory,
        scored_candidates_df: pd.DataFrame,
        road_graph: Optional[nx.DiGraph] = None
    ) -> Dict[str, Any]:
        """
        Executes Viterbi dynamic programming decoding over candidate states.
        Returns:
        - matched_arc_ids: Array of chosen road arc IDs (one per GPS point)
        - viterbi_log_prob: Final joint sequence log probability
        - candidate_ranks: Array of model_rank of selected candidates (1 = top-1 ML agreed)
        - is_connected_transitions: Boolean array indicating whether consecutive arcs were topological neighbors
        - route_summary: Statistical metrics on decoding
        """
        t0 = time.time()
        G = road_graph if road_graph is not None else self.build_road_graph(prep)

        point_indices = sorted(scored_candidates_df["point_idx"].unique())
        num_timesteps = len(point_indices)
        if num_timesteps == 0:
            return {
                "matched_arc_ids": np.array([], dtype=int),
                "viterbi_log_prob": 0.0,
                "candidate_ranks": np.array([], dtype=int),
                "is_connected_transitions": np.array([], dtype=bool),
                "route_summary": {},
            }

        # Group candidates by timestep
        grouped = scored_candidates_df.groupby("point_idx")
        cands_per_step = []
        for p_idx in point_indices:
            sub = grouped.get_group(p_idx).copy()
            cands_per_step.append(sub.to_dict("records"))

        # Precompute GPS track coordinates and consecutive step displacements
        track_metric = prep.track_metric[point_indices]
        d_gps_steps = [0.0]
        for t in range(1, num_timesteps):
            disp = float(np.linalg.norm(track_metric[t] - track_metric[t - 1]))
            d_gps_steps.append(disp)

        # Dijkstra shortest-path query cache
        cached_dijkstra: Dict[int, Dict[int, float]] = {}

        # -------------------------------------------------------------
        # VITERBI INITIALIZATION (t = 0)
        # -------------------------------------------------------------
        init_cands = cands_per_step[0]
        num_init = len(init_cands)
        viterbi_scores = np.zeros(num_init, dtype=np.float64)

        for i, cand in enumerate(init_cands):
            emit_lp = self.compute_emission_log_prob(
                cand["match_prob"], cand["dist_to_segment"]
            )
            # Prior: slight penalty for candidates with lower ML rank
            rank_prior = -0.5 * (cand.get("dist_rank", 1) - 1)
            viterbi_scores[i] = emit_lp + rank_prior

        # Backpointers: backpointers[t][i] = index of best predecessor candidate at t-1
        backpointers: List[np.ndarray] = []
        transition_connectivity: List[np.ndarray] = []

        # -------------------------------------------------------------
        # VITERBI RECURSION (t = 1 to T-1)
        # -------------------------------------------------------------
        for t in range(1, num_timesteps):
            curr_cands = cands_per_step[t]
            prev_cands = cands_per_step[t - 1]
            num_curr = len(curr_cands)
            num_prev = len(prev_cands)
            d_gps = d_gps_steps[t]

            new_viterbi = np.full(num_curr, -1e18, dtype=np.float64)
            bp_step = np.zeros(num_curr, dtype=np.int32)
            conn_step = np.zeros(num_curr, dtype=bool)

            for j in range(num_curr):
                cand_b = curr_cands[j]
                arc_b = int(cand_b["arc_id"])
                proj_b = float(cand_b["proj_ratio"])

                emit_lp = self.compute_emission_log_prob(
                    cand_b["match_prob"], cand_b["dist_to_segment"]
                )

                best_prev_idx = 0
                max_score = -1e18
                best_connected = False

                for i in range(num_prev):
                    cand_a = prev_cands[i]
                    arc_a = int(cand_a["arc_id"])
                    proj_a = float(cand_a["proj_ratio"])

                    net_dist, is_connected = self.compute_network_distance(
                        G, arc_a, arc_b, proj_a, proj_b, prep, cached_dijkstra, d_gps
                    )

                    trans_lp = self.compute_transition_log_prob(net_dist, d_gps, is_connected)
                    score = viterbi_scores[i] + trans_lp + emit_lp

                    if score > max_score:
                        max_score = score
                        best_prev_idx = i
                        best_connected = is_connected

                new_viterbi[j] = max_score
                bp_step[j] = best_prev_idx
                conn_step[j] = best_connected

            viterbi_scores = new_viterbi
            backpointers.append(bp_step)
            transition_connectivity.append(conn_step)

        # -------------------------------------------------------------
        # VITERBI TERMINATION & BACKTRACKING
        # -------------------------------------------------------------
        best_last_idx = int(np.argmax(viterbi_scores))
        total_log_prob = float(viterbi_scores[best_last_idx])

        matched_arc_ids = np.zeros(num_timesteps, dtype=np.int32)
        matched_model_ranks = np.zeros(num_timesteps, dtype=np.int32)
        matched_probs = np.zeros(num_timesteps, dtype=np.float64)
        is_connected_list = []

        curr_idx = best_last_idx
        for t in range(num_timesteps - 1, -1, -1):
            chosen_cand = cands_per_step[t][curr_idx]
            matched_arc_ids[t] = int(chosen_cand["arc_id"])
            matched_model_ranks[t] = int(chosen_cand.get("model_rank", 1))
            matched_probs[t] = float(chosen_cand["match_prob"])
            if t > 0:
                conn_val = transition_connectivity[t - 1][curr_idx]
                is_connected_list.append(conn_val)
                curr_idx = backpointers[t - 1][curr_idx]

        is_connected_arr = np.array(is_connected_list[::-1], dtype=bool)

        # Compute summary metrics
        decoding_time = time.time() - t0
        top1_agreements = int(np.sum(matched_model_ranks == 1))
        connected_pct = float(np.mean(is_connected_arr) * 100.0) if len(is_connected_arr) > 0 else 100.0
        unique_matched_arcs = len(np.unique(matched_arc_ids))

        # Check for backward jumps
        arc_changes = np.where(matched_arc_ids[:-1] != matched_arc_ids[1:])[0]

        summary = {
            "num_timesteps": num_timesteps,
            "decoding_time_sec": float(decoding_time),
            "total_log_prob": total_log_prob,
            "mean_confidence": float(np.mean(matched_probs)),
            "top1_agreement_pct": float(top1_agreements / max(num_timesteps, 1) * 100.0),
            "topological_connectivity_pct": connected_pct,
            "unique_matched_arcs": unique_matched_arcs,
            "arc_transitions_count": len(arc_changes),
        }

        return {
            "matched_arc_ids": matched_arc_ids,
            "viterbi_log_prob": total_log_prob,
            "candidate_ranks": matched_model_ranks,
            "matched_probabilities": matched_probs,
            "is_connected_transitions": is_connected_arr,
            "route_summary": summary,
        }
