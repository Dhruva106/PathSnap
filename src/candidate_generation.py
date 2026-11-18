
"""
Candidate road-segment generation for PathSnap.

Uses a SciPy cKDTree over road-segment midpoints to efficiently
retrieve nearby arcs for each GPS observation.
"""

import numpy as np
from scipy.spatial import cKDTree

from src.feature_engineering import extract_arc_features


EARTH_RADIUS_METERS = 6_371_000.0


class CandidateGenerator:
    """Spatial index and candidate generator for one trip's road network."""

    def __init__(
        self,
        nodes,
        arcs,
        search_radius_m=50.0,
        max_candidates=20,
    ):
        self.nodes = np.asarray(nodes, dtype=np.float64)
        raw_arcs = np.asarray(arcs)

        if self.nodes.ndim != 2 or self.nodes.shape[1] != 2:
            raise ValueError("nodes must have shape (N, 2)")

        if raw_arcs.ndim != 2 or raw_arcs.shape[1] != 2:
            raise ValueError("arcs must have shape (M, 2)")

        if len(self.nodes) == 0 or len(raw_arcs) == 0:
            raise ValueError("Road network cannot be empty")

        if not np.all(np.isfinite(self.nodes)):
            raise ValueError("nodes contains non-finite coordinates")

        if not np.all(np.isfinite(raw_arcs)):
            raise ValueError("arcs contains non-finite values")

        if not np.all(raw_arcs == np.floor(raw_arcs)):
            raise ValueError("Arc endpoint IDs must be integers")

        self.arcs = raw_arcs.astype(np.int64)

        if np.any(self.arcs < 0) or np.any(
            self.arcs >= len(self.nodes)
        ):
            raise ValueError("Arc contains an invalid node ID")

        if search_radius_m <= 0:
            raise ValueError("search_radius_m must be positive")

        if max_candidates < 1:
            raise ValueError("max_candidates must be at least 1")

        self.search_radius_m = float(search_radius_m)
        self.max_candidates = int(max_candidates)

        # Use one local coordinate system for the entire trip.
        self.reference_lat = float(np.mean(self.nodes[:, 1]))
        self.longitude_scale = np.cos(
            np.radians(self.reference_lat)
        )

        if abs(self.longitude_scale) < 1e-8:
            raise ValueError("Unsupported coordinate reference latitude")

        self.arc_midpoints_xy = self._to_local_xy(
            (
                self.nodes[self.arcs[:, 0]]
                + self.nodes[self.arcs[:, 1]]
            ) / 2.0
        )

        start_xy = self._to_local_xy(
            self.nodes[self.arcs[:, 0]]
        )
        end_xy = self._to_local_xy(
            self.nodes[self.arcs[:, 1]]
        )

        # Half of each segment's length in the local projection.
        half_lengths = (
            np.linalg.norm(end_xy - self.arc_midpoints_xy, axis=1)
        )

        # Any point within search_radius of a segment must be within
        # search_radius + maximum_half_length of its midpoint.
        self.max_half_length_m = float(np.max(half_lengths))

        self.tree = cKDTree(self.arc_midpoints_xy)

    def _to_local_xy(self, coordinates):
        """Convert (longitude, latitude) coordinates to local metres."""
        coordinates = np.asarray(coordinates, dtype=np.float64)

        x = (
            np.radians(coordinates[..., 0])
            * EARTH_RADIUS_METERS
            * self.longitude_scale
        )
        y = (
            np.radians(coordinates[..., 1])
            * EARTH_RADIUS_METERS
        )

        return np.stack((x, y), axis=-1)

    def generate_for_point(
        self,
        gps_point,
        gps_heading=None,
        search_radius_m=None,
    ):
        """
        Generate candidate arcs for one GPS observation.

        Returns a dictionary containing candidate IDs and features.
        An empty candidate set is returned if no arc is within range.
        """
        gps_point = np.asarray(gps_point, dtype=np.float64)

        if gps_point.ndim != 1 or len(gps_point) < 2:
            raise ValueError(
                "gps_point must contain longitude and latitude"
            )

        if not np.all(np.isfinite(gps_point[:2])):
            raise ValueError("GPS coordinates must be finite")

        radius = (
            self.search_radius_m
            if search_radius_m is None
            else float(search_radius_m)
        )

        if radius <= 0:
            raise ValueError("Search radius must be positive")

        gps_xy = self._to_local_xy(gps_point[:2])

        midpoint_search_radius = (
            radius + self.max_half_length_m
        )

        possible_arc_ids = self.tree.query_ball_point(
            gps_xy,
            midpoint_search_radius,
        )

        if not possible_arc_ids:
            return self._empty_result()

        possible_arc_ids = np.asarray(
            possible_arc_ids, dtype=np.int64
        )

        features = extract_arc_features(
            gps_point,
            self.nodes,
            self.arcs,
            possible_arc_ids,
            gps_heading,
        )

        # Keep only candidates whose actual point-to-segment distance
        # falls within the requested radius.
        valid = (
            features["perpendicular_distance_m"] <= radius
        )

        candidate_ids = features["arc_ids"][valid]

        if len(candidate_ids) == 0:
            return self._empty_result()

        candidate_distances = (
            features["perpendicular_distance_m"][valid]
        )

        # Retain the closest candidates if the result is too large.
        if len(candidate_ids) > self.max_candidates:
            order = np.argsort(candidate_distances)[
                :self.max_candidates
            ]
            candidate_ids = candidate_ids[order]

        return extract_arc_features(
            gps_point,
            self.nodes,
            self.arcs,
            candidate_ids,
            gps_heading,
        )

    def generate_for_track(self, track):
        """
        Generate candidate sets for every GPS observation.

        Returns a list. Each element is the candidate-feature
        dictionary for the corresponding GPS observation.
        """
        track = np.asarray(track, dtype=np.float64)

        if track.ndim != 2 or track.shape[1] != 3:
            raise ValueError(
                "track must have shape (N, 3): "
                "longitude, latitude, timestamp"
            )

        if not np.all(np.isfinite(track)):
            raise ValueError("track contains non-finite values")

        results = []

        for index, gps_point in enumerate(track):
            gps_heading = None

            if index > 0:
                previous_point = track[index - 1]
                current_point = gps_point

                from src.feature_engineering import (
                    calculate_gps_heading,
                )

                gps_heading = calculate_gps_heading(
                    previous_point, current_point
                )

            results.append(
                self.generate_for_point(
                    gps_point,
                    gps_heading=gps_heading,
                )
            )

        return results

    @staticmethod
    def _empty_result():
        """Return a consistently shaped empty candidate result."""
        return {
            "arc_ids": np.array([], dtype=np.int64),
            "perpendicular_distance_m": np.array(
                [], dtype=np.float64
            ),
            "projection_fraction": np.array(
                [], dtype=np.float64
            ),
            "road_bearing_deg": np.array(
                [], dtype=np.float64
            ),
            "heading_difference_deg": np.array(
                [], dtype=np.float64
            ),
            "segment_length_m": np.array(
                [], dtype=np.float64
            ),
        }
