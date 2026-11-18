
import numpy as np


EARTH_RADIUS_M = 6_371_000.0


def _local_coordinates(points, origin_lon, origin_lat):
    """
    Convert longitude/latitude coordinates into local Cartesian metres.

    Uses an equirectangular projection, suitable for the local area
    covered by an individual trip.
    """
    points = np.asarray(points, dtype=np.float64)

    x = (
        np.radians(points[:, 0] - origin_lon)
        * EARTH_RADIUS_M
        * np.cos(np.radians(origin_lat))
    )

    y = (
        np.radians(points[:, 1] - origin_lat)
        * EARTH_RADIUS_M
    )

    return np.column_stack((x, y))


def _point_to_segment_distances(point, starts, ends):
    """
    Calculate perpendicular distances from one GPS point
    to multiple road segments.

    All inputs must use the same local Cartesian coordinate system.
    """
    relative_starts = starts - point
    vectors = ends - starts

    denominator = np.maximum(
        np.sum(vectors * vectors, axis=1),
        1e-12
    )

    fractions = np.clip(
        -np.sum(relative_starts * vectors, axis=1)
        / denominator,
        0.0,
        1.0
    )

    projections = (
        relative_starts
        + fractions[:, None] * vectors
    )

    return np.linalg.norm(projections, axis=1)


def align_track_to_reference_route(
    track,
    nodes,
    arcs,
    route,
    top_k=10,
    candidate_radius_m=50.0,
    emission_sigma_m=10.0,
    transition_scale_m=25.0,
):
    """
    Align GPS observations to an ordered reference route using
    candidate generation and monotonic dynamic programming.

    Parameters
    ----------
    track : ndarray
        GPS observations with columns [longitude, latitude, timestamp].

    nodes : ndarray
        Road network nodes with columns [longitude, latitude].

    arcs : ndarray
        Directed road arcs with columns [from_node_id, to_node_id].

    route : ndarray
        Ordered reference arc IDs.

    top_k : int
        Maximum number of candidate route positions per observation.

    candidate_radius_m : float
        Maximum distance for considering a route segment.

    emission_sigma_m : float
        Distance scale for emission costs.

    transition_scale_m : float
        Distance scale for transition costs.

    Returns
    -------
    dict
        route_positions:
            Selected positions in the reference route.

        arc_ids:
            Selected original road-network arc IDs.

        distances_m:
            GPS-to-selected-segment distances.

        confidence:
            Relative distance-based candidate confidence.
            This is NOT a calibrated probability.

        candidate_positions:
            Candidate route positions for each GPS observation.
    """

    # ---------------------------------------------------------
    # 1. Convert and validate input data
    # ---------------------------------------------------------

    track = np.asarray(track, dtype=np.float64)
    nodes = np.asarray(nodes, dtype=np.float64)
    arcs = np.asarray(arcs, dtype=np.int64)
    route = np.asarray(route, dtype=np.int64)

    if track.ndim != 2 or track.shape[1] < 3:
        raise ValueError("Track must have at least three columns.")

    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError("Nodes must have exactly two columns.")

    if arcs.ndim != 2 or arcs.shape[1] != 2:
        raise ValueError("Arcs must have exactly two columns.")

    if len(track) == 0 or len(route) == 0:
        raise ValueError("Track and reference route must not be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if candidate_radius_m <= 0:
        raise ValueError("candidate_radius_m must be positive.")

    if emission_sigma_m <= 0 or transition_scale_m <= 0:
        raise ValueError("Cost scales must be positive.")

    if not np.all(np.isfinite(track[:, :3])):
        raise ValueError("Track contains non-finite values.")

    if np.any(route < 0) or np.any(route >= len(arcs)):
        raise ValueError("Reference route contains invalid arc IDs.")

    # ---------------------------------------------------------
    # 2. Extract road segments used by the reference route
    # ---------------------------------------------------------

    route_arcs = arcs[route]

    if (
        np.any(route_arcs < 0)
        or np.any(route_arcs >= len(nodes))
    ):
        raise ValueError("Reference route contains invalid node IDs.")

    starts_geo = nodes[route_arcs[:, 0]]
    ends_geo = nodes[route_arcs[:, 1]]

    # ---------------------------------------------------------
    # 3. Project coordinates into local Cartesian metres
    # ---------------------------------------------------------

    origin_lon = float(np.mean(track[:, 0]))
    origin_lat = float(np.mean(track[:, 1]))

    starts = _local_coordinates(
        starts_geo,
        origin_lon,
        origin_lat
    )

    ends = _local_coordinates(
        ends_geo,
        origin_lon,
        origin_lat
    )

    gps_points = _local_coordinates(
        track[:, :2],
        origin_lon,
        origin_lat
    )

    # ---------------------------------------------------------
    # 4. Calculate reference route segment lengths
    # ---------------------------------------------------------

    segment_vectors = ends - starts

    segment_lengths = np.linalg.norm(
        segment_vectors,
        axis=1
    )

    cumulative_lengths = np.zeros(
        len(route),
        dtype=np.float64
    )

    if len(route) > 1:
        cumulative_lengths[1:] = np.cumsum(
            segment_lengths[:-1]
        )

    # ---------------------------------------------------------
    # 5. Generate candidate route positions for every GPS point
    # ---------------------------------------------------------

    candidate_positions = []
    candidate_distances = []

    for point in gps_points:

        distances = _point_to_segment_distances(
            point,
            starts,
            ends
        )

        eligible = np.flatnonzero(
            distances <= candidate_radius_m
        )

        # Retain the nearest segment if none falls within
        # the specified candidate radius.
        if len(eligible) == 0:
            eligible = np.array([
                int(np.argmin(distances))
            ])

        # Select the nearest top_k route positions.
        order = eligible[
            np.argsort(distances[eligible])[:top_k]
        ]

        candidate_positions.append(order)
        candidate_distances.append(distances[order])

    # ---------------------------------------------------------
    # 6. Dynamic programming initialization
    # ---------------------------------------------------------

    first_emission = (
        candidate_distances[0] / emission_sigma_m
    ) ** 2

    costs = [first_emission]

    backpointers = [
        np.full(
            len(first_emission),
            -1,
            dtype=np.int64
        )
    ]

    # ---------------------------------------------------------
    # 7. Sequence-aware route alignment
    # ---------------------------------------------------------

    for i in range(1, len(track)):

        previous_positions = candidate_positions[i - 1]
        current_positions = candidate_positions[i]

        previous_costs = costs[-1]
        current_distances = candidate_distances[i]

        gps_displacement = np.linalg.norm(
            gps_points[i] - gps_points[i - 1]
        )

        current_costs = np.full(
            len(current_positions),
            np.inf
        )

        current_backpointers = np.full(
            len(current_positions),
            -1,
            dtype=np.int64
        )

        for j, current_position in enumerate(current_positions):

            # Route positions must not move backwards.
            valid_previous = np.flatnonzero(
                previous_positions <= current_position
            )

            if len(valid_previous) == 0:
                continue

            previous_route_positions = (
                previous_positions[valid_previous]
            )

            route_displacement = (
                cumulative_lengths[current_position]
                - cumulative_lengths[previous_route_positions]
            )

            # Penalize disagreement between GPS displacement
            # and reference-route displacement.
            transition_cost = (
                (
                    route_displacement - gps_displacement
                ) / transition_scale_m
            ) ** 2

            total_cost = (
                previous_costs[valid_previous]
                + transition_cost
            )

            best_local_index = int(
                np.argmin(total_cost)
            )

            best_previous_index = int(
                valid_previous[best_local_index]
            )

            emission_cost = (
                current_distances[j] / emission_sigma_m
            ) ** 2

            current_costs[j] = (
                total_cost[best_local_index]
                + emission_cost
            )

            current_backpointers[j] = best_previous_index

        if not np.any(np.isfinite(current_costs)):
            raise RuntimeError(
                f"No valid monotonic alignment at GPS index {i}."
            )

        costs.append(current_costs)
        backpointers.append(current_backpointers)

    # ---------------------------------------------------------
    # 8. Backtrack to recover the optimal alignment
    # ---------------------------------------------------------

    selected_indices = np.zeros(
        len(track),
        dtype=np.int64
    )

    selected_indices[-1] = int(
        np.argmin(costs[-1])
    )

    for i in range(len(track) - 1, 0, -1):

        selected_indices[i - 1] = (
            backpointers[i][selected_indices[i]]
        )

    aligned_positions = np.array([
        candidate_positions[i][selected_indices[i]]
        for i in range(len(track))
    ], dtype=np.int64)

    aligned_distances = np.array([
        candidate_distances[i][selected_indices[i]]
        for i in range(len(track))
    ])

    aligned_arc_ids = route[aligned_positions]

    # ---------------------------------------------------------
    # 9. Calculate relative candidate confidence
    # ---------------------------------------------------------

    confidence = np.zeros(
        len(track),
        dtype=np.float64
    )

    for i in range(len(track)):

        distances = candidate_distances[i]

        scores = np.exp(
            -0.5 * (distances / emission_sigma_m) ** 2
        )

        total_score = np.sum(scores)

        if total_score > 0:
            confidence[i] = (
                scores[selected_indices[i]] / total_score
            )

    # ---------------------------------------------------------
    # 10. Return alignment results
    # ---------------------------------------------------------

    return {
        "route_positions": aligned_positions,
        "arc_ids": aligned_arc_ids,
        "distances_m": aligned_distances,
        "confidence": confidence,
        "candidate_positions": candidate_positions,
    }
