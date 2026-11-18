
"""
PathSnap - Preprocessing and Data Quality Diagnostics

This module checks GPS observations and reference-route consistency
before candidate generation and HMM decoding.

Important:
- Original observations are preserved by default.
- GPS observations are never automatically sorted.
- Suspicious GPS jumps are reported, not deleted.
- Route discontinuities are diagnosed, not automatically repaired.
"""

import numpy as np


EARTH_RADIUS_METERS = 6_371_000


def haversine_distance(point_a, point_b):
    """
    Calculate the geographic distance between two GPS coordinates.

    Parameters
    ----------
    point_a, point_b : array-like
        Coordinates in (longitude, latitude) order.

    Returns
    -------
    float
        Distance in meters.
    """

    lon1, lat1 = np.radians(point_a)
    lon2, lat2 = np.radians(point_b)

    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1

    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(delta_lon / 2) ** 2
    )

    a = np.clip(a, 0.0, 1.0)

    return float(
        2 * EARTH_RADIUS_METERS * np.arcsin(np.sqrt(a))
    )


def count_consecutive_duplicates(track):
    """
    Count exact consecutive duplicate GPS observations.

    An observation is considered an exact duplicate only when
    longitude, latitude, and timestamp are all identical.
    """

    track = np.asarray(track)

    if len(track) < 2:
        return 0

    duplicates = np.all(
        track[1:] == track[:-1],
        axis=1
    )

    return int(np.sum(duplicates))


def preprocess_track(
    track,
    remove_consecutive_duplicates=False
):
    """
    Prepare a GPS track without changing temporal ordering.

    By default, this function returns a copy of the original track.

    Exact consecutive duplicates can optionally be removed.
    No sorting, interpolation, or outlier deletion is performed.
    """

    track = np.asarray(track, dtype=np.float64)

    if track.ndim != 2 or track.shape[1] != 3:
        raise ValueError(
            "Track must have shape (number_of_points, 3)."
        )

    if not np.all(np.isfinite(track)):
        raise ValueError(
            "Track contains non-finite values."
        )

    processed = track.copy()

    if remove_consecutive_duplicates and len(processed) > 1:
        keep = np.ones(len(processed), dtype=bool)

        keep[1:] = np.any(
            processed[1:] != processed[:-1],
            axis=1
        )

        processed = processed[keep]

    return processed


def inspect_gps_track(
    track,
    max_time_gap_seconds=120.0,
    max_speed_mps=55.0
):
    """
    Inspect GPS observations for temporal and spatial anomalies.

    Large gaps and high implied speeds are diagnostic warnings.
    They do not automatically invalidate or remove observations.
    """

    track = np.asarray(track, dtype=np.float64)

    if track.ndim != 2 or track.shape[1] != 3:
        raise ValueError(
            "Track must have shape (number_of_points, 3)."
        )

    if not np.all(np.isfinite(track)):
        raise ValueError(
            "Track contains non-finite values."
        )

    if len(track) == 0:
        raise ValueError("Track is empty.")

    timestamps = track[:, 2]
    time_differences = np.diff(timestamps)

    non_monotonic_count = int(
        np.sum(time_differences < 0)
    )

    duplicate_timestamp_count = int(
        np.sum(time_differences == 0)
    )

    large_time_gap_indices = np.where(
        time_differences > max_time_gap_seconds
    )[0].tolist()

    distances = []
    implied_speeds = []
    suspicious_speed_indices = []

    for index in range(len(track) - 1):
        point_a = track[index, :2]
        point_b = track[index + 1, :2]

        distance = haversine_distance(point_a, point_b)
        time_difference = time_differences[index]

        distances.append(distance)

        if time_difference > 0:
            speed = distance / time_difference
            implied_speeds.append(speed)

            if speed > max_speed_mps:
                suspicious_speed_indices.append(index)

    return {
        "observation_count": int(len(track)),
        "non_monotonic_timestamp_count": non_monotonic_count,
        "duplicate_timestamp_count": duplicate_timestamp_count,
        "exact_consecutive_duplicate_count":
            count_consecutive_duplicates(track),
        "large_time_gap_count": len(large_time_gap_indices),
        "large_time_gap_indices": large_time_gap_indices,
        "suspicious_speed_count": len(suspicious_speed_indices),
        "suspicious_speed_indices": suspicious_speed_indices,
        "maximum_observed_gap_seconds": (
            float(np.max(time_differences))
            if len(time_differences) else 0.0
        ),
        "maximum_implied_speed_mps": (
            float(np.max(implied_speeds))
            if implied_speeds else None
        ),
        "total_observed_distance_meters": float(
            np.sum(distances)
        ),
    }


def inspect_route_connectivity(arcs, route):
    """
    Diagnose consecutive reference-route arc connectivity.

    Directed continuity:
        previous_arc.destination == next_arc.source

    Undirected continuity:
        The two arcs share at least one endpoint.

    These are diagnostics only. A discontinuity is not automatically
    proof that the reference route is incorrect.
    """

    arcs = np.asarray(arcs)
    route = np.asarray(route)

    if arcs.ndim != 2 or arcs.shape[1] != 2:
        raise ValueError(
            "Arcs must have shape (number_of_arcs, 2)."
        )

    if route.ndim != 1:
        raise ValueError("Route must be a one-dimensional array.")

    directed_discontinuities = []
    undirected_discontinuities = []
    invalid_route_ids = []

    for route_position, arc_id in enumerate(route):
        if (
            not np.isfinite(arc_id)
            or arc_id != int(arc_id)
            or arc_id < 0
            or arc_id >= len(arcs)
        ):
            invalid_route_ids.append(route_position)

    valid_route = [
        int(arc_id)
        for arc_id in route
        if (
            np.isfinite(arc_id)
            and arc_id == int(arc_id)
            and 0 <= arc_id < len(arcs)
        )
    ]

    for position, (left_id, right_id) in enumerate(
        zip(valid_route, valid_route[1:])
    ):
        left_source, left_target = arcs[left_id]
        right_source, right_target = arcs[right_id]

        directed_connected = left_target == right_source

        endpoints_left = {
            left_source,
            left_target
        }

        endpoints_right = {
            right_source,
            right_target
        }

        undirected_connected = bool(
            endpoints_left.intersection(endpoints_right)
        )

        if not directed_connected:
            directed_discontinuities.append(position)

        if not undirected_connected:
            undirected_discontinuities.append(position)

    return {
        "route_length": int(len(route)),
        "invalid_route_id_count": len(invalid_route_ids),
        "invalid_route_id_positions": invalid_route_ids,
        "directed_discontinuity_count":
            len(directed_discontinuities),
        "directed_discontinuity_positions":
            directed_discontinuities,
        "undirected_discontinuity_count":
            len(undirected_discontinuities),
        "undirected_discontinuity_positions":
            undirected_discontinuities,
    }


def inspect_trip(trip):
    """
    Run GPS and route diagnostics on a loaded TripData object.

    Parameters
    ----------
    trip : TripData
        Object returned by data_loader.load_trip().

    Returns
    -------
    dict
        Structured quality report.
    """

    gps_report = inspect_gps_track(trip.track)

    route_report = inspect_route_connectivity(
        trip.arcs,
        trip.route
    )

    return {
        "trip_id": trip.trip_id,
        "gps": gps_report,
        "route": route_report,
    }


def preprocess_trip(
    trip,
    remove_consecutive_duplicates=False
):
    """
    Return a processed copy of a trip's GPS observations.

    The road graph and reference route remain unchanged.

    This function does not mutate the original TripData object.
    """

    return {
        "trip_id": trip.trip_id,
        "track": preprocess_track(
            trip.track,
            remove_consecutive_duplicates=
                remove_consecutive_duplicates
        ),
        "nodes": np.asarray(trip.nodes).copy(),
        "arcs": np.asarray(trip.arcs).copy(),
        "route": np.asarray(trip.route).copy(),
    }
