
"""
PathSnap - Feature Engineering

Extracts geometric and directional features between GPS observations
and candidate directed road arcs.

Coordinate convention:
    GPS track: longitude, latitude, timestamp
    Nodes: longitude, latitude
    Arcs: source_node_id, destination_node_id

Distances are calculated using a local equirectangular projection.
This is suitable for short road segments and local map-matching
calculations. It is not intended for global geodesic measurements.
"""

import numpy as np

from src.preprocessing import haversine_distance


EARTH_RADIUS_METERS = 6_371_000.0


def calculate_bearing(point_a, point_b):
    """
    Calculate initial bearing from point A to point B.

    Points use (longitude, latitude) order.

    Returns
    -------
    float
        Bearing in degrees clockwise from geographic north,
        in the range [0, 360).
    """

    lon1, lat1 = np.radians(point_a)
    lon2, lat2 = np.radians(point_b)

    delta_lon = lon2 - lon1

    x = np.sin(delta_lon) * np.cos(lat2)

    y = (
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
    )

    return float(np.degrees(np.arctan2(x, y)) % 360)


def calculate_heading_difference(heading_a, heading_b):
    """
    Calculate the smallest angular difference between two bearings.

    Returns a value between 0 and 180 degrees.
    """

    difference = abs(heading_a - heading_b) % 360

    return float(min(difference, 360 - difference))


def calculate_gps_heading(previous_point, current_point):
    """
    Estimate vehicle heading from two consecutive GPS observations.

    Each point must contain at least longitude and latitude.
    """

    if previous_point is None or current_point is None:
        return None

    distance = haversine_distance(
        previous_point[:2],
        current_point[:2]
    )

    if distance < 0.01:
        return None

    return calculate_bearing(
        previous_point[:2],
        current_point[:2]
    )


def project_point_to_segment(gps_point, start_node, end_node):
    """
    Project a GPS point onto a road segment using a local
    equirectangular coordinate system.

    Returns
    -------
    tuple
        perpendicular_distance_meters,
        projection_fraction

    Projection fraction:
        0.0 = start of segment
        1.0 = end of segment
        Values outside [0, 1] indicate projection beyond an endpoint.
    """

    gps_lon, gps_lat = gps_point[:2]
    start_lon, start_lat = start_node
    end_lon, end_lat = end_node

    reference_latitude = np.radians(gps_lat)

    longitude_scale = (
        EARTH_RADIUS_METERS * np.cos(reference_latitude)
        * np.pi / 180
    )

    latitude_scale = EARTH_RADIUS_METERS * np.pi / 180

    start_x = (start_lon - gps_lon) * longitude_scale
    start_y = (start_lat - gps_lat) * latitude_scale

    end_x = (end_lon - gps_lon) * longitude_scale
    end_y = (end_lat - gps_lat) * latitude_scale

    segment_x = end_x - start_x
    segment_y = end_y - start_y

    segment_squared_length = (
        segment_x ** 2 + segment_y ** 2
    )

    if segment_squared_length == 0:
        distance = np.hypot(start_x, start_y)
        return float(distance), 0.0

    projection_fraction = -(
        start_x * segment_x + start_y * segment_y
    ) / segment_squared_length

    projection_x = (
        start_x + projection_fraction * segment_x
    )

    projection_y = (
        start_y + projection_fraction * segment_y
    )

    perpendicular_distance = np.hypot(
        projection_x,
        projection_y
    )

    return (
        float(perpendicular_distance),
        float(projection_fraction)
    )


def calculate_segment_length(start_node, end_node):
    """Calculate road-segment length in meters."""

    return haversine_distance(start_node, end_node)


def extract_arc_features(
    gps_point,
    nodes,
    arcs,
    candidate_arc_ids,
    gps_heading=None
):
    """
    Extract geometric features for candidate road arcs.

    Parameters
    ----------
    gps_point : array-like
        GPS observation: longitude, latitude, timestamp.

    nodes : numpy.ndarray
        Road-network node coordinates.

    arcs : numpy.ndarray
        Directed arcs represented by source and target node IDs.

    candidate_arc_ids : array-like
        Arc row indices to evaluate.

    gps_heading : float, optional
        GPS movement bearing in degrees.

    Returns
    -------
    dict
        One feature array per candidate arc.
    """

    nodes = np.asarray(nodes, dtype=np.float64)
    arcs = np.asarray(arcs, dtype=np.int64)

    candidate_arc_ids = np.asarray(
        candidate_arc_ids,
        dtype=np.int64
    )

    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError("Nodes must have shape (N, 2).")

    if arcs.ndim != 2 or arcs.shape[1] != 2:
        raise ValueError("Arcs must have shape (M, 2).")

    if not np.all(np.isfinite(gps_point[:3])):
        raise ValueError("GPS observation contains non-finite values.")

    if np.any(candidate_arc_ids < 0) or np.any(
        candidate_arc_ids >= len(arcs)
    ):
        raise ValueError("Invalid candidate arc ID.")

    distances = []
    projection_fractions = []
    road_bearings = []
    heading_differences = []
    segment_lengths = []

    for arc_id in candidate_arc_ids:
        source_id, target_id = arcs[arc_id]

        start_node = nodes[source_id]
        end_node = nodes[target_id]

        distance, fraction = project_point_to_segment(
            gps_point,
            start_node,
            end_node
        )

        road_bearing = calculate_bearing(
            start_node,
            end_node
        )

        segment_length = calculate_segment_length(
            start_node,
            end_node
        )

        if gps_heading is None:
            heading_difference = np.nan
        else:
            heading_difference = calculate_heading_difference(
                gps_heading,
                road_bearing
            )

        distances.append(distance)
        projection_fractions.append(fraction)
        road_bearings.append(road_bearing)
        heading_differences.append(heading_difference)
        segment_lengths.append(segment_length)

    return {
        "arc_ids": candidate_arc_ids.copy(),
        "perpendicular_distance_m": np.asarray(distances),
        "projection_fraction": np.asarray(projection_fractions),
        "road_bearing_deg": np.asarray(road_bearings),
        "heading_difference_deg": np.asarray(heading_differences),
        "segment_length_m": np.asarray(segment_lengths),
    }


def extract_transition_features(
    previous_arc_id,
    current_arc_id,
    arcs,
    nodes,
    previous_gps_point,
    current_gps_point
):
    """
    Calculate basic transition features between two candidate arcs.

    This function does not decide whether a transition is valid.
    It reports endpoint connectivity and geographic distances
    for later transition modelling.
    """

    arcs = np.asarray(arcs, dtype=np.int64)
    nodes = np.asarray(nodes, dtype=np.float64)

    if not (0 <= previous_arc_id < len(arcs)):
        raise ValueError("Invalid previous arc ID.")

    if not (0 <= current_arc_id < len(arcs)):
        raise ValueError("Invalid current arc ID.")

    previous_source, previous_target = arcs[previous_arc_id]
    current_source, current_target = arcs[current_arc_id]

    previous_end = nodes[previous_target]
    current_start = nodes[current_source]

    endpoint_gap = haversine_distance(
        previous_end,
        current_start
    )

    directed_connected = previous_target == current_source

    undirected_connected = bool(
        {previous_source, previous_target}
        .intersection({current_source, current_target})
    )

    gps_distance = haversine_distance(
        previous_gps_point[:2],
        current_gps_point[:2]
    )

    time_difference = (
        current_gps_point[2] - previous_gps_point[2]
    )

    return {
        "previous_arc_id": int(previous_arc_id),
        "current_arc_id": int(current_arc_id),
        "directed_connected": bool(directed_connected),
        "undirected_connected": undirected_connected,
        "endpoint_gap_m": float(endpoint_gap),
        "gps_displacement_m": float(gps_distance),
        "time_difference": float(time_difference),
        "implied_gps_speed_mps": (
            float(gps_distance / time_difference)
            if time_difference > 0 else None
        ),
    }
