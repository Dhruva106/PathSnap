
"""
PathSnap - Dataset Loader

Loads GPS observations, road network nodes, directed arcs,
and reference routes for an individual trip.

Expected structure:
data/raw/
    train/
        <trip_id>/
            <trip_id>.track
            <trip_id>.nodes
            <trip_id>.arcs
            <trip_id>.route
    student_test/
        <trip_id>/
            ...
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TripData:
    """Container for all data associated with one trip."""

    trip_id: str

    # GPS observations: longitude, latitude, timestamp
    track: np.ndarray

    # Road nodes: longitude, latitude
    nodes: np.ndarray

    # Directed arcs: source node ID, destination node ID
    arcs: np.ndarray

    # Ordered sequence of reference arc IDs
    route: np.ndarray


def _load_numeric_file(file_path, expected_columns):
    """
    Read a whitespace-separated numeric file.

    Parameters
    ----------
    file_path : Path
        Path to the dataset file.

    expected_columns : int
        Number of columns expected in the file.

    Returns
    -------
    numpy.ndarray
        Two-dimensional numeric array.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}"
        )

    try:
        data = np.loadtxt(
            file_path,
            dtype=np.float64,
            ndmin=2
        )
    except ValueError as error:
        raise ValueError(
            f"Unable to parse {file_path}: {error}"
        ) from error

    if data.shape[1] != expected_columns:
        raise ValueError(
            f"Invalid structure in {file_path}. "
            f"Expected {expected_columns} columns, "
            f"found {data.shape[1]}."
        )

    if not np.all(np.isfinite(data)):
        raise ValueError(
            f"Non-finite values detected in {file_path}"
        )

    return data


def load_track(file_path):
    """Load GPS observations."""

    track = _load_numeric_file(file_path, 3)

    longitude = track[:, 0]
    latitude = track[:, 1]

    if np.any((longitude < -180) | (longitude > 180)):
        raise ValueError("Invalid longitude in GPS track.")

    if np.any((latitude < -90) | (latitude > 90)):
        raise ValueError("Invalid latitude in GPS track.")

    return track


def load_nodes(file_path):
    """Load geographic coordinates of road-network nodes."""

    nodes = _load_numeric_file(file_path, 2)

    longitude = nodes[:, 0]
    latitude = nodes[:, 1]

    if np.any((longitude < -180) | (longitude > 180)):
        raise ValueError("Invalid longitude in road nodes.")

    if np.any((latitude < -90) | (latitude > 90)):
        raise ValueError("Invalid latitude in road nodes.")

    return nodes


def load_arcs(file_path, number_of_nodes):
    """
    Load directed road connections.

    Arc IDs are represented by their row positions.
    Each row contains:
        source_node_id, destination_node_id
    """

    arcs = _load_numeric_file(file_path, 2)

    if not np.all(arcs == np.floor(arcs)):
        raise ValueError("Arc endpoints must be integer node IDs.")

    arcs = arcs.astype(np.int64)

    if np.any(arcs < 0) or np.any(arcs >= number_of_nodes):
        raise ValueError(
            "Arc file contains references to invalid node IDs."
        )

    return arcs


def load_route(file_path, number_of_arcs):
    """
    Load the reference route as an ordered sequence of arc IDs.

    Arc IDs are interpreted as zero-based row indices
    into the corresponding .arcs file.
    """

    route_data = _load_numeric_file(file_path, 1)

    if not np.all(route_data == np.floor(route_data)):
        raise ValueError("Route IDs must be integers.")

    route = route_data[:, 0].astype(np.int64)

    if np.any(route < 0) or np.any(route >= number_of_arcs):
        raise ValueError(
            "Reference route contains invalid arc IDs."
        )

    return route


def load_trip(trip_directory):
    """
    Load all four files belonging to one trip.

    Parameters
    ----------
    trip_directory : str or Path
        Directory named after the trip ID.

    Returns
    -------
    TripData
        Complete trip data.
    """

    trip_directory = Path(trip_directory)

    if not trip_directory.is_dir():
        raise NotADirectoryError(
            f"Trip directory does not exist: {trip_directory}"
        )

    trip_id = trip_directory.name

    track_path = trip_directory / f"{trip_id}.track"
    nodes_path = trip_directory / f"{trip_id}.nodes"
    arcs_path = trip_directory / f"{trip_id}.arcs"
    route_path = trip_directory / f"{trip_id}.route"

    # Load the data in dependency order.
    track = load_track(track_path)
    nodes = load_nodes(nodes_path)

    arcs = load_arcs(
        arcs_path,
        number_of_nodes=len(nodes)
    )

    route = load_route(
        route_path,
        number_of_arcs=len(arcs)
    )

    return TripData(
        trip_id=trip_id,
        track=track,
        nodes=nodes,
        arcs=arcs,
        route=route
    )


def load_dataset_split(data_root, split):
    """
    Load every trip belonging to a dataset split.

    Examples:
        split='train'
        split='student_test'

    Returns
    -------
    dict[str, TripData]
        Mapping of trip IDs to loaded trip objects.
    """

    if split not in {"train", "student_test"}:
        raise ValueError(
            "split must be either 'train' or 'student_test'."
        )

    split_directory = Path(data_root) / split

    if not split_directory.is_dir():
        raise FileNotFoundError(
            f"Dataset split not found: {split_directory}"
        )

    trip_directories = sorted(
        path for path in split_directory.iterdir()
        if path.is_dir()
    )

    if not trip_directories:
        raise ValueError(
            f"No trip directories found in {split_directory}"
        )

    dataset = {}

    for trip_directory in trip_directories:
        trip = load_trip(trip_directory)
        dataset[trip.trip_id] = trip

    return dataset
