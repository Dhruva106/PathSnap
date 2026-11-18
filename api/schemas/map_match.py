"""
map_match.py - Pydantic Request and Response Schemas for PathSnap API
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, field_validator


class GPSPoint(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0, description="WGS84 latitude coordinate in degrees")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="WGS84 longitude coordinate in degrees")
    timestamp: float = Field(..., description="Epoch timestamp in seconds")


class MapMatchRequest(BaseModel):
    trajectory_id: Optional[str] = Field(
        None,
        description="Optional ID of a trajectory in data/raw/train or data/raw/student_test (e.g. '00000001' or '00000000')"
    )
    points: Optional[List[GPSPoint]] = Field(
        None,
        description="Sequence of GPS track points to match against the road network"
    )
    use_hmm: bool = Field(
        True,
        description="If True, runs HMM Viterbi sequence decoding; if False, returns independent top-1 ML argmax"
    )
    search_radius_m: float = Field(
        50.0,
        ge=10.0,
        le=150.0,
        description="Spatial search radius in meters around each GPS point for candidate generation"
    )
    max_candidates: int = Field(
        25,
        ge=1,
        le=50,
        description="Maximum candidate road arcs to consider per GPS observation"
    )

    @field_validator("points")
    @classmethod
    def validate_points(cls, v):
        if v is not None:
            if len(v) < 2:
                raise ValueError("Trajectory must contain at least 2 GPS observation points.")
            # Check timestamps are non-decreasing
            for i in range(1, len(v)):
                if v[i].timestamp < v[i - 1].timestamp:
                    raise ValueError(f"Timestamps must be non-decreasing (at index {i}: {v[i].timestamp} < {v[i-1].timestamp}).")
        return v


class MatchedPointResponse(BaseModel):
    point_idx: int
    original_latitude: float
    original_longitude: float
    timestamp: float
    matched_arc_id: int
    matched_latitude: float
    matched_longitude: float
    orthogonal_distance_m: float
    confidence: float
    candidate_count: int
    top1_ml_arc_id: int


class ReconstructedRouteResponse(BaseModel):
    ordered_arc_ids: List[int]
    num_arcs: int
    total_length_meters: float
    intermediate_arcs_filled: int
    disconnected_jumps: int
    is_fully_connected: bool
    route_polyline_wgs84: List[List[float]]


class MapMatchResponse(BaseModel):
    status: str
    trajectory_id: Optional[str]
    total_gps_points: int
    matched_points: List[MatchedPointResponse]
    reconstructed_route: ReconstructedRouteResponse
    processing_time_seconds: float
    model_used: str
    method: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    champion_model: str
    artifacts_available: bool
    service_operational: bool
    timestamp: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    version: str
    status: str
    selected_features: List[str]
    validation_metrics: Dict[str, float]
    training_instances: int
    validation_instances: int


class EvaluateRequest(BaseModel):
    trajectory_id: str = Field(..., description="ID of trajectory to evaluate against server-side ground truth")
    matched_arc_ids: List[int] = Field(..., description="Sequence of matched road arc IDs from map-matching")
    reconstructed_route_arcs: Optional[List[int]] = Field(None, description="Ordered road arcs in continuous reconstructed route")


class EvaluateResponse(BaseModel):
    status: str
    trajectory_id: str
    total_gps_points: int
    point_wise_accuracy_pct: float
    top1_baseline_accuracy_pct: float
    accuracy_improvement_pct: float
    candidate_recall_pct: float
    route_precision_pct: float
    route_recall_pct: float
    route_f1_score_pct: float
    route_jaccard_overlap_pct: float
    topological_connectivity_pct: float
    disconnected_jumps: int
    intermediate_arcs_filled: int
    num_ground_truth_route_arcs: int
    num_predicted_route_arcs: int
    ground_truth_polyline_wgs84: List[List[float]]
    evaluation_time_seconds: float
