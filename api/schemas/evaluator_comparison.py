"""
evaluator_comparison.py - Pydantic Data Contracts for Evaluator Dataset & Teammate Benchmarks
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


class EvaluatorTrajectorySummary(BaseModel):
    trajectory_id: str
    num_points: int
    num_nodes: int
    num_arcs: int
    num_route_arcs: int
    duration_seconds: float
    bounding_box: Dict[str, float]


class TrajectoryEvaluationResult(BaseModel):
    trajectory_id: str
    gps_observation_count: int
    candidate_count: int
    candidate_recall_pct: float
    point_wise_accuracy_pct: float
    ml_only_accuracy_pct: float
    hmm_accuracy_pct: float
    hmm_improvement_pct: float
    route_precision_pct: float
    route_recall_pct: float
    route_f1_score_pct: float
    route_jaccard_overlap_pct: float
    topological_connectivity_pct: float
    disconnected_jumps: int
    intermediate_arcs_filled: int
    pred_route_length_m: float
    gt_route_length_m: float
    route_distance_error_m: float
    candidate_generation_time_sec: float
    ml_inference_time_sec: float
    hmm_decoding_time_sec: float
    route_reconstruction_time_sec: float
    total_processing_time_sec: float
    evaluation_status: str


class AggregateMacroMetrics(BaseModel):
    mean_point_wise_accuracy_pct: float
    mean_top1_ml_accuracy_pct: float
    mean_hmm_improvement_pct: float
    mean_candidate_recall_pct: float
    mean_route_precision_pct: float
    mean_route_recall_pct: float
    mean_route_f1_score_pct: float
    mean_route_jaccard_overlap_pct: float
    mean_topological_connectivity_pct: float
    total_processing_time_sec: float
    average_processing_time_sec: float


class AggregatePooledMetrics(BaseModel):
    pooled_point_wise_accuracy_pct: float
    pooled_top1_ml_accuracy_pct: float
    pooled_hmm_improvement_pct: float
    pooled_candidate_recall_pct: float
    total_evaluated_points: int
    total_hmm_matches: int
    total_top1_matches: int
    total_cand_recall_hits: int


class AggregateEvaluationResult(BaseModel):
    dataset_name: str
    num_trajectories: int
    total_gps_observations: int
    successful_trajectories: int
    failed_trajectories: int
    champion_model: str
    macro_metrics_across_trajectories: AggregateMacroMetrics
    pooled_metrics_across_observations: AggregatePooledMetrics


class TeammateMetricRecord(BaseModel):
    implementation_name: str = Field(..., description="Unique name/identifier of teammate or baseline system")
    trajectory_id: str = Field(..., description="Evaluator trajectory ID (e.g. '00000015')")
    accuracy: float = Field(..., ge=0.0, le=100.0, description="Point-wise map matching accuracy (0-100%)")
    precision: Optional[float] = Field(None, ge=0.0, le=100.0, description="Route precision (0-100%)")
    recall: Optional[float] = Field(None, ge=0.0, le=100.0, description="Route recall (0-100%)")
    f1_score: Optional[float] = Field(None, ge=0.0, le=100.0, description="Route F1 score (0-100%)")
    candidate_recall: Optional[float] = Field(None, ge=0.0, le=100.0, description="Candidate recall (0-100%)")
    route_continuity: Optional[float] = Field(None, ge=0.0, le=100.0, description="Topological continuity (0-100%)")
    route_agreement: Optional[float] = Field(None, ge=0.0, le=100.0, description="Route agreement / Jaccard (0-100%)")
    processing_time_seconds: Optional[float] = Field(None, ge=0.0, description="Runtime in seconds")
    notes: Optional[str] = Field(None, description="Optional architecture details, hyperparameters, or comments")

    @field_validator("trajectory_id")
    @classmethod
    def validate_traj_id(cls, v: str) -> str:
        v_clean = v.strip().zfill(8)
        return v_clean


class TeammateRoutePrediction(BaseModel):
    implementation_name: str = Field(..., description="Unique name/identifier of teammate")
    trajectory_id: str = Field(..., description="Evaluator trajectory ID (e.g. '00000015')")
    predicted_route: List[int] = Field(..., description="Ordered list of directed road arc IDs traversed")
    matched_arcs: Optional[List[int]] = Field(None, description="Optional point-by-point matched arc IDs")
    processing_time_seconds: Optional[float] = Field(None, ge=0.0)
    notes: Optional[str] = None

    @field_validator("trajectory_id")
    @classmethod
    def validate_traj_id(cls, v: str) -> str:
        return v.strip().zfill(8)


class ImportComparisonPayload(BaseModel):
    records: Optional[List[TeammateMetricRecord]] = None
    predictions: Optional[List[TeammateRoutePrediction]] = None


class ImportComparisonResponse(BaseModel):
    status: str
    imported_count: int
    implementations: List[str]
    validation_errors: List[str]
    message: str


class ComparisonEntry(BaseModel):
    implementation_name: str
    trajectory_id: str
    is_pathsnap: bool = False
    accuracy: float
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    candidate_recall: Optional[float] = None
    route_continuity: Optional[float] = None
    route_agreement: Optional[float] = None
    processing_time_seconds: Optional[float] = None
    notes: Optional[str] = None
    has_route_geometry: bool = False
    route_polyline_wgs84: Optional[List[List[float]]] = None


class ComparisonSummaryResponse(BaseModel):
    total_evaluator_trajectories: int
    implementations_compared: List[str]
    best_overall_implementation: str
    best_overall_f1: float
    pathsnap_macro_accuracy: float
    pathsnap_macro_f1: float
    pathsnap_avg_time_sec: float
    entries: List[ComparisonEntry]
