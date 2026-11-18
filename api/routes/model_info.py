"""
model_info.py - Model metadata, feature schema, and model comparison endpoints
"""

import json
from fastapi import APIRouter, Depends
from api.schemas.map_match import ModelInfoResponse
from api.services.map_matching_service import MapMatchingService, get_map_matching_service
from src.model_utils import get_available_models, get_feature_metadata

router = APIRouter(tags=["Model Metadata"])


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Get Champion Model Information",
    description="Returns metadata, architecture, hyperparameters, and validation performance of the active champion model."
)
def get_model_info(service: MapMatchingService = Depends(get_map_matching_service)):
    meta = service.metadata
    val_metrics = meta.get("validation_metrics", {})
    clean_metrics = {k: float(v) for k, v in val_metrics.items()}

    return ModelInfoResponse(
        model_name=meta.get("model_name", "Baseline XGBoost"),
        model_type=meta.get("model_type", "Extreme Gradient Boosted Decision Trees (XGBoost)"),
        version="1.0.0",
        status=meta.get("status", "champion"),
        selected_features=meta.get("selected_features", []),
        validation_metrics=clean_metrics,
        training_instances=int(meta.get("num_training_instances", 1280935)),
        validation_instances=int(meta.get("num_validation_instances", 482577)),
    )


@router.get(
    "/features",
    summary="Get Feature Schema & Definitions",
    description="Returns full schema, definitions, and importance scores for all selected features."
)
def get_features():
    try:
        return get_feature_metadata()
    except Exception as e:
        return {"error": str(e)}


@router.get(
    "/models",
    summary="List All Available Model Artifacts",
    description="Lists all 5 baseline models, the optimized model, and champion model."
)
def list_models():
    return get_available_models()
