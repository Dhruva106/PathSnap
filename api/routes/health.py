"""
health.py - Health check and root status endpoints
"""

from datetime import datetime
from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse
from api.schemas.map_match import HealthResponse
from api.services.map_matching_service import MapMatchingService, get_map_matching_service

router = APIRouter(tags=["Health & Status"])


@router.get(
    "/",
    summary="PathSnap API Root",
    description="Returns application metadata, version, and operational status, or redirects browsers to the UI."
)
def get_root(request: Request):
    accept_header = request.headers.get("accept", "")
    if "text/html" in accept_header:
        return RedirectResponse(url="/ui/")

    return {
        "application": "PathSnap",
        "description": "ML-Powered Vehicular GPS Map Matching & Route Reconstruction System",
        "version": "1.0.0",
        "status": "online",
        "ui_url": "/ui/",
        "docs_url": "/docs",
        "endpoints": {
            "health": "/health",
            "model_info": "/model-info",
            "map_match": "/map-match",
            "evaluate": "/evaluate",
            "trajectories": "/trajectories",
        }
    }


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System Health & Readiness Probe",
    description="Verifies that the ML model is loaded and the map matching pipeline is operational."
)
def get_health(service: MapMatchingService = Depends(get_map_matching_service)):
    is_loaded = service.champion_model is not None
    model_name = service.metadata.get("model_name", "Baseline XGBoost")
    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        champion_model=model_name,
        artifacts_available=True,
        service_operational=is_loaded,
        timestamp=datetime.now().isoformat(),
    )
