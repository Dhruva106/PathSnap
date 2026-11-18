"""
comparison.py - API Endpoints for Official Evaluator Dataset and Teammate Comparison Framework
"""

import os
import io
import json
import csv
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Response, status
from fastapi.responses import PlainTextResponse, JSONResponse

from api.schemas.evaluator_comparison import (
    EvaluatorTrajectorySummary,
    TrajectoryEvaluationResult,
    AggregateEvaluationResult,
    TeammateMetricRecord,
    TeammateRoutePrediction,
    ImportComparisonPayload,
    ImportComparisonResponse,
    ComparisonSummaryResponse
)
from api.services.comparison_service import ComparisonService, get_comparison_service

router = APIRouter(tags=["Evaluator & Teammate Comparison"])


# ==========================================
# 1. EVALUATOR DATASET ENDPOINTS
# ==========================================

@router.get(
    "/evaluator/trajectories",
    response_model=List[EvaluatorTrajectorySummary],
    summary="List Official Evaluator Trajectories",
    description="Returns metadata, observation counts, road network sizes, and bounding boxes for the 10 official evaluator trajectories."
)
def get_evaluator_trajectories(
    service: ComparisonService = Depends(get_comparison_service)
) -> List[EvaluatorTrajectorySummary]:
    return service.get_evaluator_trajectories()


@router.get(
    "/evaluator/results",
    response_model=List[TrajectoryEvaluationResult],
    summary="Get PathSnap Official Evaluator Results",
    description="Returns individual map-matching and route reconstruction metrics for each of the 10 official evaluator trajectories."
)
def get_evaluator_results(
    service: ComparisonService = Depends(get_comparison_service)
) -> List[TrajectoryEvaluationResult]:
    results = service.get_pathsnap_evaluator_results()
    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluator results not yet generated. Run scripts/evaluate_evaluator_test.py first."
        )
    return results


@router.get(
    "/evaluator/results/{trajectory_id}",
    summary="Get Detailed Results for an Evaluator Trajectory",
    description="Returns full point predictions, reconstructed route, and ground truth route polyline coordinates for map visualization."
)
def get_evaluator_trajectory_result(
    trajectory_id: str,
    service: ComparisonService = Depends(get_comparison_service)
) -> Dict[str, Any]:
    tid = trajectory_id.strip().zfill(8)
    json_path = f"results/evaluator/trajectory_results/{tid}.json"
    if not os.path.exists(json_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluator trajectory '{tid}' result not found."
        )
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


@router.get(
    "/evaluator/aggregate",
    response_model=AggregateEvaluationResult,
    summary="Get Aggregate Evaluator Benchmark Metrics",
    description="Returns macro averages across the 10 trajectories and pooled metrics across all 19,674 GPS observations."
)
def get_evaluator_aggregate(
    service: ComparisonService = Depends(get_comparison_service)
) -> AggregateEvaluationResult:
    agg = service.get_pathsnap_aggregate_results()
    if not agg:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Aggregate evaluator results not found."
        )
    return agg


# ==========================================
# 2. TEAMMATE COMPARISON ENDPOINTS
# ==========================================

@router.get(
    "/comparison/implementations",
    summary="List All Implementations for Comparison",
    description="Returns list of available systems (PathSnap champion + all imported teammate models)."
)
def list_comparison_implementations(
    service: ComparisonService = Depends(get_comparison_service)
) -> Dict[str, Any]:
    impls = service.get_available_implementations()
    return {
        "count": len(impls),
        "implementations": impls
    }


@router.post(
    "/comparison/import",
    response_model=ImportComparisonResponse,
    summary="Import Teammate Evaluation Metrics (JSON or CSV)",
    description="Uploads and validates evaluation metrics from teammate implementations. Enforces strict schema and range checks."
)
def import_teammate_metrics(
    payload: ImportComparisonPayload,
    service: ComparisonService = Depends(get_comparison_service)
) -> ImportComparisonResponse:
    if not payload.records and not payload.predictions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payload must contain either 'records' (metric rows) or 'predictions' (route sequences)."
        )

    all_errors = []
    total_imported = 0
    added_impls = set()

    # 1. Process metric records
    if payload.records:
        records_dict = [r.model_dump() for r in payload.records]
        count, impls, errs = service.import_metrics_records(records_dict)
        total_imported += count
        added_impls.update(impls)
        all_errors.extend(errs)

    # 2. Process route predictions
    if payload.predictions:
        for pred in payload.predictions:
            try:
                service.evaluate_teammate_route_prediction(pred)
                total_imported += 1
                added_impls.add(pred.implementation_name)
            except Exception as e:
                all_errors.append(f"Prediction for '{pred.implementation_name}' on '{pred.trajectory_id}' failed: {str(e)}")

    return ImportComparisonResponse(
        status="success" if total_imported > 0 else "error",
        imported_count=total_imported,
        implementations=list(added_impls),
        validation_errors=all_errors,
        message=f"Successfully imported {total_imported} records across {len(added_impls)} implementations."
    )


@router.post(
    "/comparison/import-file",
    response_model=ImportComparisonResponse,
    summary="Upload Teammate Results File (.csv or .json)",
    description="Accepts multipart CSV or JSON file upload for teammates to submit benchmarks."
)
async def import_teammate_file(
    file: UploadFile = File(...),
    service: ComparisonService = Depends(get_comparison_service)
) -> ImportComparisonResponse:
    filename = file.filename.lower()
    content = await file.read()
    text = content.decode("utf-8", errors="replace")

    records_to_import = []

    if filename.endswith(".json"):
        try:
            data = json.loads(text)
            if isinstance(data, list):
                records_to_import = data
            elif isinstance(data, dict):
                if "records" in data and isinstance(data["records"], list):
                    records_to_import = data["records"]
                elif "predictions" in data and isinstance(data["predictions"], list):
                    # Handle prediction objects
                    pred_objs = [TeammateRoutePrediction(**p) for p in data["predictions"]]
                    return import_teammate_metrics(ImportComparisonPayload(predictions=pred_objs), service)
                else:
                    records_to_import = [data]
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON file format: {str(e)}"
            )

    elif filename.endswith(".csv") or filename.endswith(".txt"):
        try:
            reader = csv.DictReader(io.StringIO(text))
            for row in reader:
                records_to_import.append(row)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to parse CSV file: {str(e)}"
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a .csv or .json document."
        )

    count, impls, errs = service.import_metrics_records(records_to_import)
    return ImportComparisonResponse(
        status="success" if count > 0 else "error",
        imported_count=count,
        implementations=impls,
        validation_errors=errs,
        message=f"Uploaded {file.filename}: successfully imported {count} records across {len(impls)} implementations."
    )


@router.get(
    "/comparison/results",
    response_model=ComparisonSummaryResponse,
    summary="Get Filterable Implementation Comparison Matrix",
    description="Returns cross-implementation comparison table with best performer highlights and macro metrics."
)
def get_comparison_results(
    implementations: Optional[str] = Query(None, description="Comma-separated implementation names to include"),
    trajectories: Optional[str] = Query(None, description="Comma-separated trajectory IDs to filter"),
    service: ComparisonService = Depends(get_comparison_service)
) -> ComparisonSummaryResponse:
    impl_list = [i.strip() for i in implementations.split(",") if i.strip()] if implementations else None
    traj_list = [t.strip().zfill(8) for t in trajectories.split(",") if t.strip()] if trajectories else None

    return service.get_comparison_summary(
        implementations=impl_list,
        trajectory_ids=traj_list
    )


@router.get(
    "/comparison/results/{trajectory_id}",
    summary="Get All Implementation Routes for a Single Trajectory",
    description="Returns route polylines and metrics for all systems that evaluated a specific trajectory (for map overlay comparison)."
)
def get_trajectory_comparison_routes(
    trajectory_id: str,
    service: ComparisonService = Depends(get_comparison_service)
) -> Dict[str, Any]:
    tid = trajectory_id.strip().zfill(8)
    summary = service.get_comparison_summary(trajectory_ids=[tid])

    # Find ground truth route for map reference
    gt_polyline = []
    json_path = f"results/evaluator/trajectory_results/{tid}.json"
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                d = json.load(f)
                gt_polyline = d.get("reconstructed_route", {}).get("route_polyline_wgs84", [])
        except Exception:
            pass

    return {
        "trajectory_id": tid,
        "implementations_count": len(summary.entries),
        "ground_truth_polyline_wgs84": gt_polyline,
        "routes": [
            {
                "implementation_name": e.implementation_name,
                "is_pathsnap": e.is_pathsnap,
                "accuracy": e.accuracy,
                "f1_score": e.f1_score,
                "route_continuity": e.route_continuity,
                "has_route_geometry": e.has_route_geometry,
                "route_polyline_wgs84": e.route_polyline_wgs84
            }
            for e in summary.entries
        ]
    }


@router.get(
    "/comparison/template",
    summary="Download Submission Template (CSV or JSON)",
    description="Provides standard submission template for teammates to format their evaluation metrics."
)
def get_submission_template(
    format: str = Query("csv", pattern="^(csv|json)$", description="Template format: 'csv' or 'json'"),
    service: ComparisonService = Depends(get_comparison_service)
):
    if format == "csv":
        csv_data = service.generate_csv_template()
        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=pathsnap_teammate_benchmark_template.csv"}
        )
    else:
        json_data = service.generate_json_template()
        return JSONResponse(
            content=json_data,
            headers={"Content-Disposition": "attachment; filename=pathsnap_teammate_benchmark_template.json"}
        )


@router.post(
    "/comparison/evaluate",
    summary="Evaluate Teammate Route Arc Sequence Server-Side",
    description="Submits a teammate's predicted route arc sequence and evaluates it against official ground truth using PathSnap's exact evaluation logic."
)
def evaluate_teammate_prediction(
    prediction: TeammateRoutePrediction,
    service: ComparisonService = Depends(get_comparison_service)
) -> Dict[str, Any]:
    try:
        res = service.evaluate_teammate_route_prediction(prediction)
        return {
            "status": "success",
            "message": f"Successfully evaluated '{prediction.implementation_name}' on trajectory '{prediction.trajectory_id}'",
            "metrics": res
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Evaluation of route prediction failed: {str(e)}"
        )
