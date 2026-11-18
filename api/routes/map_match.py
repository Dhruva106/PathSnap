"""
map_match.py - Core API Endpoints for Map Matching and Trajectory Exploration
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.schemas.map_match import MapMatchRequest, MapMatchResponse, EvaluateRequest, EvaluateResponse
from api.services.map_matching_service import MapMatchingService, get_map_matching_service
from src.data_loader import load_single_trajectory

router = APIRouter(tags=["Map Matching"])

# Load split metadata once
TRAIN_IDS = set()
VAL_IDS = set()
SPLIT_REPORT_PATH = "results/trajectory_split_report.json"
if os.path.exists(SPLIT_REPORT_PATH):
    try:
        with open(SPLIT_REPORT_PATH, "r") as f:
            split_data = json.load(f)
            TRAIN_IDS = set(split_data.get("train_trajectory_ids", []))
            VAL_IDS = set(split_data.get("val_trajectory_ids", []))
    except Exception:
        pass


@router.post(
    "/map-match",
    response_model=MapMatchResponse,
    summary="Match GPS Trajectory to Road Network",
    description=(
        "Executes end-to-end map matching: project coordinates into metric tangent plane, "
        "generates candidate road arcs within spatial threshold, evaluates candidate probabilities "
        "using champion Baseline XGBoost model, decodes optimal sequence via HMM Viterbi (or top-1 ML argmax), "
        "and reconstructs continuous topological route along road graph."
    )
)
def post_map_match(
    request: MapMatchRequest,
    service: MapMatchingService = Depends(get_map_matching_service)
) -> MapMatchResponse:
    try:
        response = service.map_match(request)
        return response
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}"
        )


@router.post(
    "/evaluate",
    response_model=EvaluateResponse,
    summary="Evaluate Predicted Road Route Against Ground Truth",
    description=(
        "Evaluates the client-submitted matched road arc sequence against the internal reference "
        "ground truth (.route). Evaluates point-wise accuracy, candidate recall, route set precision, "
        "recall, F1-score, Jaccard overlap, and topological continuity. Ground truth is accessed strictly "
        "during this post-prediction evaluation phase, ensuring zero data leakage."
    )
)
def post_evaluate(
    request: EvaluateRequest,
    service: MapMatchingService = Depends(get_map_matching_service)
) -> EvaluateResponse:
    try:
        response = service.evaluate_trajectory(request)
        return response
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}"
        )


@router.get(
    "/trajectories",
    summary="List Available Trajectories",
    description="Returns an indexed list of all trajectories in the training, validation, and student test splits."
)
def list_trajectories(
    split: Optional[str] = Query(None, description="Filter by split: 'train', 'val', or 'student_test'")
) -> Dict[str, Any]:
    trajectories = []

    # Check train / val folders
    train_dir = "data/raw/train"
    if os.path.exists(train_dir):
        for entry in sorted(os.listdir(train_dir)):
            full_path = os.path.join(train_dir, entry)
            if os.path.isdir(full_path):
                if entry in VAL_IDS:
                    traj_split = "val"
                elif entry in TRAIN_IDS:
                    traj_split = "train"
                else:
                    traj_split = "train"

                if split and traj_split != split:
                    continue

                trajectories.append({
                    "trajectory_id": entry,
                    "split": traj_split,
                    "location": "data/raw/train",
                })

    # Check student_test folder
    test_dir = "data/raw/student_test"
    if os.path.exists(test_dir):
        for entry in sorted(os.listdir(test_dir)):
            full_path = os.path.join(test_dir, entry)
            if os.path.isdir(full_path):
                traj_split = "student_test"
                if split and traj_split != split:
                    continue

                trajectories.append({
                    "trajectory_id": entry,
                    "split": traj_split,
                    "location": "data/raw/student_test",
                })

    # Check evaluator_test folder
    eval_dir = "data/raw/evaluator_test"
    if os.path.exists(eval_dir):
        for entry in sorted(os.listdir(eval_dir)):
            full_path = os.path.join(eval_dir, entry)
            if os.path.isdir(full_path):
                traj_split = "evaluator_test"
                if split and traj_split != split:
                    continue

                trajectories.append({
                    "trajectory_id": entry,
                    "split": traj_split,
                    "location": "data/raw/evaluator_test",
                })

    return {
        "total_count": len(trajectories),
        "trajectories": trajectories
    }


@router.get(
    "/trajectories/{trajectory_id}",
    summary="Get Trajectory Summary and Preview Points",
    description="Returns detailed metadata, GPS point count, bounding box, and sample coordinates for a specific trajectory."
)
def get_trajectory_detail(trajectory_id: str) -> Dict[str, Any]:
    train_path = os.path.join("data/raw/train", trajectory_id)
    test_path = os.path.join("data/raw/student_test", trajectory_id)
    eval_path = os.path.join("data/raw/evaluator_test", trajectory_id)

    if os.path.exists(train_path):
        folder = train_path
        split = "val" if trajectory_id in VAL_IDS else "train"
    elif os.path.exists(test_path):
        folder = test_path
        split = "student_test"
    elif os.path.exists(eval_path):
        folder = eval_path
        split = "evaluator_test"
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trajectory '{trajectory_id}' not found."
        )

    try:
        raw = load_single_trajectory(folder, traj_id=trajectory_id)
        lons = raw.track[:, 0]
        lats = raw.track[:, 1]
        times = raw.track[:, 2]

        duration_sec = float(times[-1] - times[0]) if len(times) > 1 else 0.0

        # Sample up to 50 preview points for quick client inspection
        stride = max(1, len(raw.track) // 50)
        preview_points = [
            {"longitude": float(row[0]), "latitude": float(row[1]), "timestamp": float(row[2])}
            for row in raw.track[::stride]
        ]

        return {
            "trajectory_id": trajectory_id,
            "split": split,
            "point_count": len(raw.track),
            "duration_seconds": round(duration_sec, 2),
            "start_time": float(times[0]),
            "end_time": float(times[-1]),
            "bounding_box": {
                "min_latitude": float(np.min(lats)),
                "max_latitude": float(np.max(lats)),
                "min_longitude": float(np.min(lons)),
                "max_longitude": float(np.max(lons)),
            },
            "preview_points": preview_points
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read trajectory '{trajectory_id}': {str(e)}"
        )
