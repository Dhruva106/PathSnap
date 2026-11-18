# PathSnap — Final Comprehensive Implementation Audit

**Document:** PathSnap Pre-Integration & System Architecture Audit  
**Date:** November 2026  
**Auditor:** Senior Full-Stack ML/GIS Systems Engineer  
**Workspace:** `~/Desktop/PathSnap`  
**Champion Model:** Baseline XGBoost Classifier + Hidden Markov Model Viterbi Decoder  

---

## 1. Audit Objective & Scope

This audit independently reviews all source files, model binaries, backend services, frontend dashboard code, test suites, and empirical benchmarks in `~/Desktop/PathSnap` to establish the exact baseline before completing full-stack integration, implementing `POST /evaluate`, deploying the live server, and generating the final technical deliverables.

---

## 2. Component-by-Component Audit Findings

### 2.1 Machine Learning Pipeline
- **Status:** **VERIFIED COMPLETE**
- **Data Loader (`src/data_loader.py`):** Loads `.track`, `.nodes`, `.arcs`, `.route`. Verified clean shape checks and data structures.
- **Preprocessing (`src/preprocessing.py`):** Converts WGS84 coordinates to metric Equirectangular tangent plane centered at trajectory midpoint. Calculates finite-difference backward kinematics (speed, heading).
- **Candidate Generation (`src/candidate_generation.py`):** Spatial bounding box index with exact orthogonal point-to-segment Euclidean distance threshold $r \le 50.0\text{ m}$ (max 25 candidates/point). Top-3 nearest global fallback for isolated points. Candidate recall: $97.79\%$ overall, $99.84\%$ valid.
- **Feature Engineering (`src/feature_engineering.py`):** Computes 14 physical/kinematic features without data leakage. Top discriminatory signals: `heading_cos` (RF Gini 0.2361) and `relative_dist` (0.1909).
- **Model Training & Selection (`src/train.py`, `src/evaluate.py`):** Evaluated 5 baseline models. Baseline XGBoost achieved highest validation $F_1 = 78.83\%$, ROC-AUC $= 97.90\%$, PR-AUC $= 86.93\%$. Optuna Bayesian optimization (50 trials with TPESampler) reached max $F_1 = 78.78\% \le 78.83\%$. Baseline was strictly retained as champion.

### 2.2 Hidden Markov Model & Route Reconstruction
- **Status:** **VERIFIED COMPLETE**
- **Decoder (`src/hmm_decoder.py`):** Implements Newson-Krumm exponential transition model ($\beta = 3.0\text{ m}$, disconnection penalty $\Lambda = -20.0$), XGBoost log emission weighting ($w_{emit} = 1.2$, $\sigma_z = 10.0\text{ m}$), and Dijkstra caching with bounded distance cutoffs on `networkx.DiGraph`.
- **Route Reconstruction (`src/route_reconstruction.py`):** Shortest-path graph interpolation between non-adjacent matched road segments to form continuous directed road sequences with WGS84 centerline coordinates.
- **Validation Results:** Point-wise road accuracy: $83.00\%$ (independent ML) $\rightarrow 88.97\%$ (HMM Viterbi), a $+5.97\%$ mean gain (+18.37% on trajectory `00000084`). Mean connectivity: $98.99\%$. Only 8 disconnected jumps across 54,082 validation points.

### 2.3 Model Artifacts & Management
- **Status:** **VERIFIED COMPLETE**
- **Model Binaries (`models/`):** Champion model saved in `models/champion/best_model.joblib` (330 KB). Baseline models (`logistic_regression`, `linear_svm`, `random_forest`, `xgboost`, `catboost`) and Optuna-tuned model saved in `models/baseline/` and `models/optimized/`.
- **Schemas & Metadata (`models/metadata/`):** `feature_schema.json` (strict 14-feature schema), `preprocessing_config.json`, `hmm_config.json`, `training_config.json`. Verified clean zero-retraining model loading via `src/model_utils.py`.

### 2.4 FastAPI Backend Service
- **Status:** **PARTIAL / NEEDS EXTENSION**
- **Current Endpoints:**
  - `GET /`: API root metadata (with browser redirection).
  - `GET /health`: Health probe.
  - `GET /model-info`, `GET /features`, `GET /models`: Model and feature schemas.
  - `GET /trajectories`, `GET /trajectories/{id}`: Trajectory indexing and preview.
  - `POST /map-match`: End-to-end map matching and route reconstruction.
- **Identified Gap / Missing Requirement:**
  - **Missing `POST /evaluate` endpoint:** The official evaluator workflow requires evaluation to be a separate server-side call where ground truth is accessed only after map matching has occurred. Need to implement `POST /evaluate`, taking `trajectory_id`, `matched_arc_ids`, and `reconstructed_route_arcs`, and returning:
    - Point-wise accuracy (%)
    - Candidate recall (%)
    - Precision, Recall, $F_1$-score (%)
    - Route Jaccard overlap (%)
    - Topological connectivity (%)
    - Disconnected jumps & intermediate arcs filled
    - Ground truth route WGS84 coordinates for post-evaluation map visualization.

### 2.5 Evaluator Frontend Dashboard
- **Status:** **PARTIAL / NEEDS ENHANCEMENT**
- **Current State:** Basic Leaflet map, dropdown selector, radius sliders, and run button.
- **Identified Gaps / Missing Requirements:**
  1. **Missing "Evaluate" Workflow:** No button or flow for the evaluator to request ground-truth evaluation after map matching.
  2. **Missing Ground-Truth Route Layer:** Ground truth was not visualized on the map after evaluation.
  3. **Missing Start & End Markers:** No distinct visual markers indicating trajectory origin and destination.
  4. **Limited Point Inspector:** Point-by-point table was not interactively linked to the map (clicking a row did not zoom/highlight the point and its candidates on the map).
  5. **Configurable API URL for Two-Laptop Setup:** API calls were hardcoded to relative paths; needed a configurable settings modal allowing Laptop 2 to point to Laptop 1's LAN IP (`http://192.168.X.X:8000`).
  6. **Design Polish:** Needs full three-panel layout (Left: Controls & Selection, Center: Full-height interactive map, Right: Evaluation & Metrics Summary) with professional styling suitable for faculty evaluation.

### 2.6 Server Daemon & Availability
- **Status:** **DEFECT / NEEDS CORRECTION**
- **Issue:** Earlier, the backend server process was not launched as a persistent daemon process. When previous turns completed, uvicorn terminated, causing the website to not launch when visited in a browser (`lsof -i :8000` was empty).
- **Correction:** Must start the FastAPI server as a persistent background daemon on `0.0.0.0:8000` and verify HTTP connectivity.

### 2.7 Automated Testing
- **Status:** **VERIFIED (Needs update for `/evaluate`)**
- 21 unit and integration tests passing in `tests/test_unit.py` and `tests/test_integration.py`. Must add unit and integration tests for `POST /evaluate`.

---

## 3. Corrective Action Plan

1. **Backend Extension:**
   - Define `EvaluateRequest` and `EvaluateResponse` schemas in `api/schemas/map_match.py`.
   - Implement `evaluate_trajectory()` in `api/services/map_matching_service.py`.
   - Add `POST /evaluate` route in `api/routes/map_match.py`.
2. **Frontend Redesign & Implementation:**
   - Enhance `frontend/index.html`, `frontend/css/style.css`, `frontend/js/app.js`, `frontend/js/map.js`.
   - Add professional three-panel layout: Control Sidebar, Main Map Viewport, and Dedicated Evaluation Dashboard.
   - Implement "Evaluate" button, loading state, and evaluation results card.
   - Add Start (green flag) and End (red flag) map markers.
   - Add Ground Truth route layer (blue/purple dashed) that appears ONLY after evaluation.
   - Interactive Point Inspector: clicking a row centers the map on the point and renders candidate snapping tethers.
   - Two-Laptop LAN Config: Add Backend URL Settings modal in top bar with localStorage persistence.
3. **Static File Serving:**
   - Mount static frontend files directly in `api/main.py` so visiting `http://localhost:8000/` or `http://0.0.0.0:8000/` immediately renders the dashboard.
4. **Test Suite Expansion:**
   - Add tests for `POST /evaluate` and verify that all 22+ tests pass.
5. **Persistent Server Startup:**
   - Launch Uvicorn daemon on `0.0.0.0:8000` and verify live HTTP response.
6. **Documentation:**
   - Author comprehensive 30-section technical report in `reports/PathSnap_Final_Technical_Report.md`.
   - Update `README.md` with exact two-laptop LAN deployment instructions.
