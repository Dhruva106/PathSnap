# PathSnap: Final Technical & Engineering Report

**Project Title:** PathSnap: ML-Powered Vehicular GPS Map Matching and Continuous Road Route Reconstruction  
**Workspace:** `~/Desktop/PathSnap`  
**Champion Model:** Baseline Extreme Gradient Boosting (XGBoost) Classifier + Hidden Markov Model (HMM) Viterbi Sequence Decoder  
**Author:** PathSnap Engineering Team  
**Date:** November 2026  

---

## 1. Abstract

Vehicular GPS trajectories collected in dense metropolitan settings suffer from multipath reflection, satellite occlusion, and sparse spatial-temporal sampling rates ($\approx 35\text{ m}$ average point separation). Naive geometric snapping to the nearest road arc produces disconnected routes, direction reversals, and severe hopping across parallel service roads. **PathSnap** resolves this challenge through a production-grade, two-stage hybrid machine learning and probabilistic graph architecture. First, a 14-feature Extreme Gradient Boosted decision tree (Baseline XGBoost) computes candidate road match probabilities, achieving a validation $F_1$-score of **$78.83\%$**, ROC-AUC of **$97.90\%$**, and PR-AUC of **$86.93\%$**. Second, a Hidden Markov Model (HMM) implementing the Newson-Krumm exponential transition model over a directed road network ($G$) executes global Viterbi sequence decoding with localized Dijkstra query caching. 

On 20 validation trajectories ($54,082$ observations), PathSnap improves point-wise road matching accuracy from **$83.00\%$** (independent ML) to **$88.97\%$** (**$+5.97\%$** mean gain, with outlier improvements up to **$+18.37\%$**) while achieving **$98.99\%$** mean topological route continuity. On 10 completely unseen **Student Test** holdout trajectories ($19,160$ observations), PathSnap achieves **$89.18\%$** point-wise accuracy (**$+5.51\%$** gain over independent ML), **$99.16\%$** topological connectivity, **$90.94\%$** route Jaccard agreement, and **$0$ disconnected jumps**. The entire system is integrated into an asynchronous FastAPI backend and a dark-mode Leaflet geospatial dashboard supporting two-laptop network deployment.

---

## 2. Introduction

Modern Intelligent Transportation Systems (ITS), autonomous fleet dispatchers, and traffic management platforms rely on accurate reconstruction of vehicular paths from raw GPS sensor telemetry. However, raw telematics coordinates do not inherently align with road network graphs. Map matching is the process of mapping a raw coordinate sequence onto the true sequence of road segments in a digital transport network. PathSnap provides a full-stack, mathematically rigorous solution that combines domain-engineered kinematic features, gradient-boosted decision trees, and network-topology-constrained sequence inference.

---

## 3. Problem Statement & Mathematical Formulation

Let a road network be represented as a directed graph $G = (V, E)$, where $V$ is a set of road junction vertices with geodetic coordinates $(lon_v, lat_v)$ and $E$ is a set of directed road arcs $e = (u, v)$ with metric length $l(e)$. A vehicular trajectory consists of a discrete time series of noisy GPS observations:
$$T = \{(p_t, \tau_t)\}_{t=1}^N, \quad p_t = (\lambda_t, \phi_t) \in \mathbb{R}^2, \quad \tau_1 < \tau_2 < \dots < \tau_N$$
where $\lambda_t$ is longitude, $\phi_t$ is latitude, and $\tau_t$ is the epoch timestamp.

The objective is to deduce the true traversed sequence of directed road arcs:
$$R^* = (e_1^*, e_2^*, \dots, e_N^*), \quad e_t^* \in E$$
such that $R^*$ represents a feasible, connected path in $G$.

Pointwise classification independently chooses $\hat{e}_t = \arg\max_{c_{t,i}} P(y_{t,i}=1 \mid \mathbf{x}_{t,i})$. Sequence-level HMM inference globally maximizes:
$$R^* = \arg\max_{e_1, \dots, e_N} \left[ \ln P(e_1) + \sum_{t=1}^N \ln P(p_t \mid e_t) + \sum_{t=2}^N \ln P(e_t \mid e_{t-1}) \right]$$

---

## 4. Dataset Architecture & Split Integrity

The dataset comprises 90 metropolitan vehicular trajectories:
- `.track`: Time-indexed GPS coordinates `[longitude, latitude, timestamp]` (WGS84).
- `.nodes`: Road junction vertices `[longitude, latitude]`.
- `.arcs`: Directed edges `[from_node_id, to_node_id]` forming directed graph $G$.
- `.route`: Ground-truth road arc traversal sequence (used strictly for offline training label generation and post-inference evaluation).

### Strict Trajectory-Level Splitting
Splitting was executed at the trajectory level using a deterministic seed (`seed=42`) to prevent spatial-temporal leakage:
- **Training Set (60 trajectories):** 1,280,935 candidate observation pairs ($10.42\%$ positive class prevalence).
- **Validation Set (20 trajectories):** 482,577 candidate observation pairs ($11.04\%$ positive class prevalence).
- **Student Test Set (10 trajectories):** 19,160 GPS observations (Held-out benchmark; never seen during feature selection, model training, or hyperparameter optimization).

---

## 5. System Architecture

```
Raw GPS Stream (.track)
   │
   ▼
[1] Equirectangular Local Tangent Plane Projection (src/preprocessing.py)
   │
   ▼
[2] Spatial Segment Indexing & Candidate Generation (src/candidate_generation.py)
   │
   ▼
[3] 14 Physical & Kinematic Feature Extraction (src/feature_engineering.py)
   │
   ▼
[4] Champion Baseline XGBoost Probability Scoring (src/inference.py)
   │
   ▼
[5] Hidden Markov Model Viterbi Sequence Decoding (src/hmm_decoder.py)
   │
   ▼
[6] Shortest-Path Continuous Route Reconstruction (src/route_reconstruction.py)
   │
   ▼
[7] FastAPI Asynchronous REST Service (api/main.py)
   │
   ▼
[8] Evaluator Geospatial Web Dashboard (frontend/index.html)
```

---

## 6. Data Preprocessing

Geodetic coordinates $(\lambda, \phi)$ in degrees are projected into an Equirectangular Local Tangent Plane centered at the trajectory midpoint $(\phi_{ref}, \lambda_{ref})$:
$$x = R \cdot (\lambda - \lambda_{ref}) \cdot \cos(\phi_{ref}) \cdot \frac{\pi}{180}$$
$$y = R \cdot (\phi - \phi_{ref}) \cdot \frac{\pi}{180}$$
where $R = 6,371,000\text{ m}$. Over typical urban spans ($< 25\text{ km}$), distortion is $< 0.01\%$.

Vehicular speed and heading are computed using backward kinematic finite differences:
$$\Delta t_k = \tau_k - \tau_{k-1}, \quad \mathbf{v}_k = \frac{\mathbf{p}_k - \mathbf{p}_{k-1}}{\Delta t_k}, \quad v_k = \|\mathbf{v}_k\|_2, \quad \theta_k = \text{atan2}(v_{y,k}, v_{x,k})$$

---

## 7. Candidate Generation

Road segments are indexed into a spatial bounding-box index. For each GPS point $\mathbf{p}_t$, road arcs within radius $r_{max} = 50.0\text{ m}$ are queried. For candidate arc $e = (u, v)$ with endpoints $\mathbf{u}, \mathbf{v}$:
$$\mathbf{w} = \mathbf{v} - \mathbf{u}, \quad t_{proj} = \text{clip}\left(\frac{(\mathbf{p}_t - \mathbf{u}) \cdot \mathbf{w}}{\max(\|\mathbf{w}\|_2^2, 10^{-12})}, 0.0, 1.0\right)$$
$$\mathbf{p}_{proj} = \mathbf{u} + t_{proj} \mathbf{w}, \quad d_{\perp} = \|\mathbf{p}_t - \mathbf{p}_{proj}\|_2$$
Candidates with $d_{\perp} \le 50.0\text{ m}$ are retained (capped at $K=25$). If 0 candidates fall within $50\text{ m}$, the 3 nearest segments globally are retrieved as fallback.

Candidate recall: **$97.79\%$** overall, **$99.84\%$** for points with nearby road arcs.

---

## 8. Feature Engineering

14 domain-specific physical, kinematic, and topological features are computed per candidate:
1. `heading_cos`: $\cos(\theta_{gps} - \theta_{arc}) \in [-1.0, 1.0]$
2. `relative_dist`: $d_{\perp} / \max(\min_j d_{\perp,j}, 0.1)$
3. `dist_rank`: Discrete spatial distance rank ($1 \dots 25$)
4. `dist_to_segment`: Orthogonal distance $d_{\perp}$ in meters
5. `min_node_dist`: $\min(\|\mathbf{p} - \mathbf{u}\|_2, \|\mathbf{p} - \mathbf{v}\|_2)$
6. `proj_ratio`: Segment position $t_{proj} \in [0.0, 1.0]$
7. `dist_to_midpoint`: Distance to segment midpoint
8. `dist_to_u`: Distance to junction node $u$
9. `dist_to_v`: Distance to junction node $v$
10. `arc_length`: Metric segment length $\|\mathbf{v} - \mathbf{u}\|_2$
11. `node_asymmetry`: $\|d_u - d_v\| / (d_u + d_v)$
12. `speed`: Instantaneous GPS velocity $\|\mathbf{v}_{gps}\|_2$
13. `delta_x`: Tangent plane Easting offset
14. `delta_y`: Tangent plane Northing offset

---

## 9. Feature Selection Pipeline

Feature selection was conducted strictly on the 60 training trajectories using `LeakageFreeFeatureSelector`:
1. **Variance Thresholding ($\sigma^2 \ge 0.001$):** Eliminated constant features.
2. **Collinearity Pruning ($|r| \ge 0.85$ with Mutual Information Arbitration):** Pairwise correlation dropped `dist_sq` ($r=0.96$), `log_dist` ($r=0.94$), and `heading_diff_deg` (redundant with `heading_cos`).
3. **Random Forest Gini Ranking:** Identified `heading_cos` ($0.2361$) and `relative_dist` ($0.1909$) as top signals.

---

## 10. Five Machine Learning Models

Five architectures were trained on 1,280,935 instances and evaluated on 482,577 validation instances:

| Model | Accuracy (%) | Precision (%) | Recall (%) | $F_1$-Score (%) | ROC-AUC (%) | PR-AUC (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 91.24 | 60.59 | 63.85 | 62.18 | 90.04 | 60.15 |
| Linear SVM (Calibrated) | 91.80 | 64.21 | 58.62 | 61.29 | 89.88 | 58.91 |
| Random Forest | 95.38 | 80.44 | 76.81 | 78.58 | 97.82 | 86.55 |
| **Baseline XGBoost (Champion)** | **95.42** | **80.48** | **77.24** | **78.83** | **97.90** | **86.93** |
| CatBoost | 95.39 | 80.35 | 77.06 | 78.67 | 97.85 | 86.69 |

---

## 11. Model Comparison Analysis

Tree-based ensemble models substantially outperformed linear models (boosting $F_1$ from $\approx 62\%$ to $\approx 79\%$). Nonlinear directional alignment (`heading_cos`) and localized distance ratios (`relative_dist`) require decision tree thresholding to separate true road segments from parallel candidates.

---

## 12. Champion Model Selection Justification

**Baseline XGBoost was officially designated Champion** based on:
1. Highest validation $F_1$-score ($78.83\%$).
2. Highest ROC-AUC ($97.90\%$) and PR-AUC ($86.93\%$).
3. Fast execution latency ($< 25\text{ ms}$ for 1,500 candidates).

---

## 13. Bayesian Hyperparameter Optimization

Bayesian optimization was conducted using `optuna` with `TPESampler(seed=42)` across 50 trials:
- Parameter space: `n_estimators` [100, 300], `max_depth` [4, 10], `learning_rate` [0.01, 0.2], `subsample` [0.6, 1.0], `colsample_bytree` [0.6, 1.0], `scale_pos_weight` [1.0, 5.0].
- **Result:** Best Trial #31 achieved $F_1 = 78.78\%$.
- **Decision:** Because $78.78\% \le 78.83\%$ (baseline), the baseline XGBoost model was strictly retained. Neither Grid Search nor Random Search was used.

---

## 14. Hidden Markov Model Formulation

Observations $Z = (\mathbf{p}_1, \dots, \mathbf{p}_N)$ are associated with hidden states $C_t = \{c_{t,1}, \dots, c_{t,K_t}\}$ representing candidate road arcs.

---

## 15. Emission Model

In `src/hmm_decoder.py`:
$$\ln P(\mathbf{p}_t \mid c_{t,i}) = w_{emit} \cdot \ln\left(\max(P_{ML}(y=1 \mid \mathbf{x}_{t,i}), 10^{-6})\right) - 0.2 \cdot \frac{d_{\perp,i}^2}{2\sigma_z^2}$$
where $w_{emit} = 1.2$, $\sigma_z = 10.0\text{ m}$.

---

## 16. Transition Model

Newson-Krumm exponential difference model:
$$\ln P(c_{t,j} \mid c_{t-1,i}) = - \frac{|d_{net}(c_{t-1,i}, c_{t,j}) - d_{gps}(\mathbf{p}_{t-1}, \mathbf{p}_t)|}{\beta} + \Lambda_{disconnect}$$
where $\beta = 3.0\text{ m}$, $\Lambda_{disconnect} = -20.0$. Shortest network path $d_{net}$ is computed via Dijkstra search bounded by $\text{cutoff} = \max(2.5 \cdot d_{gps} + 50.0, 150.0)\text{ m}$ with junction node caching.

---

## 17. Viterbi Algorithm

- Initialization ($t=0$): $V_0(i) = \ln P(\mathbf{p}_0 \mid c_{0,i}) - 0.5 \cdot (\text{dist\_rank}_{0,i} - 1)$.
- Recursion ($t=1 \dots N-1$): $V_t(j) = \max_i [V_{t-1}(i) + \ln P(c_{t,j} \mid c_{t-1,i})] + \ln P(\mathbf{p}_t \mid c_{t,j})$.
- Backpointers stored for backtracking: $s_N^* = \arg\max_j V_N(j)$; $s_t^* = B_{t+1}(s_{t+1}^*)$.

---

## 18. Continuous Route Reconstruction

In `src/route_reconstruction.py`:
- Consecutive identical arcs are deduplicated.
- If $v_{t-1}^* == u_t^*$, segment is directly appended.
- If $v_{t-1}^* \ne u_t^*$, `nx.shortest_path(G, source=prev_v, target=curr_u)` bridges missing intermediate arcs.
- Centerline coordinates are converted to WGS84 for mapping.

---

## 19. Model Export & Artifact Management

Serialized in `models/`:
- `models/champion/best_model.joblib`: Baseline XGBoost binary.
- `models/metadata/feature_schema.json`: Strict 14-feature schema.
- `models/metadata/hmm_config.json`: Calibrated HMM parameters.
- `models/metadata/training_config.json`: Dataset split statistics.

---

## 20. FastAPI Backend

Implemented in `api/main.py` with 9 endpoints:
- `GET /`: Metadata & browser redirect to UI.
- `GET /health`: Readiness probe.
- `GET /model-info`, `GET /features`, `GET /models`: Model and schema endpoints.
- `GET /trajectories`, `GET /trajectories/{id}`: Trajectory indexing and preview.
- `POST /map-match`: End-to-end inference pipeline.
- `POST /evaluate`: Server-side ground truth evaluation after map matching.

---

## 21. Frontend Architecture

Built in `frontend/` using HTML5, CSS3, ES6 JavaScript, and Leaflet.js. Mounted at `/ui/`.
- Three-panel layout: Control Sidebar, Main Map Viewport, and Dedicated Evaluation Dashboard.
- Controls: Dynamic trip dropdown, quick presets, algorithm toggle (HMM vs. Top-1 ML), search radius and candidate count sliders.
- Layers: Raw GPS (red dashed), snapped projections (cyan dots), reconstructed route (green solid line), ground truth (purple dashed, post-evaluation).
- Interactive Point Inspector: Clicking any row highlights the point and its candidate projections on the map.

---

## 22. Evaluator Workflow

1. Backend starts on Laptop 1 (`0.0.0.0:8000`).
2. Frontend opens on Laptop 2.
3. Evaluator selects an unseen trip from the dropdown.
4. Evaluator clicks **Run Map Matching**; backend executes ML + HMM inference.
5. Interactive map renders the predicted continuous route.
6. Evaluator clicks **Evaluate Against Ground Truth**; backend accesses internal `.route` reference and computes accuracy metrics.
7. Map renders the reference ground-truth route polyline for visual comparison.

---

## 23. Two-Laptop Deployment

- **Backend Binding:** Uvicorn binds to `0.0.0.0:8000`.
- **Frontend Configuration:** Top-bar "API: Settings" modal allows entering Laptop 1's LAN IP (`http://192.168.X.X:8000`) with localStorage persistence.

---

## 24. Automated Testing

23 automated unit and integration tests passing in `tests/`:
```
======================== 23 passed, 1 warning in 18.53s ========================
```

---

## 25. Validation Results (20 Trajectories, 54,082 Points)

- Independent ML Point-wise Accuracy: **83.00%**
- HMM Viterbi Point-wise Accuracy: **88.97%** (**+5.97%** mean gain)
- Topological Continuity: **98.99%**
- Route Jaccard Overlap: **82.39%**
- Disconnected Jumps: 8 across all 20 trajectories.

---

## 26. Student-Test Holdout Results (10 Trajectories, 19,160 Points)

| Trajectory ID | GPS Points | Ground Truth Arcs | Indep ML Acc (%) | HMM Viterbi Acc (%) | Accuracy Gain (%) | Route Jaccard Overlap (%) | Topological Connectivity (%) | Disconnected Jumps | Total Latency (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `00000000` | 2,503 | 179 | 92.85 | 97.04 | +4.19 | 87.94 | 99.8 | 0 | 3.98 |
| `00000007` | 1,354 | 123 | 90.10 | 88.48 | -1.62 | 96.03 | 99.6 | 0 | 1.58 |
| `00000009` | 1,049 | 429 | 82.46 | 88.66 | +6.20 | 90.79 | 99.8 | 0 | 3.54 |
| `00000040` | 2,995 | 309 | 88.05 | 93.72 | +5.68 | 95.96 | 99.1 | 0 | 6.99 |
| `00000052` | 262 | 95 | 93.13 | 99.24 | +6.11 | 97.89 | 100.0 | 0 | 0.35 |
| `00000056` | 3,958 | 1,068 | 86.38 | 92.75 | +6.37 | 92.27 | 99.4 | 0 | 33.50 |
| `00000065` | 768 | 145 | 79.95 | 86.46 | +6.51 | 94.77 | 98.2 | 0 | 1.41 |
| `00000076` | 1,486 | 374 | 78.87 | 90.78 | **+11.91** | 81.32 | 99.5 | 0 | 4.92 |
| `00000077` | 3,911 | 268 | 63.62 | 70.54 | +6.93 | 79.23 | 96.6 | 0 | 9.18 |
| `00000091` | 874 | 232 | 81.24 | 84.10 | +2.86 | 93.15 | 99.5 | 0 | 1.99 |
| **Mean / Total** | **19,160** | **3,222** | **83.66%** | **89.18%** | **+5.51%** | **90.94%** | **99.16%** | **0 Jumps** | **6.74s avg** |

---

## 27. Evaluator-Test Results

The 10 student test trajectories serve as the official holdout benchmark. Evaluated dynamically via `POST /evaluate`, achieving **$89.18\%$** point-wise accuracy, **$99.16\%$** topological continuity, and **$0$ disconnected jumps**.

---

## 28. Limitations & Failure Modes

1. **2D Overpass Ambiguity:** In multi-level highway stacks, 2D coordinates map equally to upper expressways and lower frontage roads without vertical altitude telemetry.
2. **Extended Tunnel Outages:** GPS signal loss $> 500\text{ m}$ requires bridging multiple unobserved junctions via shortest path, occasionally selecting alternate arterials.

---

## 29. Future Work

1. Integrating 3D ellipsoidal elevation and road grade features.
2. Bidirectional A* routing to reduce graph traversal time.
3. Fixed-lag Viterbi windowing for real-time onboard vehicle telematics streams.

---

## 30. Conclusion

PathSnap demonstrates that combining domain-specific kinematic feature engineering, high-capacity gradient boosted classification (Baseline XGBoost), and topology-constrained Hidden Markov sequence decoding produces a robust, high-accuracy GPS map-matching and route reconstruction system. By respecting strict statistical discipline—evaluating genuine unseen test data, verifying 100% trajectory-level splits, and establishing end-to-end automated testing—PathSnap achieves **$89.18\%$** point-wise matching accuracy and **$99.16\%$** topological connectivity on unseen test trajectories, providing a production-ready solution for modern intelligent transportation systems.
