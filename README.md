PathSnap
GPS Trajectory Map Matching and Route Reconstruction using Machine
Learning

PathSnap converts noisy or sparse GPS observations into plausible routes
on a road network. It generates nearby road-segment candidates,
engineers geometric and movement-related features, scores candidates
with machine-learning models, and reconstructs a coherent route using
HMM/Viterbi-style sequence decoding.
Key Features
- Road-segment candidate generation using spatial indexing.
- Point-to-segment distance and projection-based geometric features.
- Movement and heading features derived from GPS observations.
- Machine-learning candidate scoring.
- HMM/Viterbi route reconstruction.
- FastAPI backend with interactive API documentation.
- Evaluation and model-comparison endpoints.
Machine-Learning Pipeline
The documented experiment uses 80 training trajectories, split by
trajectory into 60 training trajectories and 20 validation trajectories.

  Split          Trajectories   Candidate instances
  Training                 60             1,280,935
Validation               20               482,577

The 14 selected features are:

dist_to_segment, dist_to_u, dist_to_v, dist_to_midpoint,
arc_length, proj_ratio, speed, dist_rank, relative_dist,
delta_x, delta_y, heading_cos, min_node_dist, and
node_asymmetry.

The training code includes Logistic Regression, Linear SVM, Random
Forest, XGBoost, and CatBoost. The previously reported baseline XGBoost
validation F1-score is 0.7883. Reproduce the experiment in the
current environment before treating this as a newly verified result.

Evaluation discipline: Keep evaluator and student test trajectories
separate from model fitting, feature selection, and hyperparameter
tuning.
Repository Structure
PathSnap/
├── api/                  # FastAPI application, routes and services
├── data/                 # Raw and processed trajectory data
├── frontend/             # Frontend / evaluator interface, if included
├── models/               # Model artifacts and metadata
├── notebooks/            # Experiment notebooks
├── results/              # Evaluation metrics and reports
├── src/                  # Data processing, features, training and decoding
├── visualizations/       # Generated visualizations
├── requirements.txt
└── README.md
Setup
1. Clone the repository
git clone https://github.com/Dhruva106/PathSnap.git
cd PathSnap
2. Create and activate a virtual environment
Windows PowerShell
py -m venv .venv
.venv\Scripts\Activate.ps1
macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
If installation fails, check the Python and package versions supported
by the dependency set. Native scientific-computing packages can be
sensitive to Python-version compatibility.
Run the API
From the repository root, with the virtual environment activated:
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
Open:
- API root: http://127.0.0.1:8000/
- Swagger documentation: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc
- OpenAPI schema: http://127.0.0.1:8000/openapi.json
Use 127.0.0.1 in the browser for local access. 0.0.0.0 is a server
bind address, not normally the address to enter in the browser.
API Endpoints
  Method                  Endpoint                        Purpose
  GET                     /                             API root

  GET                     /health                       Service and model
health

  GET                     /model-info                   Active model
information and metrics

  GET                     /features                     Feature information

  GET                     /models                       Available model
information

  GET                     /trajectories                 Trajectory listing

  GET                     /evaluator/trajectories       Evaluator trajectory
listing

  GET                     /evaluator/results            Evaluation results

  GET                     /evaluator/aggregate          Aggregate evaluation
metrics

  POST                    /map-match                    Run map matching

  POST                    /evaluate                     Run evaluation

  GET                     /comparison/implementations   Comparison
implementation
information

  GET                     /comparison/results           Comparison results
POST                    /comparison/evaluate          Run implementation
comparison
For request schemas and response formats, use the live documentation at
/docs.
Model Artifacts
Model files and metadata are organized under:
- models/baseline/
- models/optimized/
- models/champion/
- models/metadata/
Serialized models can depend on the versions of Python libraries used to
create them. If a model fails to load, verify compatibility and artifact
integrity; changing the API configuration alone does not repair an
incompatible artifact.
Training
The training entry point is:
python -m src.train
To request a fresh Optuna optimization run:
python -m src.train --rerun-optuna
Caution: The training pipeline writes model artifacts, metadata, and
result files into the repository. Back up existing outputs before
running it. Confirm the configured input paths and data split first. Do
not include evaluator or student test data in training.
Reproducibility
When reproducing experiments, record:
- Python and dependency versions.
- Dataset version and trajectory split.
- Feature list and preprocessing configuration.
- Model hyperparameters and random seed.
- Validation metrics, training duration, and artifact path.
Do not compare metrics from different data splits as if they were
directly equivalent.
Project Status
The repository includes the PathSnap source code, API, model-training
pipeline, and project artifacts. The model selected by the running API
depends on the local configuration and available artifacts. Verify
/health and /model-info after startup to identify the model actually
loaded.
Contributing
1. Create a feature branch.
2. Make focused changes.
3. Run relevant tests and verify API startup.
4. Keep credentials, virtual environments, caches, and private datasets
   out of commits.
5. Open a pull request describing the change and validation performed.
License
No license is specified in this README. Add a LICENSE file before
presenting the repository as open source or granting reuse permissions.