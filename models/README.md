# PathSnap Model Registry & Artifact Package

This directory contains all trained machine learning models, calibrated hyperparameters, preprocessing configurations, feature schemas, and serialization metadata for PathSnap.

---

## Directory Layout

```
models/
├── champion/
│   ├── best_model.joblib               # AUTHORITATIVE CHAMPION (Baseline XGBoost)
│   └── best_model_metadata.json        # Champion metadata & validation metrics
├── baseline/
│   ├── xgboost.joblib                  # Baseline XGBoost classifier
│   ├── xgboost_metadata.json
│   ├── catboost.joblib                 # Baseline CatBoost classifier
│   ├── catboost_metadata.json
│   ├── random_forest.joblib            # Baseline Random Forest classifier
│   ├── random_forest_metadata.json
│   ├── logistic_regression.joblib      # Baseline Logistic Regression pipeline (StandardScaler + L2)
│   ├── logistic_regression_metadata.json
│   ├── linear_svm.joblib               # Baseline Linear SVM pipeline (StandardScaler + CalibratedClassifierCV)
│   └── linear_svm_metadata.json
├── optimized/
│   ├── xgboost_optimized.joblib        # Bayesian-Optimized XGBoost classifier (Optuna Trial 4)
│   └── xgboost_optimized_metadata.json
├── metadata/
│   ├── model_metadata.json             # Master metadata catalog for all 7 saved models
│   ├── feature_schema.json             # Strict schema, order, and bounds for the 14 features
│   ├── feature_metadata.json           # Physical descriptions and importance scores
│   ├── preprocessing_config.json       # Projection and kinematic configurations
│   ├── training_config.json            # Dataset split and model training parameters
│   └── hmm_config.json                 # HMM Viterbi decoder parameters and validation metrics
├── best_model.joblib                   # Backward-compatible link to champion
└── best_model_metadata.json            # Backward-compatible link to champion metadata
```

---

## Champion Model Specifications

- **Algorithm:** Extreme Gradient Boosted Decision Trees (`XGBClassifier`)
- **Variant:** Baseline XGBoost with default regularization
- **Validation F1-Score:** **0.7883** (78.83%)
- **Validation ROC-AUC:** **0.9790** (97.90%)
- **Validation PR-AUC:** **0.8693** (86.93%)
- **Validation Accuracy:** **0.9542** (95.42%)
- **Inference Latency:** **0.047 seconds** for 482,577 candidate instances ($> 10\text{ million candidates/sec}$)
- **Selected Features:** Exactly 14 geometric, topological, and kinematic features (see `metadata/feature_schema.json`)

---

## Python Loading Usage

```python
from src.model_utils import load_champion_model, load_model, get_model_metadata

# 1. Load the authoritative champion
champion = load_champion_model()
print("Loaded:", champion.__class__.__name__)

# 2. Load any baseline or optimized model
catboost = load_model("catboost")
opt_xgb = load_model("xgboost_optimized")

# 3. Read metadata
meta = get_model_metadata("champion")
print("Validation F1:", meta["validation_metrics"]["F1-Score"])
```
