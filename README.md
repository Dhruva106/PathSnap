# PathSnap – GPS Trajectory Map Matching Using Machine Learning

<p align="center">
  <strong>Machine Learning-Based GPS Trajectory Map Matching</strong>
</p>

## 1. Project Overview

PathSnap is a machine-learning-based GPS trajectory map-matching project that aims to identify the most likely road segments corresponding to noisy GPS observations.

GPS devices record geographical coordinates at regular intervals. However, due to signal interference, environmental conditions, device limitations and positional inaccuracies, recorded GPS coordinates may not lie exactly on the roads travelled by a vehicle.

Map matching addresses this problem by associating GPS observations with the appropriate road segments in a digital road network.

PathSnap uses a candidate-based supervised machine learning approach. It generates nearby road-segment candidates for GPS observations, extracts relevant geographical and movement-related features, and trains classification models to distinguish correct road candidates from incorrect ones.

Five machine learning algorithms are compared, and Bayesian Hyperparameter Optimization is applied to tune the selected model.

**Baseline XGBoost achieved the highest validation F1-score of 0.7883 and was selected as the champion model.**

---

## 2. Problem Statement

Raw GPS trajectories frequently contain positional errors. As a result, a vehicle's recorded location may appear:

- Between two nearby roads.
- On the wrong side of a divided highway.
- Outside the actual road network.
- Near multiple possible road segments.
- Away from the road actually travelled.

A simple nearest-road approach may select an incorrect road segment, especially in dense urban environments where several roads are located close to one another.

The objective of PathSnap is to identify the most probable road segment associated with each GPS observation using machine learning.

The project focuses on learning the relationship between GPS observations, road geometry, spatial proximity and movement characteristics to improve candidate-road classification.

---

## 3. Project Objectives

The primary objectives are:

1. Understand the GPS trajectory map-matching problem.
2. Explore and understand the available GPS trajectory dataset.
3. Preprocess GPS observations and road-network information.
4. Generate candidate road segments for each GPS observation.
5. Engineer meaningful geographical and movement-related features.
6. Perform feature selection to identify useful predictive features.
7. Train and compare five machine learning classification algorithms.
8. Select the best-performing model using validation metrics.
9. Apply Bayesian Hyperparameter Optimization using Optuna.
10. Evaluate the optimized model against its baseline.
11. Establish a foundation for accurate GPS trajectory map matching and route reconstruction.

---

## 4. Dataset Description

PathSnap uses GPS trajectory data and corresponding road-network information for supervised machine learning.

### Dataset Statistics

| Description | Value |
|---|---:|
| Training trajectories | 80 |
| GPS observations | 208,417 |
| Training trajectories used for model fitting | 60 |
| Validation trajectories | 20 |
| Student-test trajectories | 10 |
| Training candidate instances | 1,280,935 |
| Validation candidate instances | 482,577 |

### Dataset Split

The 80 training trajectories are divided at the trajectory level:

| Dataset | Number of trajectories | Purpose |
|---|---:|---|
| Training | 60 | Model training |
| Validation | 20 | Model comparison and selection |
| Student test | 10 | Held-out evaluation |

The 10 student-test trajectories are kept untouched during model training, validation and hyperparameter optimization.

### Class Distribution

The candidate-classification dataset is imbalanced.

| Class | Description | Distribution |
|---|---|---:|
| Positive (1) | Correct road candidates | 10.59% |
| Negative (0) | Incorrect road candidates | 89.41% |

The positive class represents candidate road segments that correspond to the correct road, while the negative class represents incorrect candidate segments.

Because of this class imbalance, accuracy alone is not sufficient to evaluate model performance.

---

## 5. Understanding the Map-Matching Workflow

PathSnap follows a candidate-based classification workflow.

### Step 1: GPS Trajectory Input

The system receives GPS observations containing geographical information such as:

- Latitude
- Longitude
- Timestamp
- Trajectory identifier

Each trajectory represents a sequence of GPS observations collected during a journey.

### Step 2: Data Preprocessing

Raw GPS data is inspected and processed to improve consistency.

The preprocessing stage includes:

- Validating GPS coordinates.
- Checking timestamps.
- Identifying missing values.
- Checking invalid geographical coordinates.
- Organizing observations by trajectory.
- Converting geographic coordinates into a local metric coordinate system for distance calculations.
- Calculating movement-related information such as speed and heading.

### Step 3: Candidate Road Generation

For each GPS observation, nearby road segments are identified using spatial indexing.

Rather than evaluating every road in the entire network, the system generates a smaller set of nearby candidate road segments.

For example, a GPS observation may have three possible candidate roads:

- Candidate A
- Candidate B
- Candidate C

These candidates are passed to the feature-engineering stage for further evaluation.

### Step 4: Feature Engineering

Features are calculated for each GPS-observation and candidate-road pair.

These features describe spatial proximity, road geometry, movement and directional alignment.

### Step 5: Feature Selection

Initially, 17 features were engineered.

Variance analysis and correlation-based feature selection were used to identify useful features and reduce unnecessary redundancy.

The final feature set contains 14 selected features.

### Step 6: Model Training

Five classification algorithms are trained to distinguish correct road candidates from incorrect ones.

### Step 7: Model Evaluation

The models are compared using validation metrics, including:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

### Step 8: Hyperparameter Optimization

Optuna Bayesian Optimization using the TPE sampler is applied to tune the selected model.

The optimized model is compared with its baseline, and the better-performing model is retained.

---

## 6. Data Preprocessing

Data preprocessing ensures that the GPS observations and road-network information are suitable for machine learning.

The following operations are performed:

### 6.1 Coordinate Validation

GPS coordinates are checked for missing values and invalid geographical ranges.

### 6.2 Timestamp Validation

Timestamps are inspected to ensure that observations can be ordered correctly and used for movement calculations.

### 6.3 Coordinate Transformation

Geographic coordinates are converted into a local metric coordinate system so that geographical distances can be calculated using consistent units.

### 6.4 Speed Calculation

Speed is estimated using the distance travelled between consecutive GPS observations divided by the corresponding time difference.

### 6.5 Heading Calculation

Heading represents the direction of movement between consecutive GPS observations.

It provides useful information for comparing vehicle movement with the direction of a candidate road segment.

### 6.6 Candidate Generation

Spatial indexing is used to identify nearby road segments for each GPS observation.

### 6.7 Trajectory-Level Data Splitting

Data is divided by complete trajectories rather than randomly splitting individual observations.

This prevents observations from the same trajectory from appearing in both training and validation datasets and helps reduce data leakage.

---

## 7. Feature Engineering and Feature Selection

Feature engineering is an important part of PathSnap because the model must distinguish between multiple nearby road candidates.

Initially, 17 features were engineered.

After variance analysis and correlation-based feature selection, 14 features were retained.

### Final Selected Features

| No. | Feature | Description |
|---:|---|---|
| 1 | Distance to road segment | Distance between the GPS observation and the candidate road segment |
| 2 | Distance to endpoint u | Distance from the GPS observation to the first endpoint |
| 3 | Distance to endpoint v | Distance from the GPS observation to the second endpoint |
| 4 | Distance to midpoint | Distance between the GPS observation and the segment midpoint |
| 5 | Road segment length | Length of the candidate road segment |
| 6 | Projection ratio | Relative position of the projected GPS observation along the segment |
| 7 | GPS speed | Estimated speed of the vehicle between consecutive observations |
| 8 | Candidate distance rank | Rank of the candidate based on its distance from the GPS observation |
| 9 | Relative distance | Candidate distance relative to other nearby candidates |
| 10 | Delta X | Horizontal coordinate displacement |
| 11 | Delta Y | Vertical coordinate displacement |
| 12 | Heading cosine similarity | Similarity between GPS movement direction and road direction |
| 13 | Minimum node distance | Minimum distance from the GPS observation to the segment's endpoints |
| 14 | Node asymmetry | Difference-based measure of the relative distances to the two endpoints |

### Why These Features Matter

The selected features represent four important aspects:

**Spatial proximity:** Distance to segment, endpoint distances and midpoint distance help identify nearby roads.

**Road geometry:** Road length, projection ratio and node-related features describe the structure and position of a candidate road segment.

**Movement characteristics:** Speed, Delta X and Delta Y describe vehicle movement.

**Directional alignment:** Heading cosine similarity helps determine whether the direction of GPS movement agrees with the candidate road's direction.

Combining these features allows machine learning models to evaluate candidates using more information than simple nearest-road distance.

---

## 8. Understanding Variance and Bias

Bias and variance are two important concepts in machine learning that influence how well a model generalizes to unseen data.

### Bias

Bias is the error caused by a model making overly simple assumptions about the relationship between input features and the target.

A model with high bias may fail to capture important patterns in the dataset.

### Variance

Variance is the error caused by a model being overly sensitive to variations in the training data.

A model with high variance may perform very well on training data but poorly on unseen data.

### Bias-Variance Trade-off

The goal is to select a model that balances bias and variance.

PathSnap compares different classification algorithms to identify a model that captures useful patterns while maintaining good validation performance.

---

## 9. Overfitting and Underfitting

### Overfitting

Overfitting occurs when a model learns the training data too closely, including noise and unnecessary patterns.

An overfitted model may achieve high training performance but perform poorly on unseen trajectories.

### Underfitting

Underfitting occurs when a model is too simple to learn important relationships in the dataset.

An underfitted model performs poorly on both training and validation data.

### How PathSnap Addresses These Issues

- Uses trajectory-level train-validation splitting.
- Compares multiple machine learning algorithms.
- Evaluates models using validation metrics.
- Uses regularization techniques available in selected algorithms.
- Applies Bayesian Hyperparameter Optimization.
- Keeps the student-test trajectories separate from model selection.

---

## 10. Understanding Classification Errors

The candidate-classification dataset contains positive and negative examples.

- Positive examples represent correct road candidates.
- Negative examples represent incorrect road candidates.

The models learn from these labelled examples by minimizing their classification loss.

Their predictions are evaluated using a confusion matrix.

| Term | Meaning |
|---|---|
| True Positive (TP) | Correct road candidate predicted as correct |
| True Negative (TN) | Incorrect road candidate predicted as incorrect |
| False Positive (FP) | Incorrect road candidate predicted as correct |
| False Negative (FN) | Correct road candidate predicted as incorrect |

Understanding these errors helps evaluate how effectively a model identifies the correct road candidates.

---

## 11. Machine Learning Models

Five machine learning algorithms are compared in PathSnap.

### 11.1 Logistic Regression

Logistic Regression is a supervised classification algorithm that calculates a weighted combination of input features and applies a sigmoid function to estimate the probability of a candidate belonging to the positive class.

In PathSnap, it predicts whether a candidate road segment is likely to be correct.

**Why it is used:**
- Simple and interpretable.
- Computationally efficient.
- Provides probability estimates.
- Useful as a baseline model.

### 11.2 Linear Support Vector Machine (SVM)

Linear SVM identifies a decision boundary that separates correct and incorrect road candidates while maximizing the margin between the two classes.

It learns how the selected numerical features distinguish positive and negative candidates.

**Why it is used:**
- Effective for classification.
- Learns useful linear decision boundaries.
- Works well with appropriately scaled numerical features.
- Provides a different approach from tree-based algorithms.

### 11.3 Random Forest

Random Forest is an ensemble learning algorithm that combines multiple independently trained decision trees.

It uses bagging, where different trees are trained using sampled training data. Their predictions are combined using voting or probability averaging.

**Why it is used:**
- Captures nonlinear relationships.
- Learns interactions between features.
- Reduces variance through ensemble learning.
- Handles structured numerical data effectively.

### 11.4 XGBoost

XGBoost stands for Extreme Gradient Boosting.

It is a gradient-boosting algorithm that builds decision trees sequentially. Each new tree attempts to improve the predictions of the previous trees by reducing the remaining errors.

It also includes regularization techniques to control model complexity.

**Why it is used:**
- Strong performance on structured datasets.
- Captures nonlinear feature relationships.
- Learns interactions between geographical and movement-related features.
- Supports regularization and flexible hyperparameter tuning.

### 11.5 CatBoost

CatBoost is a gradient-boosting algorithm that builds decision trees sequentially and uses ordered boosting techniques.

It is particularly effective for datasets containing categorical features, while also supporting numerical features.

**Why it is used:**
- Uses advanced boosting techniques.
- Can model nonlinear relationships.
- Supports numerical and categorical input features.
- Provides another boosting-based comparison with XGBoost.

---

## 12. Model Evaluation Metrics

PathSnap uses multiple evaluation metrics to compare model performance.

| Metric | Description |
|---|---|
| Accuracy | Percentage of all predictions classified correctly |
| Precision | Percentage of predicted positive candidates that are actually positive |
| Recall | Percentage of actual positive candidates correctly identified |
| F1-score | Harmonic mean of precision and recall |
| ROC-AUC | Measures the model's ability to distinguish between positive and negative classes |
| PR-AUC | Summarizes the precision-recall relationship across different classification thresholds |

Because the dataset contains 10.59% positive candidates and 89.41% negative candidates, F1-score is prioritized over accuracy alone.

F1-score provides a balance between precision and recall, making it useful for evaluating performance on this imbalanced classification problem.

---

## 13. Best Model Selection

After evaluating the five baseline models, **baseline XGBoost was selected as the champion model**.

### Baseline XGBoost Validation Results

| Metric | Score |
|---|---:|
| Accuracy | 95.42% |
| Precision | 80.48% |
| Recall | 77.24% |
| F1-score | 78.83% |
| ROC-AUC | 97.90% |
| PR-AUC | 86.93% |

### Why Was XGBoost Selected?

XGBoost achieved the highest validation F1-score among the five baseline models.

It can capture nonlinear relationships and feature interactions between:

- GPS-to-road distances.
- Heading similarity.
- Road geometry.
- Movement characteristics.
- Candidate ranking information.

It also provides regularization techniques to control model complexity and reduce overfitting.

Since the dataset is imbalanced, F1-score was prioritized over accuracy alone.

Therefore, baseline XGBoost was selected as the champion model.

---

## 14. Bayesian Hyperparameter Optimization

Hyperparameter optimization is the process of finding suitable configuration values that improve a machine learning model's performance.

PathSnap uses Optuna with Bayesian Hyperparameter Optimization through the Tree-structured Parzen Estimator (TPE) sampler.

Unlike Grid Search, which systematically tests combinations, Bayesian optimization uses previous trial results to guide subsequent parameter selections.

### Optimization Configuration

| Parameter | Configuration |
|---|---|
| Optimization framework | Optuna |
| Optimization approach | Bayesian Optimization |
| Sampler | TPE (Tree-structured Parzen Estimator) |
| Number of trials | 25 |
| Cross-validation | 3-fold GroupKFold |
| Grouping | Trajectory identifier |
| Optimization objective | F1-score |
| Model being tuned | XGBoost |

### Optimization Workflow

1. Define the hyperparameter search space.
2. Initialize the Optuna study with the TPE sampler.
3. Generate a proposed hyperparameter configuration.
4. Evaluate the configuration using 3-fold GroupKFold cross-validation.
5. Calculate the objective F1-score.
6. Use previous trial results to guide subsequent trials.
7. Repeat for 25 trials.
8. Identify the best hyperparameter configuration.
9. Evaluate the optimized model against the baseline.

### Optimization Result

The optimized XGBoost model did not improve the baseline validation F1-score.

Therefore, the baseline XGBoost model was retained as the champion model.

This demonstrates that hyperparameter optimization does not guarantee an improvement in every experiment. Model selection should always be based on actual evaluation results.

---

## 15. Model Validation and Data Leakage Prevention

PathSnap uses trajectory-level data splitting to evaluate generalization.

The 80 training trajectories are divided into:

- 60 training trajectories.
- 20 validation trajectories.

The student-test trajectories are kept separate.

### Why Trajectory-Level Splitting?

GPS observations belonging to the same journey are related to one another.

If individual GPS observations were randomly divided between training and validation, observations from the same journey could appear in both datasets.

This could allow the model to learn trajectory-specific patterns and produce overly optimistic validation results.

Splitting by complete trajectories helps evaluate how well the model performs on journeys it has not encountered during training.

Grouped cross-validation is also used during Bayesian optimization to preserve trajectory-level separation between cross-validation folds.

---

## 16. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Primary programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Scikit-learn | Classification models and evaluation |
| XGBoost | Gradient-boosted classification |
| CatBoost | Gradient-boosted classification |
| Optuna | Bayesian hyperparameter optimization |
| SciPy / Spatial indexing tools | Spatial computations and candidate generation |
| Matplotlib | Data visualization |
| Joblib | Model serialization |
| Jupyter Notebook | Dataset exploration and experimentation |

The exact package dependencies and versions are maintained in `requirements.txt`.

---

## 17. Project Directory Structure

```text
PathSnap/
│
├── data/
│   ├── raw/
│   │   ├── train/
│   │   └── student_test/
│   │
│   └── processed/
│
├── notebooks/
│   └── 01_dataset_exploration.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── candidate_generation.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── hmm_decoder.py
│   └── inference.py
│
├── models/
│
├── api/
│   └── main.py
│
├── frontend/
│
├── results/
│
├── visualizations/
│
├── requirements.txt
│
└── README.md
```

### Main Components

| Component | Responsibility |
|---|---|
| `data/` | Stores raw and processed datasets |
| `notebooks/` | Dataset exploration and experiments |
| `data_loader.py` | Loads trajectory and road-network data |
| `preprocessing.py` | Cleans and prepares GPS observations |
| `candidate_generation.py` | Generates nearby road candidates |
| `feature_engineering.py` | Extracts candidate-related features |
| `train.py` | Trains classification models |
| `evaluate.py` | Evaluates and compares model performance |
| `hmm_decoder.py` | Intended for sequence-based route decoding |
| `inference.py` | Intended for prediction workflows |
| `models/` | Stores trained model artifacts |
| `api/` | Intended for API integration |
| `frontend/` | Intended for user-facing visualization |
| `results/` | Stores evaluation results |
| `visualizations/` | Stores plots and visual analysis |

---

## 18. Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version.
- Git.
- pip.
- Visual Studio Code or another Python IDE.
- The PathSnap dataset.

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd PathSnap
```

Replace `<repository-url>` with the actual repository URL.

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it.

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Prepare the Dataset

Place the dataset in the appropriate directories under:

```text
data/raw/
```

Ensure that the folder structure matches the data-loading implementation and dataset documentation.

### Step 5: Execute the Project

Run the project scripts using their supported entry points and command-line arguments.

Refer to the implemented scripts for the exact execution commands.

---

## 19. Limitations

Although candidate-road classification provides a useful approach to GPS map matching, several challenges remain:

1. **GPS Noise:** Large positional errors can make the correct road difficult to identify.
2. **Candidate Generation:** If the correct road is not included in the candidate set, the classifier cannot select it.
3. **Closely Spaced Roads:** Parallel roads and nearby road segments can have similar geographical features.
4. **Class Imbalance:** Incorrect candidates significantly outnumber correct candidates.
5. **Trajectory Consistency:** Independent candidate predictions may not form a continuous or physically realistic route.
6. **Generalization:** Model performance may vary across different geographical regions and road networks.

Candidate-generation recall is an important additional measure because classifier performance alone does not show whether the correct road was available among the generated candidates.

---

## 20. Future Scope

The broader PathSnap system can be extended through the following improvements:

### Hidden Markov Model (HMM)

Use sequential GPS observations and road-network transitions to improve consistency between consecutive road predictions.

### Route Reconstruction

Combine selected road segments into continuous reconstructed routes.

### Interactive Map Visualization

Display raw GPS trajectories and predicted road matches on an interactive map.

### API Integration

Provide an interface for submitting GPS trajectories and retrieving map-matching predictions.

### Frontend Development

Develop a user-friendly interface for uploading trajectories, running predictions and visualizing reconstructed routes.

### Performance Optimization

Improve spatial candidate generation and model inference efficiency for larger road networks.

---

## 21. Conclusion

PathSnap explores supervised machine learning for GPS trajectory map matching.

The project combines GPS data preprocessing, candidate road generation, feature engineering, feature selection, model training and evaluation to identify the most likely road segment corresponding to each GPS observation.

Five machine learning algorithms were compared:

1. Logistic Regression
2. Linear SVM
3. Random Forest
4. XGBoost
5. CatBoost

Baseline XGBoost achieved the highest validation F1-score of **0.7883**, with an accuracy of **95.42%**, precision of **80.48%** and recall of **77.24%**.

Bayesian Hyperparameter Optimization using Optuna's TPE sampler was applied to tune XGBoost. Since the optimized model did not outperform the baseline, baseline XGBoost was retained as the champion model.

The project establishes a machine learning foundation for identifying road candidates from noisy GPS observations, with further scope for sequence-based decoding and complete route reconstruction.

---

## Project Information

| Field | Details |
|---|---|
| Project Name | PathSnap |
| Domain | Machine Learning / GPS Trajectory Analysis |
| Application | GPS Map Matching |
| Learning Type | Supervised Machine Learning |
| Primary Task | Candidate Road Segment Classification |
| Models | Logistic Regression, Linear SVM, Random Forest, XGBoost, CatBoost |
| Feature Selection | Variance and Correlation Analysis |
| Hyperparameter Optimization | Bayesian Optimization using Optuna TPE |
| Champion Model | Baseline XGBoost |
| Validation F1-score | 0.7883 |
| Programming Language | Python |
