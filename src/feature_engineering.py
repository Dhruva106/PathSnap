"""
feature_engineering.py - PathSnap Feature Engineering and Leakage-Free Feature Selection
Provides:
1. Domain-specific physical and kinematic feature transformations.
2. Leakage-free FeatureSelector class with fit/transform methods for nested cross-validation.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier


def extract_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain-specific engineered features from candidate observation pairs:
    - Non-linear distance penalties (quadratic, logarithmic)
    - Directional cosine alignment (-1.0 to +1.0)
    - Node proximity and asymmetry interactions
    """
    df = df.copy()

    # 1. Physical Gaussian emission surrogate: squared distance
    df["dist_sq"] = df["dist_to_segment"] ** 2

    # 2. Log-transformed distance for heavy-tail stabilization
    df["log_dist"] = np.log1p(np.maximum(df["dist_to_segment"], 0.0))

    # 3. Directional cosine alignment (-1.0 to +1.0)
    # +1 indicates exact motion alignment with road arc direction, -1 indicates wrong-way motion
    heading_rad = np.radians(df["heading_diff_deg"])
    df["heading_cos"] = np.cos(heading_rad)

    # 4. Minimum node distance (distance to closest intersection)
    df["min_node_dist"] = np.minimum(df["dist_to_u"], df["dist_to_v"])

    # 5. Node asymmetry ratio (indicates whether vehicle is near middle or end of block)
    denom = np.maximum(df["dist_to_u"] + df["dist_to_v"], 1e-6)
    df["node_asymmetry"] = np.abs(df["dist_to_u"] - df["dist_to_v"]) / denom

    return df


class LeakageFreeFeatureSelector:
    """
    Multi-stage feature selector with strict fit/transform discipline.
    Fits only on training data (variance threshold, correlation pruning with MI arbitration),
    and applies identical feature masks to validation/test folds.
    """
    def __init__(
        self,
        candidate_features: List[str],
        corr_threshold: float = 0.85,
        variance_threshold: float = 0.001,
        random_state: int = 42
    ):
        self.candidate_features = list(candidate_features)
        self.corr_threshold = corr_threshold
        self.variance_threshold = variance_threshold
        self.random_state = random_state

        self.selected_features_: List[str] = []
        self.removed_features_: List[Dict[str, str]] = []
        self.importance_df_: Optional[pd.DataFrame] = None
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        current_features = [f for f in self.candidate_features if f in X.columns]
        self.removed_features_ = []

        # Stage 1: Variance Threshold Filtering
        selector = VarianceThreshold(threshold=self.variance_threshold)
        selector.fit(X[current_features])
        variances = selector.variances_

        retained_after_var = []
        for feat, var in zip(current_features, variances):
            if var <= self.variance_threshold:
                self.removed_features_.append({
                    "feature": feat,
                    "stage": "Variance Filtering",
                    "reason": f"Low variance ({var:.5f} <= threshold {self.variance_threshold})"
                })
            else:
                retained_after_var.append(feat)

        # Stage 2: Correlation Analysis (Multicollinearity Pruning)
        corr_matrix = X[retained_after_var].corr().abs()

        # Compute mutual information on training slice for arbitration
        sample_size = min(len(X), 50000)
        idx_sample = np.random.RandomState(self.random_state).choice(len(X), size=sample_size, replace=False) if len(X) > 50000 else np.arange(len(X))

        mi_scores = mutual_info_classif(
            X[retained_after_var].iloc[idx_sample],
            y.iloc[idx_sample],
            random_state=self.random_state,
            n_neighbors=5
        )
        mi_dict = dict(zip(retained_after_var, mi_scores))

        to_drop = set()
        for i in range(len(retained_after_var)):
            feat_i = retained_after_var[i]
            if feat_i in to_drop:
                continue
            for j in range(i + 1, len(retained_after_var)):
                feat_j = retained_after_var[j]
                if feat_j in to_drop:
                    continue

                r = corr_matrix.loc[feat_i, feat_j]
                if r >= self.corr_threshold:
                    if mi_dict[feat_i] >= mi_dict[feat_j]:
                        drop_feat = feat_j
                        keep_feat = feat_i
                    else:
                        drop_feat = feat_i
                        keep_feat = feat_j

                    to_drop.add(drop_feat)
                    self.removed_features_.append({
                        "feature": drop_feat,
                        "stage": "Correlation Analysis",
                        "reason": f"High collinearity with '{keep_feat}' (|r| = {r:.3f} >= {self.corr_threshold}); '{keep_feat}' retained due to higher MI score"
                    })

        self.selected_features_ = [f for f in retained_after_var if f not in to_drop]

        # Stage 3: Random Forest Feature Importance on Retained Features
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=self.random_state, n_jobs=-1)
        rf.fit(X[self.selected_features_].iloc[idx_sample], y.iloc[idx_sample])

        self.importance_df_ = pd.DataFrame({
            "feature": self.selected_features_,
            "mutual_information": [mi_dict[f] for f in self.selected_features_],
            "rf_importance": rf.feature_importances_,
        }).sort_values(by="rf_importance", ascending=False).reset_index(drop=True)

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise ValueError("LeakageFreeFeatureSelector must be fitted before transforming.")
        return X[self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
