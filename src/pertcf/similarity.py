"""
similarity.py
-------------
SHAP-weighted similarity and distance functions.

Design
------
For numeric features:
    sim(a, b) = 1 - |a - b| / feature_range

For categorical (nominal) features:
    sim(a, b) = 1  if a == b
    sim(a, b) = custom_value  if provided via similarity_matrices
    sim(a, b) = 0  otherwise

Weighted similarity (amalgamation function):
    sim(x, y | class_k) = sum_i( w_ki * sim_i(x_i, y_i) )
                          where w_ki = normalised SHAP weight for
                          feature i w.r.t. class k.

Distance:
    dist(x, y | class_k) = 1 - sim(x, y | class_k)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class SHAPWeightedSimilarity:
    """
    Computes SHAP-weighted similarity/distance between two instances.

    Parameters
    ----------
    feature_names : list of str
        Ordered list of feature names (must match column order in data).
    categorical_features : list of str
        Names of categorical (nominal/ordinal) features.
    shap_weights : pd.DataFrame
        DataFrame of shape (n_classes, n_features) with normalised SHAP
        weights. Index = class labels, columns = feature names.
    feature_ranges : dict {str: float}
        Range (max - min) for each numeric feature, used for normalisation.
        Computed from training data in PertCFExplainer.fit().
    similarity_matrices : dict {str: dict}, optional
        Custom per-feature similarity matrices for categorical features.
        Format: {feature_name: {(val_a, val_b): similarity_score, ...}}
        If not provided, a simple equality check (0 or 1) is used.
    """

    def __init__(
        self,
        feature_names: List[str],
        categorical_features: List[str],
        shap_weights: pd.DataFrame,
        feature_ranges: Dict[str, float],
        similarity_matrices: Optional[Dict[str, Dict]] = None,
    ):
        self.feature_names = feature_names
        self.categorical_features = categorical_features
        self.shap_weights = shap_weights
        self.feature_ranges = feature_ranges
        self.similarity_matrices = similarity_matrices or {}

    # Per-feature similarity

    def feature_similarity(
        self, feature: str, val_a, val_b, class_label=None
    ) -> float:
        """
        Compute similarity for a single feature.

        Parameters
        ----------
        feature : str
            Feature name.
        val_a, val_b :
            Values to compare.
        class_label :
            Unused (kept for API consistency with ordinal extensions).

        Returns
        -------
        float in [0, 1]  (1 = identical)
        """
        if feature in self.categorical_features:
            return self._categorical_sim(feature, val_a, val_b)
        else:
            return self._numeric_sim(feature, val_a, val_b)

    def _numeric_sim(self, feature: str, val_a, val_b) -> float:
        rng = self.feature_ranges.get(feature, 1.0)
        if rng == 0:
            return 1.0
        return float(1.0 - abs(float(val_a) - float(val_b)) / rng)

    def _categorical_sim(self, feature: str, val_a, val_b) -> float:
        mat = self.similarity_matrices.get(feature, {})
        key = (val_a, val_b)
        rev_key = (val_b, val_a)
        if key in mat:
            return float(mat[key])
        if rev_key in mat:
            return float(mat[rev_key])
        return 1.0 if str(val_a) == str(val_b) else 0.0

    # Weighted similarity / distance between two instances

    def similarity(
        self,
        x: pd.Series,
        y: pd.Series,
        class_label,
    ) -> float:
        """
        Compute the SHAP-weighted similarity between x and y for the
        given class.

        Parameters
        ----------
        x, y : pd.Series indexed by feature names.
        class_label : class for which to use SHAP weights.

        Returns
        -------
        float in [0, 1]
        """
        weights = self.shap_weights.loc[str(class_label), self.feature_names]
        w_total = weights.sum()
        if w_total == 0:
            # fallback: uniform weights
            weights = pd.Series(
                np.ones(len(self.feature_names)) / len(self.feature_names),
                index=self.feature_names,
            )
            w_total = 1.0

        weighted_sim = 0.0
        for feat in self.feature_names:
            w = weights[feat]
            s = self.feature_similarity(feat, x[feat], y[feat])
            weighted_sim += w * s

        return float(weighted_sim / w_total)

    def distance(self, x: pd.Series, y: pd.Series, class_label) -> float:
        """Distance = 1 - similarity."""
        return 1.0 - self.similarity(x, y, class_label)

    # Per-feature similarity breakdown (for transparency / metrics)

    def feature_similarities(
        self, x: pd.Series, y: pd.Series
    ) -> Dict[str, float]:
        """Return a dict of per-feature similarity values."""
        return {
            feat: self.feature_similarity(feat, x[feat], y[feat])
            for feat in self.feature_names
        }
