"""
metrics.py
----------
Evaluation metrics for generated counterfactual explanations.

Metrics match those reported in the PertCF paper (SGAI 2023):
- dissimilarity  (proximity, lower is better)
- sparsity       (fraction of features changed, lower = fewer changes)
- instability    (stability under input perturbation, lower is better)

All metrics accept plain pd.Series / np.ndarray and an optional
SHAPWeightedSimilarity object for distance calculation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional, Union

from .similarity import SHAPWeightedSimilarity


# ---------------------------------------------------------------------------
# Dissimilarity (proximity)
# ---------------------------------------------------------------------------

def dissimilarity(
    query: pd.Series,
    counterfactual: pd.Series,
    similarity_fn: SHAPWeightedSimilarity,
    class_label,
) -> float:
    """
    Measure how dissimilar the query is from the counterfactual.

    dissimilarity = dist(x, x')   (lower is better)

    Parameters
    ----------
    query : pd.Series
        Original instance (feature values only, no label).
    counterfactual : pd.Series
        Generated counterfactual (feature values only, no label).
    similarity_fn : SHAPWeightedSimilarity
        Similarity function configured with the relevant SHAP weights.
    class_label :
        Class label whose SHAP weights are used for the distance.

    Returns
    -------
    float
    """
    return similarity_fn.distance(query, counterfactual, class_label)


def mean_dissimilarity(
    queries: List[pd.Series],
    counterfactuals: List[pd.Series],
    similarity_fn: SHAPWeightedSimilarity,
    class_labels: list,
) -> float:
    """Average dissimilarity over a list of query/CF pairs."""
    vals = [
        dissimilarity(q, cf, similarity_fn, cl)
        for q, cf, cl in zip(queries, counterfactuals, class_labels)
    ]
    return float(np.mean(vals))


# Sparsity
def sparsity(
    query: pd.Series,
    counterfactual: pd.Series,
    feature_names: Optional[List[str]] = None,
) -> float:
    """
    Fraction of features that remain **unchanged** between query and CF.

    sparsity = (# unchanged features) / (total features)

    Higher sparsity → fewer features changed → more actionable CF.
    The paper computes this as L0 norm / m (fraction *changed*);
    we follow the same convention (lower fraction changed = lower sparsity
    score, which equals a higher proportion unchanged).

    To match the paper's Table 2 numbers, we return the fraction of
    features that **changed** (i.e., 1 - fraction_unchanged).

    Parameters
    ----------
    query : pd.Series
        Original instance features.
    counterfactual : pd.Series
        Counterfactual features.
    feature_names : list of str, optional
        Subset of features to compare. Defaults to all shared features.

    Returns
    -------
    float in [0, 1]
    """
    if feature_names is None:
        feature_names = query.index.tolist()

    changed = sum(
        1 for f in feature_names
        if str(query[f]) != str(counterfactual[f])
    )
    return float(changed / len(feature_names))


def mean_sparsity(
    queries: List[pd.Series],
    counterfactuals: List[pd.Series],
    feature_names: Optional[List[str]] = None,
) -> float:
    """Average sparsity over a list of pairs."""
    vals = [sparsity(q, cf, feature_names) for q, cf in zip(queries, counterfactuals)]
    return float(np.mean(vals))


# Instability
def instability(
    query: pd.Series,
    counterfactual: pd.Series,
    explainer,  # PertCFExplainer instance
    perturbation: float = 0.01,
) -> float:
    """
    Stability of the generated CF under small input perturbation.

    Procedure (following the paper):
    1. Create a slightly perturbed version y of the query x.
    2. Generate a CF y' for y using the same explainer.
    3. Return dist(x', y').

    Lower instability → more stable explainer.

    Parameters
    ----------
    query : pd.Series
        Original query (feature values only, no label).
    counterfactual : pd.Series
        CF generated for query.
    explainer : PertCFExplainer
        Fitted explainer used to generate the CF for the perturbed query.
    perturbation : float
        Relative perturbation magnitude for numeric features (default 1%).

    Returns
    -------
    float
    """
    # Build perturbed query
    perturbed = query.copy().astype(object)
    for feat in explainer.feature_names:
        if feat not in explainer.categorical_features:
            perturbed[feat] = float(query[feat]) * (1 + perturbation)
        # Categorical features are left unchanged for small perturbation

    # Predict class of perturbed instance
    p_class = explainer.adapter.predict(
        pd.DataFrame([perturbed[explainer.feature_names]])
    )[0]

    perturbed_with_label = perturbed.copy()
    perturbed_with_label[explainer.label] = p_class

    # Generate CF for perturbed instance
    cf_perturbed = explainer.explain(perturbed_with_label)
    if cf_perturbed is None:
        return float("nan")

    # Distance between original CF and perturbed CF
    cf_features = counterfactual[explainer.feature_names]
    cf_p_features = cf_perturbed[explainer.feature_names]

    return explainer.sim_fn.distance(cf_features, cf_p_features, str(explainer.adapter.class_names[0]))


def mean_instability(
    queries: List[pd.Series],
    counterfactuals: List[pd.Series],
    explainer,
    perturbation: float = 0.01,
) -> float:
    """Average instability over a list of pairs."""
    vals = []
    for q, cf in zip(queries, counterfactuals):
        v = instability(q, cf, explainer, perturbation)
        if not np.isnan(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


# Convenience: compute all metrics at once
def evaluate(
    queries: List[pd.Series],
    counterfactuals: List[pd.Series],
    explainer,
    class_labels: Optional[list] = None,
    perturbation: float = 0.01,
    compute_instability: bool = True,
) -> dict:
    """
    Compute all three metrics for a list of query/CF pairs.

    Returns
    -------
    dict with keys: dissimilarity, sparsity, instability
    """
    if class_labels is None:
        # Use the CF's class label for the distance weight
        class_labels = [
            cf.get(explainer.label, explainer.adapter.class_names[0])
            for cf in counterfactuals
        ]

    cf_features = [cf[explainer.feature_names] for cf in counterfactuals]
    q_features = [q[explainer.feature_names] for q in queries]

    result = {
        "dissimilarity": mean_dissimilarity(
            q_features, cf_features, explainer.sim_fn, class_labels
        ),
        "sparsity": mean_sparsity(q_features, cf_features, explainer.feature_names),
    }

    if compute_instability:
        result["instability"] = mean_instability(
            queries, cf_features, explainer, perturbation
        )

    return result
