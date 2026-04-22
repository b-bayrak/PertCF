"""
pertcf
======
A dependency-light Python package for PertCF: a perturbation-based
counterfactual explanation method that leverages SHAP feature attributions.

Based on:
    Bayrak & Bach (2023). "PertCF: A Perturbation-Based Counterfactual
    Generation Approach." SGAI AI-2023, LNAI 14381, pp. 174-187.
    https://doi.org/10.1007/978-3-031-47994-6_13

Quick start
-----------
    from pertcf import PertCFExplainer

    explainer = PertCFExplainer(
        model=clf,
        X_train=X_train,
        y_train=y_train,
        categorical_features=["gender", "purpose"],
        label="class",
        num_iter=5,
        coef=5,
    )
    explainer.fit()
    cf = explainer.explain(test_instance)
"""

from . import metrics
from .adapters import ModelAdapter, wrap_model
from .core import PertCFExplainer
from .similarity import SHAPWeightedSimilarity

__version__ = "1.0.0"
__author__ = "Betül Bayrak"
__license__ = "MIT"

__all__ = [
    "PertCFExplainer",
    "SHAPWeightedSimilarity",
    "wrap_model",
    "ModelAdapter",
    "metrics",
]
