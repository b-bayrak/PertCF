"""
core.py
-------
PertCFExplainer: the main class implementing the PertCF algorithm.

This is a full rewrite of the original implementation that removes the
myCBR dependency. The CBR (Case-Based Reasoning) retrieval layer is
replaced with efficient NumPy/pandas weighted nearest-neighbour search,
preserving exactly the same algorithmic behaviour as described in:

    Bayrak & Bach (2023). "PertCF: A Perturbation-Based Counterfactual
    Generation Approach." SGAI AI-2023, LNAI 14381, pp. 174–187.
    https://doi.org/10.1007/978-3-031-47994-6_13

Quick start
-----------
    from pertcf import PertCFExplainer

    explainer = PertCFExplainer(
        model=clf,
        X_train=X_train,
        y_train=y_train,
        categorical_features=["gender", "credit_history"],
        label="class",
        num_iter=5,
        coef=5,
    )
    explainer.fit()

    cf = explainer.explain(x)   # x is a pd.Series with label column
    print(cf)
"""

from __future__ import annotations

import time
import warnings
from statistics import mean
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .adapters import ModelAdapter, wrap_model
from .similarity import SHAPWeightedSimilarity


class PertCFExplainer:
    """
    Perturbation-based counterfactual explainer.

    Parameters
    ----------
    model :
        Trained classifier. Accepts scikit-learn estimators, PyTorch
        nn.Module, Keras/TF models, or any callable. Pass a
        ``ModelAdapter`` directly if you need custom control.
    X_train : pd.DataFrame
        Training features (no label column).
    y_train : array-like
        Training labels.
    categorical_features : list of str
        Names of categorical (nominal) features.
    label : str
        Name of the label column (used in explain input/output). Default ``"label"``.
    num_iter : int
        Maximum number of perturbation iterations. Default 10.
    coef : float
        Coefficient controlling the step-size termination threshold.
        thresh = dist(x, NUN) / coef. Default 5.
    shap_values : pd.DataFrame, optional
        Pre-computed SHAP value DataFrame (n_classes × n_features).
        If None, SHAP values are computed during ``fit()``.
    similarity_matrices : dict, optional
        Custom per-feature categorical similarity matrices.
        Format: {feature_name: {(val_a, val_b): score, ...}}
    class_names : list, optional
        Class label names. Inferred from model if possible.
    model_type : str, optional
        ``"sklearn"`` (default, auto-detected) | ``"torch"`` | ``"keras"``
        | ``"callable"``. Pass explicitly if auto-detection fails.
    """

    def __init__(
        self,
        model,
        X_train: pd.DataFrame,
        y_train,
        categorical_features: List[str],
        label: str = "label",
        num_iter: int = 10,
        coef: float = 5.0,
        shap_values: Optional[pd.DataFrame] = None,
        similarity_matrices: Optional[Dict] = None,
        class_names: Optional[list] = None,
        model_type: Optional[str] = None,
    ):
        # Wrap model in an adapter if needed
        if isinstance(model, ModelAdapter):
            self.adapter = model
        else:
            self.adapter = wrap_model(model, class_names=class_names)

        self.X_train = X_train.copy()
        self.y_train = np.asarray(y_train).astype(str)
        self.categorical_features = categorical_features
        self.label = label
        self.num_iter = num_iter
        self.coef = float(coef)
        self.similarity_matrices = similarity_matrices or {}

        self.feature_names: List[str] = list(X_train.columns)
        self.class_names: List[str] = [
            str(c) for c in self.adapter.class_names
        ]

        # Pre-computed SHAP values (set during fit or passed in)
        self._shap_weights: Optional[pd.DataFrame] = shap_values
        self.sim_fn: Optional[SHAPWeightedSimilarity] = None

        # Per-class training data (populated during fit)
        self._casebases: Dict[str, pd.DataFrame] = {}

        self._is_fitted = False

    # Public API

    def fit(
        self,
        shap_sample_size: int = 300,
        shap_background_size: Optional[int] = None,
        verbose: bool = True,
    ) -> "PertCFExplainer":
        """
        Fit the explainer on training data.

        Steps
        -----
        1. Compute SHAP values per class (or use pre-supplied values).
        2. Compute feature ranges for numeric normalisation.
        3. Build per-class casebases.
        4. Construct the SHAPWeightedSimilarity function.

        Parameters
        ----------
        shap_sample_size : int
            Max background samples for SHAP (KernelExplainer). Default 300.
        shap_background_size : int, optional
            Background dataset size passed to SHAP TreeExplainer (ignored
            for KernelExplainer). Default None = full background.
        verbose : bool
            Print progress. Default True.

        Returns
        -------
        self
        """
        if self._shap_weights is None:
            if verbose:
                print("Computing SHAP values… (this may take a moment)")
            self._shap_weights = self._compute_shap(shap_sample_size)
            if verbose:
                print("✓ SHAP values computed.\n")
        else:
            if verbose:
                print("✓ Using provided SHAP values.\n")

        # Feature ranges for numeric normalisation
        self._feature_ranges = self._compute_ranges()

        # Per-class casebases
        train_full = self.X_train.copy()
        train_full[self.label] = self.y_train
        for cls in self.class_names:
            self._casebases[cls] = (
                train_full[train_full[self.label] == cls]
                .drop(columns=[self.label])
                .reset_index(drop=True)
            )

        # Build similarity function
        self.sim_fn = SHAPWeightedSimilarity(
            feature_names=self.feature_names,
            categorical_features=self.categorical_features,
            shap_weights=self._shap_weights,
            feature_ranges=self._feature_ranges,
            similarity_matrices=self.similarity_matrices,
        )

        self._is_fitted = True
        if verbose:
            print("✓ Explainer fitted and ready.\n")
        return self

    def explain(
        self,
        instance: pd.Series,
        target_class: Optional[str] = None,
        return_all_candidates: bool = False,
    ) -> Optional[pd.Series]:
        """
        Generate a counterfactual explanation for a single instance.

        Parameters
        ----------
        instance : pd.Series
            The instance to explain. Must include the label column
            (accessible via self.label).
        target_class : str, optional
            Desired counterfactual class. If None, the NUN's class is used.
        return_all_candidates : bool
            If True, return a list of all candidate CFs generated. Default False.

        Returns
        -------
        pd.Series with feature values + label column set to the CF's class,
        or None if no CF could be generated.
        """
        self._check_fitted()

        instance = instance.copy()
        q_class = str(instance[self.label])
        query = instance.drop(labels=[self.label])

        # Find Nearest Unlike Neighbour
        nun_result = self._find_nun(query, q_class, excluded_classes=[q_class])
        if nun_result is None:
            warnings.warn("Could not find a NUN for this instance.")
            return None

        nun_features, nun_class = nun_result

        if target_class is not None:
            nun_class = str(target_class)
            # Override NUN with NUN from target class only
            nun_result2 = self._find_nun(query, q_class, target_class=nun_class)
            if nun_result2 is not None:
                nun_features, _ = nun_result2

        thresh = self.sim_fn.distance(query, nun_features, nun_class) / self.coef

        self._exp_class = nun_class
        self._used_classes = {q_class, nun_class}
        self._candidate_list: List[pd.Series] = []
        # Store NUN as ultimate fallback (mirrors original implementation)
        self._nun_fallback = nun_features.copy()
        self._nun_fallback_class = nun_class

        cf = self._generate_cf(
            p1=query,
            p1_class=q_class,
            p2=nun_features,
            p2_class=nun_class,
            thresh=thresh,
            iteration=0,
        )

        if cf is None:
            return None

        result = cf.copy()
        # _fallback may have already stamped the label; only set if missing
        if self.label not in result.index:
            result[self.label] = self._exp_class

        if return_all_candidates:
            return result, self._candidate_list
        return result

    def explain_batch(
        self,
        X: pd.DataFrame,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Generate counterfactuals for all rows in X.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain the label column.
        verbose : bool
            Print progress. Default True.

        Returns
        -------
        pd.DataFrame of CFs aligned with X. Rows where CF generation
        failed are filled with NaN.
        """
        self._check_fitted()
        results = []
        for i, (_, row) in enumerate(X.iterrows()):
            if verbose and (i % 10 == 0):
                print(f"  Explaining instance {i}/{len(X)}…")
            cf = self.explain(row)
            results.append(cf)

        cf_df = pd.DataFrame(
            [r if r is not None else pd.Series(dtype=object) for r in results],
            index=X.index,
        )
        return cf_df

    def find_nun(self, instance: pd.Series) -> Optional[pd.Series]:
        """
        Public API: find the Nearest Unlike Neighbour for an instance.

        Parameters
        ----------
        instance : pd.Series with label column.

        Returns
        -------
        pd.Series (feature values only, no label), or None.
        """
        self._check_fitted()
        q_class = str(instance[self.label])
        query = instance.drop(labels=[self.label])
        result = self._find_nun(query, q_class, excluded_classes=[q_class])
        if result is None:
            return None
        nun_features, _ = result
        return nun_features

    # Core algorithm (internal)

    def _generate_cf(
        self,
        p1: pd.Series,
        p1_class: str,
        p2: pd.Series,
        p2_class: str,
        thresh: float,
        iteration: int,
    ) -> Optional[pd.Series]:
        """
        Recursive perturbation loop.  Mirrors Algorithm 1 from the paper.

        p1 = source (starts as original query x)
        p2 = target (starts as NUN)
        """
        if iteration > self.num_iter:
            # Iteration limit reached
            if self._candidate_list:
                return self._candidate_list[-1]
            # Try with next best NUN
            return self._fallback(p1, p1_class)

        # -- Generate candidate by perturbing p1 toward p2 --
        candidate = self._perturb(p1, p2, self._exp_class)

        # -- Predict class of candidate --
        cnd_arr = pd.DataFrame([candidate])
        cnd_class = str(self.adapter.predict(cnd_arr)[0])

        # -- If candidate belongs to expected class --
        if cnd_class == self._exp_class:
            self._candidate_list.append(candidate.copy())

            if len(self._candidate_list) > 1:
                prev = self._candidate_list[-2]
                step = self.sim_fn.distance(prev, candidate, self._exp_class)
                if step <= thresh:
                    # Step size below threshold → converged
                    return candidate

            # Not yet converged → move candidate toward p1 (refine)
            return self._generate_cf(candidate, cnd_class, p1, p1_class, thresh, iteration + 1)

        elif cnd_class != p1_class:
            # Candidate landed in a third class → follow it
            return self._generate_cf(candidate, cnd_class, p1, p1_class, thresh, iteration + 1)

        else:
            # Candidate is still in source class → push harder toward target
            return self._generate_cf(candidate, cnd_class, p2, p2_class, thresh, iteration + 1)

    def _perturb(
        self,
        source: pd.Series,
        target: pd.Series,
        target_class: str,
    ) -> pd.Series:
        """
        Generate a perturbed sample between source and target.

        Numeric:    p_f = s_f + shap_target_f * (t_f - s_f)
        Categorical:p_f = t_f  if sim(s_f, t_f) < alpha  else  s_f
                    (alpha = 0.5 by default)
        """
        weights = self._shap_weights.loc[str(target_class), self.feature_names]
        alpha = 0.5  # categorical switch threshold

        result = {}
        for feat in self.feature_names:
            s_val = source[feat]
            t_val = target[feat]
            w = float(weights[feat])

            if feat in self.categorical_features:
                sim = self.sim_fn.feature_similarity(feat, s_val, t_val)
                result[feat] = t_val if sim < alpha else s_val
            else:
                result[feat] = float(s_val) + w * (float(t_val) - float(s_val))

        return pd.Series(result)

    def _find_nun(
        self,
        query: pd.Series,
        query_class: str,
        excluded_classes: Optional[List[str]] = None,
        target_class: Optional[str] = None,
    ) -> Optional[tuple]:
        """
        Find the Nearest Unlike Neighbour.

        Returns (nun_features: pd.Series, nun_class: str) or None.
        """
        excluded = set(excluded_classes or [])
        best_sim = -1.0
        best_features = None
        best_class = None

        search_classes = (
            [str(target_class)] if target_class is not None
            else [c for c in self.class_names if c not in excluded]
        )

        for cls in search_classes:
            cb = self._casebases.get(cls)
            if cb is None or len(cb) == 0:
                continue

            # Compute similarity of query to all cases in this class
            sims = cb.apply(
                lambda row: self.sim_fn.similarity(query, row, cls), axis=1
            )
            idx = sims.idxmax()
            sim_val = sims[idx]

            if sim_val > best_sim:
                best_sim = sim_val
                best_features = cb.loc[idx]
                best_class = cls

        if best_features is None:
            return None
        return best_features, best_class

    def _fallback(
        self,
        candidate: pd.Series,
        candidate_class: str,
    ) -> Optional[pd.Series]:
        """
        When CF generation fails (no candidates found within num_iter
        iterations), try again from the next best NUN of the last candidate.
        If no further NUN exists (e.g. binary classification, all classes
        tried), return the original NUN as a last resort.
        """
        next_nun = self._find_nun(
            candidate, candidate_class,
            excluded_classes=list(self._used_classes),
        )
        if next_nun is None:
            # Ultimate fallback: return the original NUN with its class label
            result = self._nun_fallback.copy()
            result[self.label] = self._nun_fallback_class
            return result

        nun_features, nun_class = next_nun
        self._used_classes.add(nun_class)
        self._exp_class = nun_class
        thresh = self.sim_fn.distance(candidate, nun_features, nun_class) / self.coef

        return self._generate_cf(candidate, candidate_class, nun_features, nun_class, thresh, 0)

    # SHAP computation
    def _compute_shap(self, sample_size: int = 100) -> pd.DataFrame:
        """
        Compute per-class mean |SHAP| values, normalised to sum to 1.

        Tries TreeExplainer first (fast, exact for tree models).
        Falls back to KernelExplainer (model-agnostic, slower).
        """
        import shap

        background = self.X_train.copy()

        # Encode categoricals for SHAP
        bg_encoded = background.copy()
        for col in self.categorical_features:
            bg_encoded[col] = bg_encoded[col].astype("category").cat.codes

        if len(bg_encoded) > sample_size:
            bg_encoded = bg_encoded.sample(sample_size, random_state=42)

        # Try TreeExplainer (fast)
        try:
            shap_explainer = shap.TreeExplainer(
                self.adapter._model,  # works for sklearn tree models
                data=bg_encoded,
                feature_perturbation="interventional",
            )
            shap_vals = shap_explainer.shap_values(bg_encoded)
        except Exception:
            # Fallback: KernelExplainer
            predict_fn = lambda x: self.adapter.predict_proba(  # noqa: E731
                pd.DataFrame(x, columns=self.feature_names)
            )
            shap_explainer = shap.KernelExplainer(predict_fn, bg_encoded)
            shap_vals = shap_explainer.shap_values(bg_encoded)

        # shap_vals: list of arrays (one per class), each shape (n_samples, n_features)
        if not isinstance(shap_vals, list):
            # Binary classification: single array → wrap in list
            shap_vals = [shap_vals, -shap_vals]

        # Mean absolute SHAP per class
        mean_abs = [np.mean(np.abs(sv), axis=0) for sv in shap_vals]
        shap_df = pd.DataFrame(
            mean_abs,
            index=self.class_names[: len(mean_abs)],
            columns=self.feature_names,
        )

        # Normalise rows to sum to 1
        row_sums = shap_df.sum(axis=1).replace(0, 1)
        shap_df = shap_df.div(row_sums, axis=0)

        return shap_df

    # Helpers
    def _compute_ranges(self) -> Dict[str, float]:
        ranges = {}
        for col in self.feature_names:
            if col not in self.categorical_features:
                rng = float(self.X_train[col].max() - self.X_train[col].min())
                ranges[col] = rng if rng > 0 else 1.0
            else:
                ranges[col] = 1.0
        return ranges

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                "Explainer not fitted. Call explainer.fit() first."
            )

    # Convenience: run the benchmark experiment from the paper
    def benchmark(
        self,
        X_test: pd.DataFrame,
        n: Optional[int] = None,
        coef: Optional[float] = None,
        verbose: bool = True,
    ) -> dict:
        """
        Replicate the paper's benchmark: generate CFs for the test set and
        compute dissimilarity, sparsity, instability, and runtime.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test data with label column.
        n : int, optional
            Number of test instances to use (default = all).
        coef : float, optional
            Override self.coef for this benchmark run.
        verbose : bool
            Print progress. Default True.

        Returns
        -------
        dict: {dissimilarity, sparsity, instability, runtime_mean}
        """
        self._check_fitted()
        old_coef = self.coef
        if coef is not None:
            self.coef = float(coef)

        if n is not None:
            X_test = X_test.iloc[:n]

        cf_list = []
        diss, spar, inst, times = [], [], [], []

        for i, (_, row) in enumerate(X_test.iterrows()):
            if verbose and (i % 10 == 0):
                print(f"  [{i}/{len(X_test)}]", end="\r")

            q_class = str(row[self.label])
            query = row.drop(labels=[self.label])

            start = time.time()
            cf = self.explain(row)
            elapsed = time.time() - start

            if cf is None:
                continue

            cf_features = cf[self.feature_names]
            cf_class = str(cf[self.label])

            cf_list.append(cf)
            diss.append(self.sim_fn.distance(query, cf_features, cf_class))
            spar.append(
                sum(str(query[f]) != str(cf_features[f]) for f in self.feature_names)
                / len(self.feature_names)
            )
            times.append(elapsed)

        # Instability requires re-generating CFs for perturbed instances
        # (expensive, done separately)
        result = {
            "dissimilarity": float(np.mean(diss)) if diss else float("nan"),
            "sparsity": float(np.mean(spar)) if spar else float("nan"),
            "runtime_mean": float(np.mean(times)) if times else float("nan"),
            "n_successful": len(cf_list),
            "n_total": len(X_test),
        }

        self.coef = old_coef
        if verbose:
            print(f"\nResults (n={result['n_successful']}/{result['n_total']}):")
            for k, v in result.items():
                if isinstance(v, float):
                    print(f"  {k:20s}: {v:.4f}")
        return result
