"""
tests/test_pertcf.py
--------------------
Unit tests for the PertCF package.

Run with:  pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification



# Fixtures

@pytest.fixture(scope="module")
def synthetic_data():
    """Small synthetic binary classification dataset."""
    X, y = make_classification(
        n_samples=200, n_features=5, n_informative=4,
        n_redundant=1, random_state=42
    )
    feature_names = [f"feat_{i}" for i in range(5)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y.astype(str))
    return X_df, y_series, feature_names


@pytest.fixture(scope="module")
def fitted_explainer(synthetic_data):
    from pertcf import PertCFExplainer

    X_df, y_series, feature_names = synthetic_data
    clf = GradientBoostingClassifier(n_estimators=20, random_state=42)
    clf.fit(X_df, y_series)

    explainer = PertCFExplainer(
        model=clf,
        X_train=X_df,
        y_train=y_series,
        categorical_features=[],   # all numeric for simplicity
        label="label",
        num_iter=5,
        coef=5.0,
    )
    explainer.fit(shap_sample_size=50, verbose=False)
    return explainer, X_df, y_series, clf


# Similarity tests

class TestSHAPWeightedSimilarity:
    def test_identical_instances(self, fitted_explainer):
        explainer, X_df, y_series, _ = fitted_explainer
        row = X_df.iloc[0]
        sim = explainer.sim_fn.similarity(row, row, class_label="0")
        assert sim == pytest.approx(1.0, abs=1e-6)

    def test_distance_range(self, fitted_explainer):
        explainer, X_df, y_series, _ = fitted_explainer
        r1 = X_df.iloc[0]
        r2 = X_df.iloc[1]
        dist = explainer.sim_fn.distance(r1, r2, class_label="0")
        assert 0.0 <= dist <= 1.0

    def test_categorical_sim_exact_match(self):
        from pertcf.similarity import SHAPWeightedSimilarity
        import pandas as pd

        weights = pd.DataFrame([[0.5, 0.5]], index=["A"], columns=["x", "y"])
        sim_fn = SHAPWeightedSimilarity(
            feature_names=["x", "y"],
            categorical_features=["x"],
            shap_weights=weights,
            feature_ranges={"x": 1.0, "y": 1.0},
        )
        s1 = pd.Series({"x": "cat1", "y": 0.0})
        s2 = pd.Series({"x": "cat1", "y": 0.0})
        assert sim_fn.feature_similarity("x", "cat1", "cat1") == pytest.approx(1.0)
        assert sim_fn.feature_similarity("x", "cat1", "cat2") == pytest.approx(0.0)


# Core explainer tests

class TestPertCFExplainer:
    def test_fit_sets_ready(self, fitted_explainer):
        explainer, *_ = fitted_explainer
        assert explainer._is_fitted

    def test_shap_weights_shape(self, fitted_explainer):
        explainer, X_df, y_series, _ = fitted_explainer
        n_classes = len(explainer.class_names)
        n_features = len(explainer.feature_names)
        assert explainer._shap_weights.shape == (n_classes, n_features)

    def test_shap_weights_normalised(self, fitted_explainer):
        explainer, *_ = fitted_explainer
        row_sums = explainer._shap_weights.sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-6)

    def test_explain_returns_series(self, fitted_explainer):
        explainer, X_df, y_series, clf = fitted_explainer
        row = X_df.iloc[0].copy()
        row["label"] = clf.predict(X_df.iloc[[0]])[0]
        cf = explainer.explain(row)
        assert cf is not None
        assert isinstance(cf, pd.Series)

    def test_cf_has_label(self, fitted_explainer):
        explainer, X_df, y_series, clf = fitted_explainer
        row = X_df.iloc[0].copy()
        row["label"] = clf.predict(X_df.iloc[[0]])[0]
        cf = explainer.explain(row)
        assert "label" in cf.index

    def test_cf_different_class(self, fitted_explainer):
        """CF must belong to a different class than the query."""
        explainer, X_df, y_series, clf = fitted_explainer
        row = X_df.iloc[0].copy()
        q_class = clf.predict(X_df.iloc[[0]])[0]
        row["label"] = q_class
        cf = explainer.explain(row)
        if cf is not None:
            assert str(cf["label"]) != str(q_class)

    def test_not_fitted_raises(self):
        from pertcf import PertCFExplainer
        import pandas as pd
        X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        y = pd.Series(["0", "1"])
        from sklearn.dummy import DummyClassifier
        clf = DummyClassifier().fit(X, y)
        explainer = PertCFExplainer(clf, X, y, [], label="label")
        with pytest.raises(RuntimeError, match="not fitted"):
            explainer.explain(X.iloc[0])


# Adapter tests

class TestAdapters:
    def test_sklearn_adapter(self, synthetic_data):
        from pertcf.adapters import wrap_model
        X_df, y_series, _ = synthetic_data
        clf = GradientBoostingClassifier(n_estimators=10, random_state=42)
        clf.fit(X_df, y_series)
        adapter = wrap_model(clf)
        preds = adapter.predict(X_df)
        assert len(preds) == len(X_df)
        probas = adapter.predict_proba(X_df)
        assert probas.shape == (len(X_df), 2)

    def test_callable_adapter(self, synthetic_data):
        from pertcf.adapters import wrap_model
        X_df, y_series, feature_names = synthetic_data
        clf = GradientBoostingClassifier(n_estimators=10, random_state=42)
        clf.fit(X_df, y_series)

        # Callable adapter: user is responsible for encoding; we use a
        # DataFrame-aware lambda to match real usage patterns.
        def _pred(x):
            df = pd.DataFrame(x, columns=feature_names)
            return clf.predict(df)

        def _proba(x):
            df = pd.DataFrame(x, columns=feature_names)
            return clf.predict_proba(df)

        adapter = wrap_model(
            clf,
            class_names=["0", "1"],
            predict_fn=_pred,
            predict_proba_fn=_proba,
        )
        preds = adapter.predict(X_df)
        assert len(preds) == len(X_df)


# Metrics tests

class TestMetrics:
    def test_sparsity_identical(self):
        from pertcf.metrics import sparsity
        s = pd.Series({"a": 1, "b": 2, "c": 3})
        assert sparsity(s, s) == pytest.approx(0.0)

    def test_sparsity_all_different(self):
        from pertcf.metrics import sparsity
        s1 = pd.Series({"a": 1, "b": 2})
        s2 = pd.Series({"a": 9, "b": 9})
        assert sparsity(s1, s2) == pytest.approx(1.0)

    def test_dissimilarity_identical(self, fitted_explainer):
        from pertcf.metrics import dissimilarity
        explainer, X_df, *_ = fitted_explainer
        row = X_df.iloc[0]
        d = dissimilarity(row, row, explainer.sim_fn, "0")
        assert d == pytest.approx(0.0, abs=1e-6)
