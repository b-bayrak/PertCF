"""
adapters.py
-----------
Thin wrappers that give PertCF a unified "predict(X) -> array[int/str]"
interface regardless of whether the underlying model is a scikit-learn
estimator, a PyTorch nn.Module, a Keras/TF model, or a plain callable.

Usage
-----
    from pertcf.adapters import wrap_model

    adapter = wrap_model(my_model)
    labels = adapter.predict(X_df)          # returns array of class labels
    probas = adapter.predict_proba(X_df)    # returns 2-D probability array
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, Optional, Union


# Public helper
def wrap_model(
    model,
    class_names: Optional[list] = None,
    predict_fn: Optional[Callable] = None,
    predict_proba_fn: Optional[Callable] = None,
) -> "ModelAdapter":
    """
    Automatically detect the model type and return the appropriate adapter.

    Parameters
    ----------
    model :
        A scikit-learn estimator, PyTorch nn.Module, Keras Model, or any
        callable that accepts a numpy array and returns class labels.
    class_names : list, optional
        Class labels in the same order as the model's output.
        Required for PyTorch / Keras models (inferred from sklearn models).
    predict_fn : callable, optional
        Override the predict function (receives np.ndarray, returns labels).
    predict_proba_fn : callable, optional
        Override the predict_proba function (receives np.ndarray, returns
        2-D probability array).

    Returns
    -------
    ModelAdapter
    """
    if predict_fn is not None:
        return _CallableAdapter(
            model, class_names, predict_fn, predict_proba_fn
        )

    # scikit-learn
    if hasattr(model, "predict") and hasattr(model, "classes_"):
        return _SklearnAdapter(model)

    # PyTorch
    try:
        import torch  # noqa: F401
        import torch.nn as nn
        if isinstance(model, nn.Module):
            if class_names is None:
                raise ValueError(
                    "class_names must be supplied for PyTorch models."
                )
            return _TorchAdapter(model, class_names)
    except ImportError:
        pass

    # Keras / TensorFlow
    try:
        import tensorflow as tf  # noqa: F401
        if hasattr(model, "predict") and hasattr(model, "layers"):
            if class_names is None:
                raise ValueError(
                    "class_names must be supplied for Keras/TF models."
                )
            return _KerasAdapter(model, class_names)
    except ImportError:
        pass

    # Plain callable fallback
    if callable(model):
        if class_names is None:
            raise ValueError(
                "class_names must be supplied for callable models."
            )
        return _CallableAdapter(model, class_names, model, None)

    raise TypeError(
        f"Cannot automatically wrap model of type {type(model)}. "
        "Pass explicit predict_fn / predict_proba_fn."
    )


# Base adapter
class ModelAdapter:
    """Abstract base for model adapters."""

    @property
    def class_names(self) -> list:
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        """Return predicted class labels for X."""
        raise NotImplementedError

    def predict_proba(self, X) -> np.ndarray:
        """Return class probability array of shape (n_samples, n_classes)."""
        raise NotImplementedError

    # Helpers shared by all adapters
    @staticmethod
    def _to_numpy(X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.values
        if isinstance(X, pd.Series):
            return X.values.reshape(1, -1)
        return np.asarray(X)


# scikit-learn adapter
class _SklearnAdapter(ModelAdapter):
    def __init__(self, model):
        self._model = model
        # Store feature names if the model was fitted with named features
        self._feature_names = getattr(model, "feature_names_in_", None)

    @property
    def class_names(self) -> list:
        return list(self._model.classes_)

    def _to_frame(self, X) -> pd.DataFrame:
        """Return a DataFrame, preserving feature names when available."""
        if isinstance(X, pd.DataFrame):
            return X
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if self._feature_names is not None:
            return pd.DataFrame(arr, columns=self._feature_names)
        return pd.DataFrame(arr)

    def predict(self, X) -> np.ndarray:
        return self._model.predict(self._to_frame(X))

    def predict_proba(self, X) -> np.ndarray:
        return self._model.predict_proba(self._to_frame(X))


# PyTorch adapter
class _TorchAdapter(ModelAdapter):
    def __init__(self, model, class_names: list):
        self._model = model
        self._class_names = class_names

    @property
    def class_names(self) -> list:
        return self._class_names

    def predict_proba(self, X) -> np.ndarray:
        import torch
        self._model.eval()
        with torch.no_grad():
            arr = self._to_numpy(X).astype(np.float32)
            tensor = torch.from_numpy(arr)
            logits = self._model(tensor)
            proba = torch.softmax(logits, dim=-1).cpu().numpy()
        return proba

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return np.array([self._class_names[i] for i in indices])


# Keras / TensorFlow adapter
class _KerasAdapter(ModelAdapter):
    def __init__(self, model, class_names: list):
        self._model = model
        self._class_names = class_names

    @property
    def class_names(self) -> list:
        return self._class_names

    def predict_proba(self, X) -> np.ndarray:
        arr = self._to_numpy(X).astype(np.float32)
        return np.array(self._model.predict(arr))

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return np.array([self._class_names[i] for i in indices])


# Callable adapter
class _CallableAdapter(ModelAdapter):
    def __init__(self, model, class_names, predict_fn, predict_proba_fn):
        self._model = model
        self._class_names = class_names or []
        self._predict_fn = predict_fn
        self._predict_proba_fn = predict_proba_fn

    @property
    def class_names(self) -> list:
        return self._class_names

    def predict(self, X) -> np.ndarray:
        return np.asarray(self._predict_fn(self._to_numpy(X)))

    def predict_proba(self, X) -> np.ndarray:
        if self._predict_proba_fn is None:
            raise NotImplementedError(
                "predict_proba_fn was not provided."
            )
        return np.asarray(self._predict_proba_fn(self._to_numpy(X)))
