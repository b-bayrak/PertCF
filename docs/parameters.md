# Parameters

## `PertCFExplainer` constructor

| Parameter | Type | Default | Description                                                   |
|---|---|---|---------------------------------------------------------------|
| `model` | any | required | Trained classifier (sklearn, PyTorch, Keras, or callable)     |
| `X_train` | `pd.DataFrame` | required | Training features - no label column                           |
| `y_train` | array-like | required | Training labels                                               |
| `categorical_features` | `list[str]` | required | Names of categorical (nominal) features                       |
| `label` | `str` | `"label"` | Name of the label column in input/output Series               |
| `num_iter` | `int` | `10` | Max perturbation iterations per CF                            |
| `coef` | `float` | `5.0` | Step-size threshold coefficient                               |
| `shap_values` | `pd.DataFrame` | `None` | Pre-computed SHAP weights - skips SHAP computation in `fit()` |
| `similarity_matrices` | `dict` | `None` | Custom per-feature categorical similarity matrices            |
| `class_names` | `list` | `None` | Class labels (required for PyTorch/Keras models)              |

---

## Choosing `num_iter` and `coef`

These two parameters control the trade-off between quality and runtime.

### Effect of `num_iter`

Higher `num_iter` allows more refinement steps, producing CFs closer to the decision boundary. The paper's results used `num_iter=5` on both datasets.

| `num_iter` | Quality | Runtime |
|---|---|---|
| 3 | Moderate | Fast |
| 5 | **Good** | Moderate |
| 10 | Best | Slower |
| 20+ | Diminishing returns | Slow |

### Effect of `coef`

Controls the termination threshold: `threshold = dist(x, NUN) / coef`.

- **Low `coef` (e.g. 1–2):** Large threshold, terminates early with a coarser CF.
- **High `coef` (e.g. 10–128):** Small threshold, keeps iterating until very fine convergence.
- Beyond `coef=10`, results tend to stabilise (see paper Fig. 5–6).

| `coef` | Convergence | Runtime |
|---|---|---|
| 1–2 | Coarse, fast | Fast |
| 5 | **Balanced** | Moderate |
| 10–20 | Fine | Slower |
| 128+ | Very fine | Slow |

### Recommended settings from the paper

| Dataset type | `num_iter` | `coef` |
|---|---|---|
| Mixed (numeric + categorical) | 5 | 5 |
| Mostly numeric | 5 | 3 |
| When stability matters most | 10 | 5–10 |

---

## `fit()` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `shap_sample_size` | `int` | `300` | Max background samples for SHAP computation |
| `verbose` | `bool` | `True` | Print progress messages |

---

## `explain()` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `instance` | `pd.Series` | required | Instance to explain (must include label column) |
| `target_class` | `str` | `None` | Force a specific target class for the CF |
| `return_all_candidates` | `bool` | `False` | Also return the full candidate list |

---

## `benchmark()` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `X_test` | `pd.DataFrame` | required | Test data with label column |
| `n` | `int` | `None` | Limit to first *n* instances (default = all) |
| `coef` | `float` | `None` | Override `self.coef` for this run |
| `verbose` | `bool` | `True` | Print progress |

---

## Custom similarity matrices

To encode domain knowledge about relationships between categorical values:

```python
explainer = PertCFExplainer(
    model=clf,
    ...
    similarity_matrices={
        "purpose": {
            # (value_a, value_b) → similarity score in [0, 1]
            ("car", "furniture"):   0.7,
            ("car", "education"):   0.2,
            ("education", "repair"): 0.5,
        },
        "housing": {
            ("own", "free"):  0.6,
            ("own", "rent"):  0.3,
        },
    }
)
```

Pairs not listed fall back to: `1.0` if equal, `0.0` otherwise.
