# PertCF

[![PyPI version](https://img.shields.io/pypi/v/pertcf.svg)](https://pypi.org/project/pertcf/)
[![Python versions](https://img.shields.io/pypi/pyversions/pertcf.svg)](https://pypi.org/project/pertcf/)
[![CI](https://github.com/b-bayrak/PertCF/actions/workflows/ci.yml/badge.svg)](https://github.com/b-bayrak/PertCF/actions/runs/24771758700)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-SGAI_2023-blue)](https://doi.org/10.1007/978-3-031-47994-6_13)

**PertCF** is a perturbation-based counterfactual explanation method that combines SHAP feature attribution with nearest-neighbour search to generate high-quality, stable counterfactuals for tabular classification models.

> **What is a counterfactual explanation?**  
> Given a model's prediction for an instance *x*, a counterfactual *x'* is the minimal change to *x* that would flip the prediction. For example: *"If Leo earned $500 more per month, his loan application would be accepted."*

---

## Why PertCF?

| Feature | PertCF | DiCE | CF-SHAP |
|---|---|---|---|
| Multi-class support | ✅ | ❌ | ❌ |
| SHAP-weighted distance | ✅ | ❌ | Partial |
| Custom domain knowledge | ✅ | ❌ | ❌ |
| Works with sklearn, PyTorch, Keras | ✅ | Partial | ❌ |
| No external server needed | ✅ | ✅ | ✅ |

PertCF outperforms DiCE and CF-SHAP on **dissimilarity** and **instability** across both benchmark datasets (South German Credit, User Knowledge Modeling). See the [paper](https://doi.org/10.1007/978-3-031-47994-6_13) for full results.

---

## Installation

```bash
pip install pertcf
```

For PyTorch or Keras model support:

```bash
pip install pertcf[torch]      # + PyTorch adapter
pip install pertcf[tensorflow] # + Keras/TF adapter
pip install pertcf[viz]        # + matplotlib/seaborn for plots
```

**Requirements:** Python ≥ 3.9, numpy, pandas, scikit-learn, shap.  
No Java. No REST server. No external frameworks.

---

## Quick Start (30 seconds)

```python
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from pertcf import PertCFExplainer

# 1. Load data and train a model (example: German Credit dataset)
df = pd.read_csv("german_credit.csv")
X = df.drop(columns=["credit_risk"])
y = df["credit_risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = GradientBoostingClassifier(random_state=42).fit(X_train, y_train)

# 2. Create and fit the explainer
explainer = PertCFExplainer(
    model=clf,
    X_train=X_train,
    y_train=y_train,
    categorical_features=["purpose", "personal_status", "housing"],
    label="credit_risk",
    num_iter=5,
    coef=5,
)
explainer.fit()

# 3. Explain a prediction
instance = X_test.iloc[0].copy()
instance["credit_risk"] = clf.predict(X_test.iloc[[0]])[0]

counterfactual = explainer.explain(instance)
print("Original:       ", instance.to_dict())
print("Counterfactual: ", counterfactual.to_dict())
```

---

## Feature Highlights

### Works with any classifier

```python
# scikit-learn (auto-detected)
from sklearn.ensemble import RandomForestClassifier
explainer = PertCFExplainer(model=RandomForestClassifier().fit(X, y), ...)

# PyTorch
from pertcf import PertCFExplainer
explainer = PertCFExplainer(
    model=my_torch_model,
    class_names=["bad", "good"],
    ...
)

# Keras / TensorFlow
explainer = PertCFExplainer(
    model=my_keras_model,
    class_names=["bad", "good"],
    ...
)

# Any callable
explainer = PertCFExplainer(
    model=my_model,
    predict_fn=lambda X: my_model.predict(X),
    predict_proba_fn=lambda X: my_model.predict_proba(X),
    class_names=["bad", "good"],
    ...
)
```

### Domain knowledge via custom similarity matrices

```python
# Model the relationship between credit purposes
explainer = PertCFExplainer(
    model=clf,
    ...
    similarity_matrices={
        "purpose": {
            ("car", "furniture"): 0.7,   # similar purposes
            ("car", "education"): 0.2,   # less similar
        }
    }
)
```

### Pre-computed SHAP values (for large datasets)

```python
import shap
# Compute SHAP once, reuse across experiments
shap_exp = shap.TreeExplainer(clf)
shap_vals = shap_exp.shap_values(X_train)
# … build shap_df with shape (n_classes, n_features) …

explainer = PertCFExplainer(
    model=clf, shap_values=shap_df, ...
)
explainer.fit()  # skips SHAP computation
```

### Built-in benchmark

```python
# Reproduce the paper's Table 2 results
results = explainer.benchmark(X_test, n=100, coef=5, verbose=True)
# Results (n=100/100):
#   dissimilarity       : 0.0517
#   sparsity            : 0.7983
#   runtime_mean        : 0.4069
```

### Evaluation metrics

```python
from pertcf import metrics

print(metrics.dissimilarity(query, cf, explainer.sim_fn, cf_class))
print(metrics.sparsity(query, cf))
print(metrics.instability(query, cf, explainer))

# All at once:
results = metrics.evaluate(queries, counterfactuals, explainer)
```

---

## How PertCF Works

```
1. Compute SHAP values per class → class-specific feature importance weights
2. For query x:
   a. Find Nearest Unlike Neighbour (NUN) using SHAP-weighted similarity
   b. Perturb x toward NUN using SHAP weights:
      - Numeric:     p_f = x_f + shap_target_f * (nun_f - x_f)
      - Categorical: p_f = nun_f  if sim(x_f, nun_f) < 0.5  else  x_f
   c. If perturbed instance flips class → refine (approach source)
   d. If not → push harder (approach target)
   e. Terminate when step size < threshold or max iterations reached
```

See [the paper](https://doi.org/10.1007/978-3-031-47994-6_13) for full algorithmic details.

---

## Examples

| Notebook | Dataset | Description |
|---|---|---|
| [quickstart_german_credit.ipynb](examples/quickstart_german_credit.ipynb) | South German Credit | Basic usage, benchmark, comparison to DiCE |
| [quickstart_knowledge.ipynb](examples/quickstart_knowledge.ipynb) | User Knowledge Modeling | Multi-class classification |
| [custom_similarity.ipynb](examples/custom_similarity.ipynb) | German Credit | Domain knowledge with custom similarity matrices |

Launch in Colab: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/b-bayrak/PertCF-Explainer/blob/main/examples/quickstart_german_credit.ipynb)

---

## Parameter Guide

| Parameter | Default | Description |
|---|---|---|
| `num_iter` | `10` | Max perturbation iterations. Higher → better quality, slower. |
| `coef` | `5` | Step-size threshold coefficient. Higher → finer convergence. |

**Recommended settings from the paper:**

| Dataset | `num_iter` | `coef` | Notes |
|---|---|---|---|
| South German Credit | 5 | 5 | Most categorical features |
| User Knowledge Modeling | 5 | 3 | All numeric features |

---

## Citation

If you use PertCF in your research, please cite:

```bibtex
@inproceedings{bayrak2023pertcf,
  title     = {PertCF: A Perturbation-Based Counterfactual Generation Approach},
  author    = {Bayrak, Bet{\"u}l and Bach, Kerstin},
  booktitle = {Artificial Intelligence XXXVIII},
  series    = {Lecture Notes in Computer Science},
  volume    = {14381},
  pages     = {174--187},
  year      = {2023},
  publisher = {Springer, Cham},
  doi       = {10.1007/978-3-031-47994-6_13}
}
```

---

## License

MIT © Betül Bayrak

This work was supported by the Research Council of Norway through the EXAIGON project (ID 304843).
