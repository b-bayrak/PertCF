# PertCF

[![PyPI version](https://img.shields.io/pypi/v/pertcf.svg)](https://pypi.org/project/pertcf/)
[![Python versions](https://img.shields.io/pypi/pyversions/pertcf.svg)](https://pypi.org/project/pertcf/)
[![CI](https://github.com/b-bayrak/PertCF-Explainer/actions/workflows/ci.yml/badge.svg)](https://github.com/b-bayrak/PertCF-Explainer/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-SGAI_2023-blue)](https://doi.org/10.1007/978-3-031-47994-6_13)

**PertCF** generates counterfactual explanations for tabular classification models by iteratively perturbing an instance toward its Nearest Unlike Neighbour (NUN), guided by SHAP feature attributions.

```
"If Leo earned $500 more per month, his loan would be approved."
```

Designed to be **dependency-light**. Works with scikit-learn, PyTorch, Keras, or any callable model.

---

## Install

```bash
pip install pertcf
```

## 30-second example

```python
from pertcf import PertCFExplainer

explainer = PertCFExplainer(
    model=clf,
    X_train=X_train,
    y_train=y_train,
    categorical_features=["purpose", "housing"],
    label="credit_risk",
    num_iter=5,
    coef=5,
)
explainer.fit()

cf = explainer.explain(test_instance)
```

## Key results (from the paper)

| Dataset | Method | Dissimilarity ↓ | Sparsity | Instability ↓ |
|---|---|---|---|---|
| South German Credit | **PertCF** | **0.0517** | 0.7983 | **0.0518** |
| South German Credit | DiCE | 0.0557 | 0.9111 | 0.0560 |
| South German Credit | CF-SHAP | 0.2555 | 0.5842 | 0.2555 |
| User Knowledge Modeling | **PertCF** | **0.0636** | 0.0585 | **0.0664** |
| User Knowledge Modeling | DiCE | 0.1727 | 0.6423 | 0.1769 |
| User Knowledge Modeling | CF-SHAP | 0.1792 | 0.0293 | 0.1806 |

PertCF leads on **dissimilarity** (proximity) and **instability** (stability) across both datasets.

---

## Navigation

- [Installation](install.md): pip, extras, requirements
- [Quickstart](quickstart.md): full working example in under 2 minutes
- [How PertCF Works](concepts.md): algorithm explained visually
- [API Reference](api/explainer.md): full class and function docs
- [Examples](examples/german_credit.md): reproducible benchmark notebooks
- [Citing](citing.md): BibTeX entry
