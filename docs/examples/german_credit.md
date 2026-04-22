# Example: South German Credit Dataset

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/b-bayrak/PertCF-Explainer/blob/main/examples/quickstart_german_credit.ipynb)

This example reproduces **Table 2** from the paper using the South German Credit dataset.

**Dataset characteristics:**

- 1,000 instances
- 21 features: 3 numeric, 18 categorical
- Binary classification: good credit (1) vs bad credit (0)
- Gradient Boosting accuracy: ~0.81

## Running the example

```bash
pip install pertcf
python examples/quickstart_german_credit.py
```

## Expected output

```
Classifier accuracy: 0.810  (paper reports ~0.81)
- SHAP values computed.
- Explainer fitted and ready.

--- Query instance ---
laufkont            2
...
credit_risk         0   ← bad credit

--- Counterfactual ---
laufkont            3   ← changed
...
credit_risk         1   ← good credit 

--- Changed features ---
  laufkont: 2  →  3
  laufzeit: 24  →  12

--- Results (n=50/50) ---
  dissimilarity       : 0.0522
  sparsity            : 0.7991
  runtime_mean        : 0.4089
```

## Comparing to paper Table 2

| Metric | Paper (PertCF) | Reproduced |
|---|---|---|
| Dissimilarity | 0.0517 | ~0.052 |
| Sparsity | 0.7983 | ~0.799 |

Minor differences are expected due to random train/test splits.

## PyTorch adapter

See [`examples/pytorch_adapter.py`](https://github.com/b-bayrak/PertCF-Explainer/blob/main/examples/pytorch_adapter.py) for a complete example using a PyTorch MLP on the same dataset.

## Domain knowledge

See [`examples/custom_similarity.py`](https://github.com/b-bayrak/PertCF-Explainer/blob/main/examples/custom_similarity.py) for how to inject custom similarity matrices for the loan purpose feature.
