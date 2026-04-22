# Example: User Knowledge Modeling Dataset

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/b-bayrak/PertCF-Explainer/blob/main/examples/quickstart_knowledge.ipynb)

Multi-class counterfactual explanations using the User Knowledge Modeling dataset.

**Dataset characteristics:**

- 403 instances
- 5 numeric features
- 4 classes: very_low, low, middle, high
- Gradient Boosting accuracy: ~0.98

## Running the example

```bash
pip install pertcf
python examples/quickstart_knowledge.py
```

## Expected output

```
Accuracy: 0.975  (paper reports ~0.98)
Classes:  ['High', 'Low', 'Middle', 'very_low']

- SHAP values computed.
- Explainer fitted and ready.

Query class: Low

--- CFs to all target classes ---
  To class 'High'     : dissimilarity=0.0721
  To class 'Middle'   : dissimilarity=0.0412
  To class 'very_low' : dissimilarity=0.0318

--- Results (n=40/40) ---
  dissimilarity       : 0.0641
  sparsity            : 0.0592
  runtime_mean        : 0.2803
```

## Comparing to paper Table 2

| Metric | Paper (PertCF) | Reproduced |
|---|---|---|
| Dissimilarity | 0.0636 | ~0.064 |
| Sparsity | 0.0585 | ~0.059 |

## Multi-class CF generation

PertCF natively supports multi-class problems, pass `target_class` to generate a CF for any desired outcome class:

```python
cf_to_high = explainer.explain(instance, target_class="High")
cf_to_low  = explainer.explain(instance, target_class="Low")
```
