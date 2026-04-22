# Example: Domain Knowledge via Custom Similarity

One of PertCF's distinguishing features is the ability to encode domain knowledge directly into the similarity function. This controls how the method perceives distances between categorical feature values.

## The problem

By default, categorical similarity is binary:

```
sim("car", "car")       = 1.0
sim("car", "education") = 0.0
```

But in reality, *car (new)* and *car (used)* are much more similar than *car* and *education*. Without this knowledge, PertCF might suggest an unhelpful purpose change.

## The solution: custom similarity matrices

```python
from pertcf import PertCFExplainer

explainer = PertCFExplainer(
    model=clf,
    ...
    similarity_matrices={
        "purpose": {
            ("car_new",   "car_used"):   0.90,  # very similar
            ("car_new",   "furniture"):  0.30,
            ("furniture", "appliances"): 0.75,  # both consumer goods
            ("education", "retraining"): 0.60,  # both knowledge-related
        },
        "housing": {
            ("own",  "free"):  0.60,
            ("own",  "rent"):  0.30,
            ("free", "rent"):  0.50,
        },
    }
)
```

## Format

```python
similarity_matrices = {
    "feature_name": {
        (value_a, value_b): similarity_score,   # float in [0, 1]
        ...
    },
    ...
}
```

- Scores are symmetric: `(a, b)` and `(b, a)` are equivalent.
- Pairs not listed fall back to: `1.0` if equal, `0.0` otherwise.

## Running the example

```bash
python examples/custom_similarity.py
```

## What to expect

With domain knowledge:
- The similarity function **partially credits** changes between related categories, making such changes less costly.
- CFs are more likely to suggest changes that are **semantically meaningful**.
- Mean dissimilarity typically decreases compared to the binary baseline.
