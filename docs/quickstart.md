# Quickstart

This page walks through a complete example using the South German Credit dataset, the same dataset used in the paper.

## 1. Install

```bash
pip install pertcf
```

## 2. Load data and train a model

```python
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("german_credit.csv")
X = df.drop(columns=["credit_risk"])
y = df["credit_risk"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
clf = GradientBoostingClassifier(random_state=42).fit(X_train, y_train)
print(f"Accuracy: {clf.score(X_test, y_test):.3f}")
```

## 3. Fit the explainer

```python
from pertcf import PertCFExplainer

explainer = PertCFExplainer(
    model=clf,
    X_train=X_train,
    y_train=y_train,
    categorical_features=[
        "purpose", "personal_status", "housing",
        "savings_account", "employment", "other_debtors",
        "property", "other_installment_plans", "telephone", "foreign_worker"
    ],
    label="credit_risk",
    num_iter=5,
    coef=5.0,
)
explainer.fit()
```

Output:
```
Computing SHAP values… (this may take a moment)
- SHAP values computed.
- Explainer fitted and ready.
```

## 4. Explain a prediction

```python
# Get a test instance predicted as "bad" credit
bad_instances = X_test[y_test == "0"]
instance = bad_instances.iloc[0].copy()
instance["credit_risk"] = "0"

cf = explainer.explain(instance)
print(cf)
```

```
credit_amount     2500.0       # was 4200.0, lower loan amount
duration           12.0        # was 24.0, shorter duration
credit_risk           1        # ← flipped to "good credit"
...
```

## 5. See what changed

```python
for feat in explainer.feature_names:
    orig = instance[feat]
    new  = cf[feat]
    if str(orig) != str(new):
        print(f"  {feat}: {orig}  →  {new}")
```

```
  credit_amount : 4200.0  →  2500.0
  duration      : 24      →  12
  savings_account: A65    →  A61
```

## 6. Multi-class: CF to every class

For multi-class problems, you can request a CF to each target class:

```python
for target in explainer.class_names:
    if target == instance["label"]:
        continue
    cf_t = explainer.explain(instance, target_class=target)
    if cf_t is not None:
        d = explainer.sim_fn.distance(
            instance[explainer.feature_names],
            cf_t[explainer.feature_names],
            target,
        )
        print(f"  → class '{target}'  dissimilarity={d:.4f}")
```

## 7. Batch + metrics

```python
from pertcf import metrics

test_with_labels = X_test.assign(credit_risk=y_test)
results = explainer.benchmark(test_with_labels, n=100, verbose=True)
```

```
Results (n=97/100):
  dissimilarity       : 0.0522
  sparsity            : 0.8012
  runtime_mean        : 0.4103
```

## Next steps

- [Parameters guide](parameters.md): tune `num_iter` and `coef`
- [Custom similarity matrices](examples/custom_similarity.md): add domain knowledge
- [PyTorch / Keras example](examples/german_credit.md#pytorch-adapter)
- [API Reference](api/explainer.md)
