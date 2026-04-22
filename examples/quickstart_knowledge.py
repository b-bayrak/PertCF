"""
PertCF Example: User Knowledge Modeling Dataset (Multi-class)

Reproduces the benchmark results from Table 2 of the paper for the
User Knowledge Modeling dataset (Kahraman et al., 2013):

    403 instances, 5 numeric features, 4 classes
    Gradient Boosting accuracy: ~0.98

Paper settings: num_iter=5, coef=3
"""

# ── Cell 1: Install & Import ───────────────────────────────────────────────
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from pertcf import PertCFExplainer
from pertcf import metrics

# ── Cell 2: Load the Dataset ───────────────────────────────────────────────
url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00257/User%20Knowledge%20Modeling.csv"
)

try:
    df = pd.read_csv(url)
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Columns:", df.columns.tolist())
except Exception:
    # Fallback: construct small synthetic data matching paper characteristics
    print("Using synthetic data (5 numeric features, 4 classes).")
    np.random.seed(42)
    n = 403
    df = pd.DataFrame({
        "STG":  np.random.uniform(0, 1, n),
        "SCG":  np.random.uniform(0, 1, n),
        "STR":  np.random.uniform(0, 1, n),
        "LPR":  np.random.uniform(0, 1, n),
        "PEG":  np.random.uniform(0, 1, n),
        "UNS":  np.random.choice(["very_low", "low", "middle", "high"], n),
    })

LABEL = "UNS"
CATEGORICAL_FEATURES = []   # all features are numeric

X = df.drop(columns=[LABEL])
y = df[LABEL].astype(str)

# ── Cell 3: Train Model ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
print(f"Accuracy: {clf.score(X_test, y_test):.3f}  (paper reports ~0.98)")
print(f"Classes:  {clf.classes_.tolist()}")

# ── Cell 4: Fit Explainer ──────────────────────────────────────────────────
explainer = PertCFExplainer(
    model=clf,
    X_train=X_train,
    y_train=y_train,
    categorical_features=CATEGORICAL_FEATURES,
    label=LABEL,
    num_iter=5,    # paper: num_iter=5
    coef=3.0,      # paper: coef=3
)
explainer.fit(verbose=True)

# ── Cell 5: Multi-class CF for a single instance ───────────────────────────
instance = X_test.iloc[0].copy()
instance[LABEL] = clf.predict(X_test.iloc[[0]])[0]
print(f"\nQuery class: {instance[LABEL]}")

cf = explainer.explain(instance)
if cf is not None:
    print(f"CF class:    {cf[LABEL]}")
    changed = {
        f: (round(float(instance[f]), 4), round(float(cf[f]), 4))
        for f in explainer.feature_names
        if abs(float(instance[f]) - float(cf[f])) > 1e-6
    }
    print("Changed features:", changed)

# ── Cell 6: Generate CFs to all other classes (multi-class insight) ────────
print("\n--- CFs to all target classes ---")
for target in explainer.class_names:
    if target == str(instance[LABEL]):
        continue
    cf_t = explainer.explain(instance, target_class=target)
    if cf_t is not None:
        d = explainer.sim_fn.distance(
            instance[explainer.feature_names],
            cf_t[explainer.feature_names],
            target
        )
        print(f"  To class '{target}': dissimilarity={d:.4f}")

# ── Cell 7: Benchmark ─────────────────────────────────────────────────────
print("\n--- Running benchmark ---")
results = explainer.benchmark(
    X_test.assign(**{LABEL: y_test}),
    n=40,
    verbose=True,
)

print("\n--- Paper Table 2 (User Knowledge Modeling, PertCF) ---")
print("  dissimilarity : 0.0636  (paper)")
print(f"  dissimilarity : {results['dissimilarity']:.4f}  (reproduced)")
print("  sparsity      : 0.0585  (paper)")
print(f"  sparsity      : {results['sparsity']:.4f}  (reproduced)")
