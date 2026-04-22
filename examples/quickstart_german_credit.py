"""
PertCF Quickstart: South German Credit Dataset
==============================================
This example reproduces the benchmark results from Table 2 of the paper:

    Bayrak & Bach (2023). "PertCF: A Perturbation-Based Counterfactual
    Generation Approach." SGAI AI-2023. DOI: 10.1007/978-3-031-47994-6_13

Dataset: South German Credit (Grömping, 2019)
    https://www.kaggle.com/datasets/uciml/german-credit
    1000 instances, 21 features (3 numeric, 18 categorical), binary class.

Run: python quickstart_german_credit.py
     or open as a Jupyter notebook (each section is a cell).
"""

# ── Cell 1: Install & Import ───────────────────────────────────────────────
# pip install pertcf pandas scikit-learn

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pertcf import PertCFExplainer
from pertcf import metrics

# ── Cell 2: Load the South German Credit Dataset ───────────────────────────
# The dataset is freely available from UCI / Kaggle.
# Here we load it directly from a public URL.

url = "https://raw.githubusercontent.com/uci-ml-repo/ucimlrepo/main/datasets/south_german_credit/SouthGermanCredit.asc"

# Fallback: load from local file if URL is unavailable
try:
    df = pd.read_csv(url, sep=" ")
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
except Exception:
    print("Could not fetch from URL. Place SouthGermanCredit.asc locally.")
    raise

# Target column: 'kredit' (1 = good credit, 0 = bad credit)
LABEL = "kredit"
CATEGORICAL_FEATURES = [
    "laufkont", "moral", "verw", "sparkont",
    "beszeit", "rate", "famges", "buerge", "wohnzeit",
    "verm", "weitkred", "wohn", "bishkred", "beruf",
    "pers", "telef", "gastarb",
]

X = df.drop(columns=[LABEL])
y = df[LABEL].astype(str)

# ── Cell 3: Train a Gradient Boosting Classifier ───────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print(f"Classifier accuracy: {acc:.3f}  (paper reports ~0.81)")

# ── Cell 4: Fit the PertCF Explainer ──────────────────────────────────────
explainer = PertCFExplainer(
    model=clf,
    X_train=X_train,
    y_train=y_train,
    categorical_features=CATEGORICAL_FEATURES,
    label=LABEL,
    num_iter=5,    # paper: num_iter=5
    coef=5.0,      # paper: coef=5
)
explainer.fit(verbose=True)

# ── Cell 5: Explain a Single Prediction ───────────────────────────────────
# Pick a test instance predicted as "bad credit" (class 0)
bad_credit_mask = y_test == "0"
instance = X_test[bad_credit_mask].iloc[0].copy()
instance[LABEL] = "0"   # add predicted label

print("\n--- Query instance ---")
print(instance)

cf = explainer.explain(instance)
print("\n--- Counterfactual ---")
print(cf)

# Show what changed
changes = {
    feat: (instance[feat], cf[feat])
    for feat in explainer.feature_names
    if str(instance[feat]) != str(cf[feat])
}
print("\n--- Changed features ---")
for feat, (orig, new) in changes.items():
    print(f"  {feat}: {orig}  →  {new}")

# ── Cell 6: Run the Benchmark (reproduces Table 2) ────────────────────────
print("\n--- Running benchmark (n=50 for speed; use n=None for full) ---")
results = explainer.benchmark(
    X_test.assign(**{LABEL: y_test}),
    n=50,
    coef=5.0,
    verbose=True,
)

print("\n--- Paper Table 2 (South German Credit, PertCF) ---")
print("  dissimilarity : 0.0517  (paper)")
print(f"  dissimilarity : {results['dissimilarity']:.4f}  (reproduced)")
print("  sparsity      : 0.7983  (paper)")
print(f"  sparsity      : {results['sparsity']:.4f}  (reproduced)")

# ── Cell 7: Evaluate Metrics ──────────────────────────────────────────────
# Generate CFs for a batch
test_batch = X_test.assign(**{LABEL: y_test}).iloc[:20]
cfs = []
queries = []
for _, row in test_batch.iterrows():
    cf_i = explainer.explain(row)
    if cf_i is not None:
        queries.append(row)
        cfs.append(cf_i)

if cfs:
    scores = metrics.evaluate(
        queries, cfs, explainer, compute_instability=False
    )
    print("\n--- Evaluation Metrics ---")
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}")
