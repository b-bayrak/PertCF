"""
PertCF Example: Domain Knowledge via Custom Similarity Matrices

PertCF lets you inject expert knowledge into the distance function by
providing custom similarity scores between categorical values.

This matters because the default similarity (0 if different, 1 if equal)
treats all categorical mismatches as equally distant. In practice, some
values are semantically closer than others.

Example: Loan purpose
  "car (new)"  vs  "car (used)"  →  very similar (0.9)
  "car (new)"  vs  "education"   →  less similar  (0.2)

This example uses the South German Credit dataset.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from pertcf import PertCFExplainer

# ── Load data ──────────────────────────────────────────────────────────────
url = "https://raw.githubusercontent.com/uci-ml-repo/ucimlrepo/main/datasets/south_german_credit/SouthGermanCredit.asc"
try:
    df = pd.read_csv(url, sep=" ")
except Exception:
    raise RuntimeError("Please download SouthGermanCredit.asc and place it locally.")

LABEL = "kredit"
CATEGORICAL_FEATURES = [
    "laufkont", "moral", "verw", "sparkont", "beszeit", "rate",
    "famges", "buerge", "wohnzeit", "verm", "weitkred", "wohn",
    "bishkred", "beruf", "pers", "telef", "gastarb",
]

X = df.drop(columns=[LABEL])
y = df[LABEL].astype(str)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = GradientBoostingClassifier(random_state=42).fit(X_train, y_train)

# ── Define custom similarity matrices ─────────────────────────────────────
# 'verw' = purpose of the loan (encoded as integers in this dataset)
# Encoding: 0=car(new), 1=car(used), 2=furniture, 3=TV, 4=appliances,
#           5=repairs, 6=education, 7=vacation, 8=retraining, 9=business
#
# Domain knowledge: new vs used car are similar; consumer goods are similar

purpose_similarity = {
    # car new vs car used — very similar
    (0, 1): 0.9,
    # consumer goods cluster
    (2, 3): 0.7,   # furniture vs TV
    (2, 4): 0.7,   # furniture vs appliances
    (3, 4): 0.8,   # TV vs appliances
    # repairs is somewhat similar to business improvements
    (5, 9): 0.4,
    # education vs retraining — related
    (6, 8): 0.6,
}

# 'laufkont' = checking account status (0=none, 1=<0, 2=0-200, 3=200+)
# Nearby values are more similar
checking_similarity = {
    (0, 1): 0.6,
    (1, 2): 0.7,
    (2, 3): 0.7,
    (0, 2): 0.3,
    (1, 3): 0.3,
    (0, 3): 0.1,
}

similarity_matrices = {
    "verw": purpose_similarity,
    "laufkont": checking_similarity,
}

# ── Fit without custom similarity (baseline) ──────────────────────────────
print("=== Baseline (no custom similarity) ===")
explainer_base = PertCFExplainer(
    model=clf, X_train=X_train, y_train=y_train,
    categorical_features=CATEGORICAL_FEATURES,
    label=LABEL, num_iter=5, coef=5,
)
explainer_base.fit(verbose=False)

# ── Fit with custom similarity ─────────────────────────────────────────────
print("\n=== With custom similarity matrices ===")
explainer_domain = PertCFExplainer(
    model=clf, X_train=X_train, y_train=y_train,
    categorical_features=CATEGORICAL_FEATURES,
    label=LABEL, num_iter=5, coef=5,
    similarity_matrices=similarity_matrices,
)
explainer_domain.fit(verbose=False)

# ── Compare CFs on a few instances ────────────────────────────────────────
bad_instances = X_test[y_test == "0"].assign(**{LABEL: "0"})

print("\n--- Comparison: purpose feature changes ---")
print(f"{'Instance':>5}  {'Baseline purpose change':>25}  {'Domain purpose change':>22}")
print("-" * 60)

for i in range(min(5, len(bad_instances))):
    inst = bad_instances.iloc[i]

    cf_base   = explainer_base.explain(inst)
    cf_domain = explainer_domain.explain(inst)

    orig_purpose = inst["verw"]
    base_change  = f"{orig_purpose}→{cf_base['verw']}"   if cf_base   is not None else "N/A"
    dom_change   = f"{orig_purpose}→{cf_domain['verw']}" if cf_domain is not None else "N/A"

    same = "=" if base_change == dom_change else "≠"
    print(f"  [{i}]    {base_change:>25}    {dom_change:>22}  {same}")

# ── Dissimilarity comparison ───────────────────────────────────────────────
diss_base, diss_dom = [], []
n_compare = 30
test_sample = bad_instances.head(n_compare)

for _, inst in test_sample.iterrows():
    cf_b = explainer_base.explain(inst)
    cf_d = explainer_domain.explain(inst)
    q    = inst[explainer_base.feature_names]

    if cf_b is not None:
        diss_base.append(
            explainer_base.sim_fn.distance(q, cf_b[explainer_base.feature_names], str(cf_b[LABEL]))
        )
    if cf_d is not None:
        diss_dom.append(
            explainer_domain.sim_fn.distance(q, cf_d[explainer_domain.feature_names], str(cf_d[LABEL]))
        )

print(f"\nMean dissimilarity over {n_compare} instances:")
print(f"  Baseline  : {np.mean(diss_base):.4f}")
print(f"  Domain    : {np.mean(diss_dom):.4f}")
print("\nDomain knowledge typically reduces dissimilarity by allowing")
print("semantically close categories to be treated as partial matches.")
