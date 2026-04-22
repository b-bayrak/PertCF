"""
PertCF Example: PyTorch Model Adapter

Demonstrates using PertCF with a PyTorch neural network classifier.
Uses the South German Credit dataset with a simple MLP.

Requirements:
    pip install pertcf[torch]
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pertcf import PertCFExplainer

# ── Load and encode data ───────────────────────────────────────────────────
url = "https://raw.githubusercontent.com/uci-ml-repo/ucimlrepo/main/datasets/south_german_credit/SouthGermanCredit.asc"
try:
    df = pd.read_csv(url, sep=" ")
except Exception:
    raise RuntimeError("Download SouthGermanCredit.asc first.")

LABEL = "kredit"
CATEGORICAL_FEATURES = [
    "laufkont", "moral", "verw", "sparkont", "beszeit", "rate",
    "famges", "buerge", "wohnzeit", "verm", "weitkred", "wohn",
    "bishkred", "beruf", "pers", "telef", "gastarb",
]

X = df.drop(columns=[LABEL])
y = df[LABEL].astype(str)

# Keep original X for PertCF (categorical columns as-is)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Encode categoricals numerically for PyTorch
X_enc = X.copy()
le = LabelEncoder()
for col in CATEGORICAL_FEATURES:
    X_enc[col] = le.fit_transform(X_enc[col].astype(str))

X_train_enc, X_test_enc = X_enc.iloc[X_train_raw.index], X_enc.iloc[X_test_raw.index]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_enc)
X_test_scaled  = scaler.transform(X_test_enc)

# ── Define and train a simple MLP ─────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.net(x)

n_classes = len(y.unique())
model = MLP(in_features=X_train_scaled.shape[1], n_classes=n_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

class_names = sorted(y.unique().tolist())
label_to_idx = {c: i for i, c in enumerate(class_names)}

y_train_idx = torch.tensor([label_to_idx[c] for c in y_train], dtype=torch.long)
X_train_t   = torch.from_numpy(X_train_scaled.astype(np.float32))

# Train for 50 epochs
model.train()
for epoch in range(50):
    optimizer.zero_grad()
    loss = criterion(model(X_train_t), y_train_idx)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    preds = model(torch.from_numpy(X_test_scaled.astype(np.float32)))
    acc = (preds.argmax(1).numpy() == np.array([label_to_idx[c] for c in y_test])).mean()
    print(f"PyTorch MLP accuracy: {acc:.3f}")

# ── Wrap PyTorch model with a custom predict_fn ───────────────────────────
# The model expects scaled numpy, but PertCF works with raw (unscaled)
# feature values from X_train_raw. We need a wrapper that handles encoding.

label_encoders = {}
for col in CATEGORICAL_FEATURES:
    _le = LabelEncoder()
    _le.fit(X[col].astype(str))
    label_encoders[col] = _le

def encode_and_scale(X_df: pd.DataFrame) -> np.ndarray:
    X_e = X_df.copy()
    for col in CATEGORICAL_FEATURES:
        X_e[col] = label_encoders[col].transform(X_e[col].astype(str))
    return scaler.transform(X_e.astype(float))

def predict_fn(X_arr: np.ndarray) -> np.ndarray:
    X_df = pd.DataFrame(X_arr, columns=X_train_raw.columns)
    X_s  = encode_and_scale(X_df)
    with torch.no_grad():
        logits = model(torch.from_numpy(X_s.astype(np.float32)))
        indices = logits.argmax(1).numpy()
    return np.array([class_names[i] for i in indices])

def predict_proba_fn(X_arr: np.ndarray) -> np.ndarray:
    X_df = pd.DataFrame(X_arr, columns=X_train_raw.columns)
    X_s  = encode_and_scale(X_df)
    with torch.no_grad():
        logits = model(torch.from_numpy(X_s.astype(np.float32)))
        return torch.softmax(logits, dim=1).numpy()

# ── Fit PertCF with the PyTorch model ─────────────────────────────────────
explainer = PertCFExplainer(
    model=model,
    X_train=X_train_raw,
    y_train=y_train,
    categorical_features=CATEGORICAL_FEATURES,
    label=LABEL,
    class_names=class_names,
    num_iter=5,
    coef=5.0,
    # Pass explicit fns so encoding happens transparently
    predict_fn=predict_fn,
    predict_proba_fn=predict_proba_fn,
)
explainer.fit(verbose=True)

# ── Explain a prediction ───────────────────────────────────────────────────
bad_instances = X_test_raw[y_test == "0"].assign(**{LABEL: "0"})
instance = bad_instances.iloc[0]

cf = explainer.explain(instance)
if cf is not None:
    changes = {
        f: (instance[f], cf[f])
        for f in explainer.feature_names
        if str(instance[f]) != str(cf[f])
    }
    print("\n--- Counterfactual changes ---")
    for feat, (orig, new) in changes.items():
        print(f"  {feat}: {orig}  →  {new}")
    print(f"  {LABEL}: {instance[LABEL]}  →  {cf[LABEL]}")
