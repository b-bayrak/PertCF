# Installation

## Requirements

- Python ≥ 3.9
- numpy ≥ 1.22
- pandas ≥ 1.4
- scikit-learn ≥ 1.1
- shap ≥ 0.41

## Basic install

```bash
pip install pertcf
```

## With framework adapters

```bash
# PyTorch models
pip install pertcf[torch]

# Keras / TensorFlow models
pip install pertcf[tensorflow]

# Visualisation helpers (matplotlib, seaborn)
pip install pertcf[viz]

# Everything
pip install pertcf[torch,tensorflow,viz]
```

## Development install

```bash
git clone https://github.com/b-bayrak/PertCF-Explainer.git
cd PertCF-Explainer
pip install -e ".[dev]"
pytest tests/ -v
```

## Verify installation

```python
import pertcf
print(pertcf.__version__)
```
