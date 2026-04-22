# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [1.0.0] — 2024

### Added
- Complete rewrite with no external server dependencies (myCBR removed)
- `PertCFExplainer` class with `fit()`, `explain()`, `explain_batch()`, `benchmark()`
- `SHAPWeightedSimilarity` — pure NumPy/pandas implementation of SHAP-weighted similarity and distance
- Model adapters: scikit-learn (auto-detected), PyTorch (`nn.Module`), Keras/TF, callable
- `metrics` module: `dissimilarity`, `sparsity`, `instability`, `evaluate()`
- Custom categorical similarity matrices for domain knowledge injection
- Pre-computed SHAP values support (skip SHAP computation in `fit()`)
- `target_class` argument in `explain()` for directed CF generation
- `explain_batch()` for generating CFs over an entire DataFrame
- Multi-class support (works with 2+ classes)
- GitHub Actions CI (Python 3.9–3.12) and trusted-publisher PyPI deployment
- MkDocs Material documentation with API reference
- Benchmark examples reproducing Table 2 from the SGAI 2023 paper
- Full pytest test suite

### Changed
- Package name: `PertCF-Explainer` (research repo) → `pertcf` (installable package)
- Dependency: myCBR REST API + Java 8 removed; replaced with pure Python
- Architecture: monolithic script → `src/` layout with modular submodules

### Paper reference
Bayrak & Bach (2023). "PertCF: A Perturbation-Based Counterfactual Generation Approach."
SGAI AI-2023, LNAI 14381, pp. 174–187. https://doi.org/10.1007/978-3-031-47994-6_13
