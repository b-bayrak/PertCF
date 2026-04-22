# How PertCF Works

## Overview

PertCF combines two complementary XAI techniques:

1. **SHAP feature attribution** to understand which features matter most for each class
2. **Perturbation-based counterfactual generation** to iteratively nudge an instance toward a different prediction

The result: counterfactuals that are close to the original instance, stable under small perturbations, and guided by meaningful feature importance weights.

---

## Step 1: Compute SHAP weights per class

For each class *k*, PertCF computes the mean absolute SHAP value per feature over the training set:

```
w_k[f] = mean(|SHAP_k(x)[f]|)  for each instance x in training data
```

These weights are then normalised to sum to 1. This gives a **class-specific importance profile**, for example, in a knowledge modelling task, feature `PEG` might dominate class 0 while `LPR` dominates class 1.

---

## Step 2: Build per-class casebases

Training instances are grouped by class. When the explainer needs to search for the nearest unlike neighbour (NUN), it searches only within the relevant class's casebase.

---

## Step 3: Find the Nearest Unlike Neighbour (NUN)

Given query instance *x* with class *A*, the NUN is the training instance from any class *B ≠ A* that is most similar to *x*, measured using the SHAP-weighted similarity function:

```
sim(x, y | class_k) = Σ_f  w_k[f] · sim_f(x[f], y[f])
```

Where per-feature similarity is:

- **Numeric features:** `sim_f(a, b) = 1 - |a - b| / range_f`
- **Categorical features:** `sim_f(a, b) = 1` if equal, `0` otherwise (or custom value from a user-provided similarity matrix)

---

## Step 4: Iterative perturbation loop

```
source = x  (original instance)
target = NUN

repeat up to num_iter times:
    candidate = perturb(source → target)
    
    if candidate.class == target_class:
        if step_size < threshold:
            RETURN candidate          ← converged 
        else:
            source = candidate        ← refine toward source
    
    elif candidate.class == neither:
        source = candidate            ← follow the drift
    
    else:
        target = NUN                  ← push harder toward target
```

The perturbation formula for each feature *f* is:

**Numeric:**
```
candidate[f] = source[f] + shap_target[f] × (target[f] - source[f])
```

**Categorical:**
```
candidate[f] = target[f]   if sim(source[f], target[f]) < 0.5
candidate[f] = source[f]   otherwise
```

The step-size threshold is:
```
threshold = dist(x, NUN) / coef
```

A higher `coef` means a tighter convergence criterion (finer granularity, more iterations).

---

## Step 5: Termination

The loop stops when one of these is true:

1. The step size (distance between last two candidates) drops below `threshold` → converged at a good CF
2. The iteration counter reaches `num_iter` → return the last valid candidate from the candidate list
3. No valid candidate was found → restart with the next best NUN from a different class

---

## Why SHAP weights matter

Without SHAP weights, a uniform distance function treats all features equally. SHAP weights ensure that **features which actually matter to the model** drive the perturbation. This is why PertCF achieves lower dissimilarity and instability than DiCE and CF-SHAP. It moves in the directions the model cares about, not arbitrary directions.

---

## Supported model types

| Model type | How it's handled                                 |
|---|--------------------------------------------------|
| scikit-learn estimator | Auto-detected via `hasattr(model, "classes_")`   |
| PyTorch `nn.Module` | `_TorchAdapter`, `softmax` over logits           |
| Keras/TF `Model` | `_KerasAdapter`, direct `.predict()`             |
| Any callable | `_CallableAdapter`, pass `predict_fn` explicitly |
