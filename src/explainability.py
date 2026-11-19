"""
Explainable AI (XAI) Engine for CardioPulse XAI:
  1. Shapley Value Feature Attribution (Monte-Carlo permutation SHAP, Lundberg & Lee 2017)
  2. Actionable Counterfactual Risk Reduction Optimizer ("What-If" clinical simulator)
"""
from __future__ import annotations
import numpy as np
from .data_pipeline import FEATURE_NAMES, FEATURE_LABELS


def compute_shapley_local(
    predict_fn,
    x_instance: np.ndarray,
    X_background: np.ndarray,
    n_permutations: int = 80,
    seed: int = 42,
) -> tuple[float, np.ndarray]:
    """
    Estimate local Shapley values for a single patient vector x_instance
    using unbiased permutation sampling (SHAP additive attribution).
    Returns: (base_value, shap_values_array)
    """
    rng = np.random.default_rng(seed)
    d = len(x_instance)
    bg_preds = predict_fn(X_background)
    base_value = float(np.mean(bg_preds))
    shap_vals = np.zeros(d, dtype=float)

    n_bg = len(X_background)
    for _ in range(n_permutations):
        perm = rng.permutation(d)
        bg_row = X_background[rng.integers(0, n_bg)].copy()
        # Build sequence of (d+1) coalitions in one batch for fast vectorized inference
        batch = np.repeat(bg_row.reshape(1, -1), d + 1, axis=0)
        for step, feat_idx in enumerate(perm):
            batch[step + 1 :, feat_idx] = x_instance[feat_idx]
        preds = predict_fn(batch)
        deltas = np.diff(preds)
        for step, feat_idx in enumerate(perm):
            shap_vals[feat_idx] += deltas[step]

    shap_vals /= n_permutations
    return base_value, shap_vals


def compute_shapley_global(
    predict_fn,
    X_eval: np.ndarray,
    X_background: np.ndarray,
    max_samples: int = 80,
    n_permutations: int = 35,
    seed: int = 42,
) -> np.ndarray:
    """Compute mean(|SHAP|) global feature importance across a cohort subset."""
    n = min(max_samples, len(X_eval))
    all_shap = np.zeros((n, X_eval.shape[1]), dtype=float)
    for i in range(n):
        _, sv = compute_shapley_local(
            predict_fn, X_eval[i], X_background,
            n_permutations=n_permutations, seed=seed + i
        )
        all_shap[i] = sv
    return all_shap


def generate_counterfactual_plan(
    predict_raw_fn,
    patient_raw: dict[str, float],
    target_risk: float = 0.35,
) -> list[dict]:
    """
    Simulate realistic, modifiable clinical interventions (blood pressure control,
    cholesterol reduction, exercise capacity improvement, glycemic control)
    and rank by predicted absolute risk reduction.
    """
    base_vec = np.array([[patient_raw[f] for f in FEATURE_NAMES]], dtype=float)
    base_risk = float(predict_raw_fn(base_vec)[0])

    interventions = [
        ("trestbps", "Reduce Resting BP by 15 mm Hg (lifestyle / antihypertensive therapy)",
         max(110.0, patient_raw["trestbps"] - 15.0)),
        ("chol", "Lower Serum Cholesterol by 40 mg/dL (diet / statin therapy)",
         max(160.0, patient_raw["chol"] - 40.0)),
        ("thalach", "Improve Max Heart Rate Reserve +15 bpm (cardiac rehab)",
         min(190.0, patient_raw["thalach"] + 15.0)),
        ("fbs", "Control Fasting Blood Sugar <= 120 mg/dL (glycemic management)", 0.0),
        ("oldpeak", "Reduce Exercise ST Depression by 0.8 mm (ischemia therapy)",
         max(0.0, round(patient_raw["oldpeak"] - 0.8, 1))),
    ]

    results = []
    for feat, desc, new_val in interventions:
        if abs(new_val - patient_raw[feat]) < 1e-6:
            continue
        mod = dict(patient_raw)
        mod[feat] = new_val
        vec = np.array([[mod[f] for f in FEATURE_NAMES]], dtype=float)
        new_risk = float(predict_raw_fn(vec)[0])
        delta = base_risk - new_risk
        if delta > 0.003:
            results.append({
                "feature": feat,
                "label": FEATURE_LABELS[feat],
                "intervention": desc,
                "from_val": patient_raw[feat],
                "to_val": new_val,
                "new_risk": new_risk,
                "risk_reduction": delta,
            })

    results.sort(key=lambda r: r["risk_reduction"], reverse=True)
    return results
