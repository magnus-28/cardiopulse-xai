"""
Clinical Evaluation Metrics, Calibration, Fairness Audit & Latency Benchmarking.
"""
from __future__ import annotations
import time
import numpy as np


def roc_curve_np(y_true: np.ndarray, y_score: np.ndarray):
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    s_sorted = y_score[order]
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    tpr = np.concatenate([[0.0], tps / max(P, 1)])
    fpr = np.concatenate([[0.0], fps / max(N, 1)])
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def pr_curve_np(y_true: np.ndarray, y_score: np.ndarray):
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    P = np.sum(y_true == 1)
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    precision = np.concatenate([[1.0], tps / np.maximum(tps + fps, 1)])
    recall = np.concatenate([[0.0], tps / max(P, 1)])
    ap = float(np.trapezoid(precision, recall))
    return recall, precision, ap


def calibration_curve_np(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 8):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mean_pred, frac_pos, counts = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob <= hi if i == n_bins - 1 else y_prob < hi)
        if np.sum(mask) > 0:
            mean_pred.append(float(np.mean(y_prob[mask])))
            frac_pos.append(float(np.mean(y_true[mask])))
            counts.append(int(np.sum(mask)))
    ece = float(sum((c / len(y_true)) * abs(mp - fp) for mp, fp, c in zip(mean_pred, frac_pos, counts)))
    return np.array(mean_pred), np.array(frac_pos), ece


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    acc = (tp + tn) / max(len(y_true), 1)
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-12)
    brier = float(np.mean((y_prob - y_true) ** 2))
    _, _, roc_auc = roc_curve_np(y_true, y_prob)
    _, _, pr_auc = pr_curve_np(y_true, y_prob)
    _, _, ece = calibration_curve_np(y_true, y_prob)
    return {
        "accuracy": round(acc, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "precision": round(prec, 4),
        "f1": round(f1, 4),
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def find_clinical_threshold(y_true: np.ndarray, y_prob: np.ndarray, target_sens: float = 0.85) -> float:
    """Select screening threshold achieving >= target_sens with maximum specificity."""
    best_thr, best_spec = 0.5, -1.0
    for thr in np.linspace(0.20, 0.70, 101):
        m = compute_classification_metrics(y_true, y_prob, threshold=float(thr))
        if m["sensitivity"] >= target_sens and m["specificity"] > best_spec:
            best_spec = m["specificity"]
            best_thr = float(thr)
    return round(best_thr, 3)


def subgroup_fairness_audit(df_test, y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    """Evaluate ROC-AUC, Sensitivity, and Specificity across Age and Sex subgroups."""
    subgroups = {
        "Female (sex=0)": df_test["sex"].values == 0,
        "Male (sex=1)": df_test["sex"].values == 1,
        "Age < 55": df_test["age"].values < 55,
        "Age >= 55": df_test["age"].values >= 55,
    }
    report = {}
    for name, mask in subgroups.items():
        if np.sum(mask) < 10:
            continue
        m = compute_classification_metrics(y_true[mask], y_prob[mask], threshold=threshold)
        report[name] = {
            "n": int(np.sum(mask)),
            "roc_auc": m["roc_auc"],
            "sensitivity": m["sensitivity"],
            "specificity": m["specificity"],
            "brier_score": m["brier_score"],
        }
    return report


def benchmark_latency_ms(predict_fn, X_sample: np.ndarray, n_runs: int = 100) -> dict:
    """Measure single-patient and batch CPU inference latency in milliseconds."""
    single = X_sample[:1]
    t0 = time.perf_counter()
    for _ in range(n_runs):
        predict_fn(single)
    single_ms = (time.perf_counter() - t0) * 1000.0 / n_runs

    t0 = time.perf_counter()
    for _ in range(20):
        predict_fn(X_sample[:100])
    batch100_ms = (time.perf_counter() - t0) * 1000.0 / 20
    return {
        "single_patient_ms": round(single_ms, 2),
        "batch_100_ms": round(batch100_ms, 2),
    }
