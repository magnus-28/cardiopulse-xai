#!/usr/bin/env python3
"""
CardioPulse XAI — End-to-End Training, 5-Fold Cross-Validation, Calibration,
Explainability (SHAP), Fairness Audit & Visualization Pipeline.

Usage:
    python train_and_evaluate.py
    python train_and_evaluate.py --csv path/to/uci_heart.csv
"""
from __future__ import annotations
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_pipeline import (
    FEATURE_NAMES, FEATURE_LABELS, StandardScalerNP,
    load_dataset, stratified_train_test_split, stratified_kfold_indices,
)
from src.models import (
    LogisticRegressionL2, RandomForestScratch,
    GradientBoostingScratch, ClinicalMLP, CalibratedEnsemble,
)
from src.explainability import (
    compute_shapley_local, compute_shapley_global, generate_counterfactual_plan,
)
from src.evaluation import (
    roc_curve_np, pr_curve_np, calibration_curve_np,
    compute_classification_metrics, find_clinical_threshold,
    subgroup_fairness_audit, benchmark_latency_ms,
)

PALETTE = {
    "primary": "#0F4C5C",
    "teal": "#1B9AAA",
    "coral": "#E36414",
    "amber": "#F4A261",
    "slate": "#475569",
    "light": "#F1F5F9",
}


def build_base_models(seed: int = 42) -> dict:
    return {
        "Logistic Regression": LogisticRegressionL2(C=1.0),
        "Random Forest": RandomForestScratch(n_estimators=36, max_depth=5, min_samples_leaf=6, seed=seed),
        "Gradient Boosting": GradientBoostingScratch(n_estimators=50, learning_rate=0.08, max_depth=3, min_samples_leaf=10, seed=seed),
        "Clinical MLP": ClinicalMLP(hidden1=32, hidden2=16, lr=0.012, epochs=220, l2=0.008, seed=seed),
    }


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate CardioPulse XAI")
    parser.add_argument("--csv", type=str, default=None, help="Optional path to UCI Heart Disease CSV")
    parser.add_argument("--out", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    df, source_desc = load_dataset(args.csv, n_samples=1200, seed=42)
    X = df[FEATURE_NAMES].values.astype(float)
    y = df["target"].values.astype(int)

    # 80/20 stratified holdout split
    X_train_all, X_test_raw, y_train_all, y_test = stratified_train_test_split(X, y, test_size=0.20, seed=42)
    # Reconstruct df_test for subgroup audit
    import pandas as pd
    df_test = pd.DataFrame(X_test_raw, columns=FEATURE_NAMES)

    # 1. Stratified 5-Fold Cross-Validation on Training Set
    cv_aucs = {name: [] for name in list(build_base_models().keys()) + ["Calibrated Ensemble"]}
    cv_briers = {name: [] for name in cv_aucs}
    weights = {"Logistic Regression": 0.25, "Random Forest": 0.20, "Gradient Boosting": 0.35, "Clinical MLP": 0.20}

    for fold_idx, (tr_idx, va_idx) in enumerate(stratified_kfold_indices(y_train_all, n_splits=5, seed=42)):
        scaler_f = StandardScalerNP()
        X_tr = scaler_f.fit_transform(X_train_all[tr_idx])
        X_va = scaler_f.transform(X_train_all[va_idx])
        y_tr, y_va = y_train_all[tr_idx], y_train_all[va_idx]

        models_f = build_base_models(seed=100 + fold_idx)
        for name, m in models_f.items():
            m.fit(X_tr, y_tr)
            p_va = m.predict_proba(X_va)
            met = compute_classification_metrics(y_va, p_va)
            cv_aucs[name].append(met["roc_auc"])
            cv_briers[name].append(met["brier_score"])

        ens_f = CalibratedEnsemble(models_f, weights)
        p_ens = ens_f.predict_proba(X_va)
        met_e = compute_classification_metrics(y_va, p_ens)
        cv_aucs["Calibrated Ensemble"].append(met_e["roc_auc"])
        cv_briers["Calibrated Ensemble"].append(met_e["brier_score"])

    # 2. Final Training + Platt Calibration Split (85% train / 15% calibration inside training pool)
    X_fit_raw, X_cal_raw, y_fit, y_cal = stratified_train_test_split(X_train_all, y_train_all, test_size=0.15, seed=99)
    scaler = StandardScalerNP()
    X_fit = scaler.fit_transform(X_fit_raw)
    X_cal = scaler.transform(X_cal_raw)
    X_test = scaler.transform(X_test_raw)

    final_models = build_base_models(seed=42)
    test_probs = {}
    test_metrics = {}
    for name, m in final_models.items():
        m.fit(X_fit, y_fit)
        p = m.predict_proba(X_test)
        test_probs[name] = p
        test_metrics[name] = compute_classification_metrics(y_test, p, threshold=0.50)

    ensemble = CalibratedEnsemble(final_models, weights).calibrate(X_cal, y_cal)
    p_ens_test = ensemble.predict_proba(X_test)
    test_probs["Calibrated Ensemble"] = p_ens_test

    # Select clinical high-sensitivity operating threshold on calibration split
    p_ens_cal = ensemble.predict_proba(X_cal)
    clin_thr = find_clinical_threshold(y_cal, p_ens_cal, target_sens=0.85)
    test_metrics["Calibrated Ensemble"] = compute_classification_metrics(y_test, p_ens_test, threshold=0.50)
    ens_clinical_metrics = compute_classification_metrics(y_test, p_ens_test, threshold=clin_thr)

    train_seconds = round(time.perf_counter() - t_start, 2)

    # 3. Explainability (Global + Local Shapley & Counterfactual Plan)
    bg_X = X_fit[:120]
    shap_matrix = compute_shapley_global(ensemble.predict_proba, X_test, bg_X, max_samples=60, n_permutations=30, seed=42)
    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
    global_importance = sorted(
        [{"feature": f, "label": FEATURE_LABELS[f], "mean_abs_shap": round(float(v), 4)}
         for f, v in zip(FEATURE_NAMES, mean_abs_shap)],
        key=lambda r: r["mean_abs_shap"], reverse=True,
    )

    # Pick a high-risk positive test patient for local clinical case study
    pos_candidates = np.where((y_test == 1) & (p_ens_test >= 0.72) & (p_ens_test <= 0.90))[0]
    case_idx = int(pos_candidates[0]) if len(pos_candidates) else int(np.argmax(p_ens_test))
    base_val, local_shap = compute_shapley_local(ensemble.predict_proba, X_test[case_idx], bg_X, n_permutations=80, seed=7)
    case_raw = {f: float(X_test_raw[case_idx, i]) for i, f in enumerate(FEATURE_NAMES)}
    case_risk = float(p_ens_test[case_idx])
    counterfactuals = generate_counterfactual_plan(lambda raw_arr: ensemble.predict_proba(scaler.transform(raw_arr)), case_raw)

    # 4. Fairness Audit & CPU Latency Benchmark
    fairness = subgroup_fairness_audit(df_test, y_test, p_ens_test, threshold=clin_thr)
    latency = benchmark_latency_ms(ensemble.predict_proba, X_test, n_runs=80)

    # 5. Save Model Bundle & JSON Metrics
    bundle = {
        "scaler": scaler,
        "ensemble": ensemble,
        "background_X": bg_X,
        "clinical_threshold": clin_thr,
        "feature_names": FEATURE_NAMES,
    }
    with open(out_dir / "model_bundle.pkl", "wb") as f:
        pickle.dump(bundle, f)

    summary = {
        "dataset_source": source_desc,
        "n_total": int(len(df)),
        "n_train": int(len(X_train_all)),
        "n_test": int(len(X_test)),
        "prevalence": round(float(np.mean(y)), 4),
        "total_training_seconds": train_seconds,
        "clinical_threshold": clin_thr,
        "cv_5fold": {
            k: {
                "roc_auc_mean": round(float(np.mean(cv_aucs[k])), 4),
                "roc_auc_std": round(float(np.std(cv_aucs[k])), 4),
                "brier_mean": round(float(np.mean(cv_briers[k])), 4),
            } for k in cv_aucs
        },
        "holdout_test_metrics_t50": test_metrics,
        "holdout_ensemble_clinical_thr": ens_clinical_metrics,
        "global_shap_top8": global_importance[:8],
        "case_study": {
            "patient_raw": case_raw,
            "predicted_risk": round(case_risk, 4),
            "base_risk": round(base_val, 4),
            "top_shap_drivers": sorted(
                [{"feature": f, "label": FEATURE_LABELS[f], "value": case_raw[f], "shap": round(float(s), 4)}
                 for f, s in zip(FEATURE_NAMES, local_shap)],
                key=lambda r: abs(r["shap"]), reverse=True,
            )[:6],
            "counterfactuals": counterfactuals[:4],
        },
        "fairness_audit": fairness,
        "cpu_latency": latency,
    }
    (out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))

    # 6. Generate 4 Publication-Grade Matplotlib Figures (220 DPI, clean print width)
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    # Fig 1: ROC & Precision-Recall Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.3), constrained_layout=True)
    colors = {
        "Calibrated Ensemble": PALETTE["coral"],
        "Gradient Boosting": PALETTE["primary"],
        "Logistic Regression": PALETTE["teal"],
        "Clinical MLP": "#6366F1",
        "Random Forest": PALETTE["slate"],
    }
    for name, probs in test_probs.items():
        lw = 2.4 if name == "Calibrated Ensemble" else 1.4
        ls = "-" if name == "Calibrated Ensemble" else "--"
        fpr, tpr, auc = roc_curve_np(y_test, probs)
        ax1.plot(fpr, tpr, label=f"{name} ({auc:.3f})", color=colors[name], lw=lw, ls=ls)
        rec, prec, ap = pr_curve_np(y_test, probs)
        ax2.plot(rec, prec, label=f"{name} ({ap:.3f})", color=colors[name], lw=lw, ls=ls)

    ax1.plot([0, 1], [0, 1], ":", color="#94A3B8", lw=1.0)
    ax1.set_title("ROC Curve (Holdout Test Set)", fontweight="bold")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate (Sensitivity)")
    ax1.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8)
    ax1.grid(alpha=0.25)

    ax2.set_title("Precision-Recall Curve", fontweight="bold")
    ax2.set_xlabel("Recall (Sensitivity)")
    ax2.set_ylabel("Precision (PPV)")
    ax2.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8)
    ax2.grid(alpha=0.25)
    fig.savefig(out_dir / "fig1_roc_pr_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Fig 2: Calibration Reliability Diagram + Clinical Confusion Matrix
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.2), constrained_layout=True)
    mp, fp_cal, ece = calibration_curve_np(y_test, p_ens_test, n_bins=8)
    ax1.plot([0, 1], [0, 1], "--", color="#94A3B8", lw=1.2, label="Perfect Calibration")
    ax1.plot(mp, fp_cal, "o-", color=PALETTE["primary"], lw=2.2, markersize=6, label=f"Calibrated Ensemble (ECE={ece:.3f})")
    ax1.set_title("Probability Calibration Curve", fontweight="bold")
    ax1.set_xlabel("Mean Predicted Risk")
    ax1.set_ylabel("Observed Disease Rate")
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8.5)
    ax1.grid(alpha=0.25)

    cm = ens_clinical_metrics["confusion"]
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    im = ax2.imshow(mat, cmap="Blues")
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["Pred: Low", "Pred: High"])
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(["Actual: Low", "Actual: High"])
    ax2.set_title(f"Confusion Matrix (Thr = {clin_thr:.2f})", fontweight="bold")
    for r in range(2):
        for c in range(2):
            txt_color = "white" if mat[r, c] > mat.max() * 0.55 else "#0F172A"
            ax2.text(c, r, f"{mat[r, c]}", ha="center", va="center", color=txt_color, fontsize=13, fontweight="bold")
    fig.savefig(out_dir / "fig2_calibration_confusion.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Fig 3: Global SHAP Importance + Local Patient Explanation Waterfall
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.4), constrained_layout=True)
    top8 = global_importance[:8][::-1]
    ax1.barh([r["feature"] for r in top8], [r["mean_abs_shap"] for r in top8], color=PALETTE["primary"], height=0.65)
    ax1.set_title("Global Feature Importance (|SHAP|)", fontweight="bold")
    ax1.set_xlabel("Mean |SHAP Value| (Risk Impact)")
    ax1.grid(axis="x", alpha=0.25)

    loc6 = summary["case_study"]["top_shap_drivers"][::-1]
    loc_colors = [PALETTE["coral"] if r["shap"] > 0 else PALETTE["teal"] for r in loc6]
    ax2.barh([f"{r['feature']}={r['value']:.0f}" for r in loc6], [r["shap"] for r in loc6], color=loc_colors, height=0.62)
    ax2.axvline(0, color="#334155", lw=1.0)
    ax2.set_title(f"Local Patient Drivers (Risk = {case_risk*100:.1f}%)", fontweight="bold")
    ax2.set_xlabel("SHAP Contribution (+ Increases Risk)")
    ax2.grid(axis="x", alpha=0.25)
    fig.savefig(out_dir / "fig3_shap_global_local.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Fig 4: Subgroup Fairness Audit (ROC-AUC & Sensitivity)
    fig, ax = plt.subplots(figsize=(6.8, 2.9), constrained_layout=True)
    groups = list(fairness.keys())
    x = np.arange(len(groups))
    w = 0.28
    aucs = [fairness[g]["roc_auc"] for g in groups]
    sens = [fairness[g]["sensitivity"] for g in groups]
    spec = [fairness[g]["specificity"] for g in groups]
    ax.bar(x - w, aucs, width=w, label="ROC-AUC", color=PALETTE["primary"])
    ax.bar(x, sens, width=w, label="Sensitivity", color=PALETTE["coral"])
    ax.bar(x + w, spec, width=w, label="Specificity", color=PALETTE["teal"])
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Subgroup Fairness Audit Across Sex and Age", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=True, facecolor="white", edgecolor="#CBD5E1")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_dir / "fig4_fairness_audit.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
