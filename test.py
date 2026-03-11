"""
╔══════════════════════════════════════════════════════════════════════╗
║           SEPSIS PREDICTION — TEST / EVALUATION SCRIPT              ║
║           test.py  |  Optimised for 16 GB RAM Laptop                ║
╠══════════════════════════════════════════════════════════════════════╣
║  REQUIREMENT: Run train.py first to generate the models/            ║
║               folder and its saved artefacts.                       ║
║                                                                     ║
║  STEP 3 — Evaluate:                                                 ║
║    python test.py                                                   ║
║                                                                     ║
║  READS FROM:                                                        ║
║    models/xgb_model.json        ← trained XGBoost                  ║
║    models/lgb_model.txt         ← trained LightGBM                 ║
║    models/imputer.pkl           ← fitted imputer                   ║
║    models/feature_cols.pkl      ← feature names                    ║
║    models/ensemble_weight.pkl   ← best ensemble weight             ║
║    models/thresholds.pkl        ← optimal classification thresholds║
║    models/X_test.npy            ← held-out test features           ║
║    models/y_test.npy            ← held-out test labels             ║
║                                                                     ║
║  OUTPUT FILES SAVED:                                                ║
║    sepsis_test_outputs/         ← all evaluation plots + CSVs      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ── Standard library ──────────────────────────────────────────────────────
import os, gc, time, warnings, psutil
import joblib

# ── Third-party ───────────────────────────────────────────────────────────
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec

# ── ML ────────────────────────────────────────────────────────────────────
import xgboost  as xgb
import lightgbm as lgb

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    roc_curve, precision_recall_curve,
    f1_score, brier_score_loss, log_loss,
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════
MODEL_DIR = "models"
OUT_DIR   = "sepsis_test_outputs"
PLOT_DPI  = 120

PALETTE = {
    "Sepsis"   : "#E84040",
    "No Sepsis": "#4C72B0",
    "XGBoost"  : "#2196F3",
    "LightGBM" : "#FF9800",
    "Ensemble" : "#4CAF50",
}

os.makedirs(OUT_DIR, exist_ok=True)

def ram():
    used  = psutil.Process().memory_info().rss / 1e9
    total = psutil.virtual_memory().total / 1e9
    return f"RAM {used:.1f}/{total:.1f} GB"


# ═════════════════════════════════════════════════════════════════════════
# 1.  LOAD MODELS & ARTEFACTS
# ═════════════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  [1/4]  Loading Saved Models & Artefacts …")
print("═"*65)
t0 = time.time()

# Verify models folder exists
required = ["xgb_model.json", "lgb_model.txt", "imputer.pkl",
            "feature_cols.pkl", "ensemble_weight.pkl",
            "thresholds.pkl", "X_test.npy", "y_test.npy"]
missing = [f for f in required if not os.path.exists(f"{MODEL_DIR}/{f}")]
if missing:
    print(f"\n  ❌  ERROR: Missing files in models/ folder:")
    for f in missing:
        print(f"       — {f}")
    print("\n  ➜  Please run train.py first!\n")
    raise SystemExit(1)

# Load XGBoost
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(f"{MODEL_DIR}/xgb_model.json")
print("  ✔  XGBoost model loaded")

# Load LightGBM
lgb_booster = lgb.Booster(model_file=f"{MODEL_DIR}/lgb_model.txt")
print("  ✔  LightGBM model loaded")

# Load preprocessing artefacts
imputer       = joblib.load(f"{MODEL_DIR}/imputer.pkl")
feature_cols  = joblib.load(f"{MODEL_DIR}/feature_cols.pkl")
ensemble_w    = joblib.load(f"{MODEL_DIR}/ensemble_weight.pkl")
thresholds    = joblib.load(f"{MODEL_DIR}/thresholds.pkl")

print(f"  ✔  Imputer loaded")
print(f"  ✔  Feature columns : {len(feature_cols)} features")
print(f"  ✔  Ensemble weight : XGBoost={ensemble_w:.2f}  LightGBM={1-ensemble_w:.2f}")
print(f"  ✔  Thresholds      : XGBoost={thresholds['XGBoost']:.4f}  "
      f"LightGBM={thresholds['LightGBM']:.4f}  "
      f"Ensemble={thresholds['Ensemble']:.4f}")

# Load held-out test set
X_test = np.load(f"{MODEL_DIR}/X_test.npy").astype(np.float32)
y_test = np.load(f"{MODEL_DIR}/y_test.npy")
print(f"\n  ✔  Test set loaded : {X_test.shape[0]:,} rows × {X_test.shape[1]} features")
print(f"  ✔  {ram()}")


# ═════════════════════════════════════════════════════════════════════════
# 2.  GENERATE PREDICTIONS
# ═════════════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  [2/4]  Generating Predictions …")
print("═"*65)

# Apply imputer (handles any remaining NaNs in test set)
X_test_imp = imputer.transform(X_test).astype(np.float32)
gc.collect()

# XGBoost predictions
xgb_prob = xgb_model.predict_proba(X_test_imp)[:, 1]
print(f"  ✔  XGBoost  predictions done  |  {ram()}")

# LightGBM predictions (booster predict returns raw proba directly)
lgb_prob = lgb_booster.predict(X_test_imp)
print(f"  ✔  LightGBM predictions done  |  {ram()}")

# Ensemble predictions
ens_prob = ensemble_w * xgb_prob + (1 - ensemble_w) * lgb_prob
print(f"  ✔  Ensemble predictions done")

del X_test_imp; gc.collect()


# ═════════════════════════════════════════════════════════════════════════
# 3.  EVALUATION METRICS
# ═════════════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  [3/4]  Computing Evaluation Metrics …")
print("═"*65)

model_names  = ["XGBoost", "LightGBM", "Ensemble"]
model_colors = [PALETTE["XGBoost"], PALETTE["LightGBM"], PALETTE["Ensemble"]]
probas       = {"XGBoost": xgb_prob, "LightGBM": lgb_prob, "Ensemble": ens_prob}

results = {}
for nm in model_names:
    proba = probas[nm]
    thr   = thresholds[nm]
    pred  = (proba >= thr).astype(int)
    cm    = confusion_matrix(y_test, pred)
    tn, fp, fn, tp = cm.ravel()

    results[nm] = dict(
        AUC         = roc_auc_score(y_test, proba),
        AP          = average_precision_score(y_test, proba),
        F1          = f1_score(y_test, pred),
        Sensitivity = tp / (tp + fn + 1e-9),
        Specificity = tn / (tn + fp + 1e-9),
        PPV         = tp / (tp + fp + 1e-9),
        NPV         = tn / (tn + fn + 1e-9),
        Brier       = brier_score_loss(y_test, proba),
        LogLoss     = log_loss(y_test, proba),
        Threshold   = thr,
        CM          = cm,
        proba       = proba,
        pred        = pred,
        TN=tn, FP=fp, FN=fn, TP=tp,
    )

    r = results[nm]
    print(f"\n  ┌─ {nm} {'─'*(42-len(nm))}")
    print(f"  │  AUC-ROC        : {r['AUC']:.4f}")
    print(f"  │  Avg Precision  : {r['AP']:.4f}")
    print(f"  │  F1 Score       : {r['F1']:.4f}")
    print(f"  │  Sensitivity    : {r['Sensitivity']:.4f}  (True Positive Rate)")
    print(f"  │  Specificity    : {r['Specificity']:.4f}  (True Negative Rate)")
    print(f"  │  PPV (Precision): {r['PPV']:.4f}")
    print(f"  │  NPV            : {r['NPV']:.4f}")
    print(f"  │  Brier Score    : {r['Brier']:.4f}  (↓ better)")
    print(f"  │  Log Loss       : {r['LogLoss']:.4f}  (↓ better)")
    print(f"  │  Threshold      : {r['Threshold']:.4f}")
    print(f"  │  TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
    print(f"  └{'─'*46}")

print("\n  Ensemble — Full Classification Report:")
print(classification_report(y_test, results["Ensemble"]["pred"],
      target_names=["No Sepsis", "Sepsis"]))


# ═════════════════════════════════════════════════════════════════════════
# 4.  VISUALISATIONS
# ═════════════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  [4/4]  Generating Evaluation Plots …")
print("═"*65)

# ── Figure 1: ROC + PR Curves ─────────────────────────────────────────
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6))
for nm, color in zip(model_names, model_colors):
    fpr, tpr, _  = roc_curve(y_test, results[nm]["proba"])
    prec, rec, _ = precision_recall_curve(y_test, results[nm]["proba"])
    a1.plot(fpr, tpr, color=color, lw=2.5,
            label=f"{nm}  AUC={results[nm]['AUC']:.4f}")
    a2.plot(rec, prec, color=color, lw=2.5,
            label=f"{nm}  AP={results[nm]['AP']:.4f}")

a1.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random")
a1.fill_between(*roc_curve(y_test, ens_prob)[:2],
                alpha=0.07, color=PALETTE["Ensemble"])
a1.set_xlabel("False Positive Rate", fontsize=12)
a1.set_ylabel("True Positive Rate",  fontsize=12)
a1.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
a1.legend(fontsize=10); a1.grid(alpha=0.3)

a2.axhline(y_test.mean(), color="k", linestyle="--", lw=1.2,
           label=f"Baseline ({y_test.mean():.3f})")
a2.set_xlabel("Recall",    fontsize=12)
a2.set_ylabel("Precision", fontsize=12)
a2.set_title("Precision-Recall Curves — All Models", fontsize=13, fontweight="bold")
a2.legend(fontsize=10); a2.grid(alpha=0.3)

plt.suptitle(f"Sepsis Prediction — Model Comparison  "
             f"(Test Set  n={len(y_test):,})",
             fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/01_roc_pr_curves.png", dpi=PLOT_DPI, bbox_inches="tight")
plt.close(); gc.collect()
print("  ✔  Saved: 01_roc_pr_curves.png")

# ── Figure 2: Confusion Matrices ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, nm, color in zip(axes, model_names, model_colors):
    cm   = results[nm]["CM"]
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    ax.imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i,
                    f"{cm_n[i,j]:.2%}\n({cm[i,j]:,})",
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color="white" if cm_n[i, j] > 0.5 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Sepsis", "Sepsis"], fontsize=11)
    ax.set_yticklabels(["No Sepsis", "Sepsis"], fontsize=11)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
    ax.set_title(f"{nm}\nAUC={results[nm]['AUC']:.4f}  F1={results[nm]['F1']:.4f}",
                 fontsize=12, fontweight="bold", color=color)
plt.suptitle("Confusion Matrices (Normalised) — Test Set",
             fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/02_confusion_matrices.png", dpi=PLOT_DPI, bbox_inches="tight")
plt.close(); gc.collect()
print("  ✔  Saved: 02_confusion_matrices.png")

# ── Figure 3: Metrics Comparison ──────────────────────────────────────
metrics = ["AUC", "AP", "F1", "Sensitivity", "Specificity", "PPV", "NPV"]
x, w    = np.arange(len(metrics)), 0.26
fig, ax = plt.subplots(figsize=(15, 6))
for i, (nm, color) in enumerate(zip(model_names, model_colors)):
    vals = [results[nm][m] for m in metrics]
    bars = ax.bar(x + i*w - w, vals, w, label=nm,
                  color=color, alpha=0.88, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.007,
                f"{val:.3f}", ha="center", fontsize=8.5, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0, 1.18)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Performance Comparison — Test Set",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=11); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/03_metrics_comparison.png", dpi=PLOT_DPI, bbox_inches="tight")
plt.close(); gc.collect()
print("  ✔  Saved: 03_metrics_comparison.png")

# ── Figure 4: Score Distributions ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, nm, color in zip(axes, model_names, model_colors):
    thr = results[nm]["Threshold"]
    ax.hist(results[nm]["proba"][y_test == 0], bins=80, alpha=0.65,
            color=PALETTE["No Sepsis"], density=True, label="No Sepsis")
    ax.hist(results[nm]["proba"][y_test == 1], bins=80, alpha=0.65,
            color=PALETTE["Sepsis"],    density=True, label="Sepsis")
    ax.axvline(thr, color="black", linestyle="--", lw=2,
               label=f"Threshold={thr:.3f}")
    ax.set_title(f"{nm} — Score Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
plt.suptitle("Predicted Probability Distributions — Test Set",
             fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/04_score_distributions.png", dpi=PLOT_DPI, bbox_inches="tight")
plt.close(); gc.collect()
print("  ✔  Saved: 04_score_distributions.png")

# ── Figure 5: Calibration Curves ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect Calibration")
for nm, color in zip(model_names, model_colors):
    fp, mp = calibration_curve(y_test, results[nm]["proba"],
                                n_bins=15, strategy="uniform")
    ax.plot(mp, fp, "s-", color=color, label=nm, lw=2, markersize=7)
ax.set_xlabel("Mean Predicted Probability", fontsize=12)
ax.set_ylabel("Fraction of Positives",      fontsize=12)
ax.set_title("Calibration Curves — Test Set", fontsize=13, fontweight="bold")
ax.legend(fontsize=11); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/05_calibration_curves.png", dpi=PLOT_DPI, bbox_inches="tight")
plt.close(); gc.collect()
print("  ✔  Saved: 05_calibration_curves.png")

# ── Figure 6: Threshold Analysis (Ensemble) ───────────────────────────
thresholds_range = np.linspace(0.01, 0.99, 200)
f1s, sens_l, spec_l, ppv_l = [], [], [], []
for t in thresholds_range:
    pred_t = (ens_prob >= t).astype(int)
    cm_t   = confusion_matrix(y_test, pred_t)
    if cm_t.shape == (2, 2):
        tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
        f1s.append(f1_score(y_test, pred_t, zero_division=0))
        sens_l.append(tp_t / (tp_t + fn_t + 1e-9))
        spec_l.append(tn_t / (tn_t + fp_t + 1e-9))
        ppv_l.append(tp_t  / (tp_t + fp_t + 1e-9))
    else:
        f1s.append(0); sens_l.append(0); spec_l.append(0); ppv_l.append(0)

thr_ens = thresholds["Ensemble"]
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(thresholds_range, f1s,    label="F1 Score",        color=PALETTE["Ensemble"], lw=2.5)
ax.plot(thresholds_range, sens_l, label="Sensitivity",     color=PALETTE["Sepsis"],   lw=2)
ax.plot(thresholds_range, spec_l, label="Specificity",     color=PALETTE["No Sepsis"],lw=2)
ax.plot(thresholds_range, ppv_l,  label="PPV (Precision)", color="purple",            lw=2)
ax.axvline(thr_ens, color="black", linestyle="--", lw=1.8,
           label=f"Optimal Threshold = {thr_ens:.3f}")
ax.set_xlabel("Classification Threshold", fontsize=12)
ax.set_ylabel("Metric Value",             fontsize=12)
ax.set_title("Ensemble — Threshold Analysis (Test Set)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/06_threshold_analysis.png", dpi=PLOT_DPI, bbox_inches="tight")
plt.close(); gc.collect()
print("  ✔  Saved: 06_threshold_analysis.png")

# ── Figure 7: Summary Dashboard ───────────────────────────────────────
fig = plt.figure(figsize=(20, 11))
fig.patch.set_facecolor("#F0F4F8")
gs2 = gridspec.GridSpec(3, 4, figure=fig, hspace=0.62, wspace=0.44)

ax_roc = fig.add_subplot(gs2[0, :2])
for nm, color in zip(model_names, model_colors):
    fpr, tpr, _ = roc_curve(y_test, results[nm]["proba"])
    ax_roc.plot(fpr, tpr, color=color, lw=2.5,
                label=f"{nm}  AUC={results[nm]['AUC']:.4f}")
ax_roc.plot([0, 1], [0, 1], "k--", lw=1.2)
ax_roc.set_title("ROC Curve", fontsize=12, fontweight="bold")
ax_roc.legend(fontsize=9); ax_roc.grid(alpha=0.3)
ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")

ax_pr = fig.add_subplot(gs2[0, 2:])
for nm, color in zip(model_names, model_colors):
    prec, rec, _ = precision_recall_curve(y_test, results[nm]["proba"])
    ax_pr.plot(rec, prec, color=color, lw=2.5,
               label=f"{nm}  AP={results[nm]['AP']:.4f}")
ax_pr.axhline(y_test.mean(), color="k", linestyle="--", lw=1)
ax_pr.set_title("Precision-Recall Curve", fontsize=12, fontweight="bold")
ax_pr.legend(fontsize=9); ax_pr.grid(alpha=0.3)
ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")

metric_pairs = [("AUC","AUC-ROC"), ("F1","F1 Score"),
                ("Sensitivity","Sensitivity"), ("Specificity","Specificity")]
for idx, (metric, label) in enumerate(metric_pairs):
    axm = fig.add_subplot(gs2[1, idx])
    vals = [results[n][metric] for n in model_names]
    brs  = axm.bar(model_names, vals, color=model_colors, alpha=0.88, edgecolor="white")
    axm.set_ylim(0, 1.18)
    axm.set_title(label, fontsize=12, fontweight="bold")
    for b, v in zip(brs, vals):
        axm.text(b.get_x()+b.get_width()/2, b.get_height()+0.02,
                 f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    axm.set_xticklabels(model_names, rotation=12, fontsize=9)
    axm.grid(axis="y", alpha=0.3)

# Calibration mini-plot
ax_cal = fig.add_subplot(gs2[2, :2])
ax_cal.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect")
for nm, color in zip(model_names, model_colors):
    fp, mp = calibration_curve(y_test, results[nm]["proba"],
                                n_bins=12, strategy="uniform")
    ax_cal.plot(mp, fp, "s-", color=color, label=nm, lw=2, markersize=5)
ax_cal.set_title("Calibration Curves", fontsize=11, fontweight="bold")
ax_cal.set_xlabel("Mean Predicted Prob"); ax_cal.set_ylabel("Fraction Positives")
ax_cal.legend(fontsize=9); ax_cal.grid(alpha=0.3)

# Class-level metrics table
ax_tbl = fig.add_subplot(gs2[2, 2:])
ax_tbl.axis("off")
tbl_data = [[nm,
             f"{results[nm]['Sensitivity']:.3f}",
             f"{results[nm]['Specificity']:.3f}",
             f"{results[nm]['PPV']:.3f}",
             f"{results[nm]['NPV']:.3f}",
             f"{results[nm]['Brier']:.4f}"]
            for nm in model_names]
tbl = ax_tbl.table(
    cellText   = tbl_data,
    colLabels  = ["Model", "Sens.", "Spec.", "PPV", "NPV", "Brier"],
    cellLoc    = "center",
    loc        = "center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.8)
ax_tbl.set_title("Detailed Metrics", fontsize=11, fontweight="bold")

plt.suptitle(
    "Sepsis Prediction — XGBoost + LightGBM Ensemble | Test Evaluation Dashboard\n"
    f"Test Set: {len(y_test):,} rows  |  Optimised for 16 GB RAM Laptop",
    fontsize=14, fontweight="bold", y=1.005)
fig.savefig(f"{OUT_DIR}/07_summary_dashboard.png", dpi=PLOT_DPI, bbox_inches="tight")
plt.close(); gc.collect()
print("  ✔  Saved: 07_summary_dashboard.png")

# ── Save results CSV ──────────────────────────────────────────────────
rows = []
for nm in model_names:
    r = results[nm]
    rows.append({
        "Model"         : nm,
        "AUC_ROC"       : round(r["AUC"],         4),
        "Avg_Precision" : round(r["AP"],           4),
        "F1"            : round(r["F1"],           4),
        "Sensitivity"   : round(r["Sensitivity"],  4),
        "Specificity"   : round(r["Specificity"],  4),
        "PPV"           : round(r["PPV"],          4),
        "NPV"           : round(r["NPV"],          4),
        "Brier_Score"   : round(r["Brier"],        4),
        "Log_Loss"      : round(r["LogLoss"],      4),
        "Threshold"     : round(r["Threshold"],    4),
        "TP"            : int(r["TP"]),
        "FP"            : int(r["FP"]),
        "TN"            : int(r["TN"]),
        "FN"            : int(r["FN"]),
    })
results_df = pd.DataFrame(rows)
results_df.to_csv(f"{OUT_DIR}/test_results.csv", index=False)
print("  ✔  Saved: test_results.csv")

# ── Final Summary ─────────────────────────────────────────────────────
total_time = time.time() - t0
print("\n" + "═"*65)
print("  ✅  TEST EVALUATION COMPLETE")
print("═"*65)
print(results_df[["Model","AUC_ROC","Avg_Precision","F1",
                   "Sensitivity","Specificity","PPV","NPV"]].to_string(index=False))
print(f"\n  Total evaluation time : {total_time:.1f}s")
print(f"  All plots saved to    : ./{OUT_DIR}/")
print(f"  Results CSV           : ./{OUT_DIR}/test_results.csv")
print(f"  {ram()}")
print("═"*65)