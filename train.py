"""
╔══════════════════════════════════════════════════════════════════════╗
║           SEPSIS PREDICTION — TRAINING SCRIPT                       ║
║           train.py  |  Optimised for 16 GB RAM Laptop               ║
╠══════════════════════════════════════════════════════════════════════╣
║  STEP 1 — Install dependencies (once):                              ║
║    pip install xgboost lightgbm scikit-learn pandas numpy           ║
║                matplotlib seaborn psutil joblib                     ║
║                                                                     ║
║  STEP 2 — Train:                                                    ║
║    python train.py                                                  ║
║                                                                     ║
║  OUTPUT FILES SAVED:                                                ║
║    models/xgb_model.json          ← XGBoost model                  ║
║    models/lgb_model.txt           ← LightGBM model                 ║
║    models/imputer.pkl             ← fitted imputer                  ║
║    models/feature_cols.pkl        ← feature column names           ║
║    models/ensemble_weight.pkl     ← optimal ensemble weight        ║
║    models/thresholds.pkl          ← optimal thresholds             ║
║    sepsis_train_outputs/          ← EDA + training plots           ║
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

from sklearn.model_selection import train_test_split
from sklearn.impute          import SimpleImputer
from sklearn.metrics         import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, f1_score,
)

warnings.filterwarnings("ignore")
np.random.seed(42)

# ═════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════
DATA_PATH      = "Dataset.csv"
MODEL_DIR      = "models"
OUT_DIR        = "sepsis_train_outputs"
TARGET         = "SepsisLabel"
DROP_COLS      = ["Unnamed: 0", "Patient_ID"]
MISSING_THRESH = 95
CHUNK_SIZE     = 200_000
PLOT_DPI       = 120

PALETTE = {
    "Sepsis"   : "#E84040",
    "No Sepsis": "#4C72B0",
    "XGBoost"  : "#2196F3",
    "LightGBM" : "#FF9800",
    "Ensemble" : "#4CAF50",
}

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR,   exist_ok=True)

def ram():
    used  = psutil.Process().memory_info().rss / 1e9
    total = psutil.virtual_memory().total / 1e9
    return f"RAM {used:.1f}/{total:.1f} GB"


# ═════════════════════════════════════════════════════════════════════════
# 1.  LOAD DATA
# ═════════════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  [1/6]  Loading Dataset.csv …")
print("═"*65)
t0 = time.time()

chunks = []
for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE, low_memory=False):
    for col in chunk.select_dtypes(include=["float64"]).columns:
        chunk[col] = chunk[col].astype(np.float32)
    for col in chunk.select_dtypes(include=["int64"]).columns:
        chunk[col] = pd.to_numeric(chunk[col], downcast="integer")
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)
del chunks; gc.collect()

mem_mb = df.memory_usage(deep=True).sum() / 1e6
print(f"  ✔  Loaded  {len(df):,} rows × {df.shape[1]} columns")
print(f"  ✔  DataFrame : {mem_mb:.0f} MB  |  {ram()}")

vc       = df[TARGET].value_counts()
pos_rate = vc[1] / len(df) * 100
print(f"\n  Target '{TARGET}':")
print(f"    No Sepsis (0) : {vc[0]:>10,}  ({100-pos_rate:.2f}%)")
print(f"    Sepsis    (1) : {vc[1]:>10,}  ({pos_rate:.2f}%)")
print(f"    Imbalance     : 1 : {vc[0]/vc[1]:.1f}")


# ═════════════════════════════════════════════════════════════════════════
# 2.  EXPLORATORY DATA ANALYSIS (EDA)
# ═════════════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  [2/6]  Exploratory Data Analysis …")
print("═"*65)

numeric_feats = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_feats = [c for c in numeric_feats if c not in [TARGET] + DROP_COLS]
miss_pct      = (df[numeric_feats].isnull().mean() * 100).sort_values(ascending=False)

# Figure 1: EDA Overview
fig = plt.figure(figsize=(18, 11))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)

ax0 = fig.add_subplot(gs[0, 0])
ax0.pie([vc[0], vc[1]],
        labels=["No Sepsis", "Sepsis"],
        colors=[PALETTE["No Sepsis"], PALETTE["Sepsis"]],
        autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2.5),
        textprops={"fontsize": 12})
ax0.set_title(f"Class Distribution\n({len(df):,} rows)", fontsize=13, fontweight="bold")

ax1 = fig.add_subplot(gs[0, 1:])
top25      = miss_pct.head(25)
bar_colors = ["#E84040" if v > MISSING_THRESH else "#4C72B0" for v in top25.values]
ax1.barh(top25.index[::-1], top25.values[::-1], color=bar_colors[::-1], alpha=0.88)
ax1.axvline(MISSING_THRESH, color="red", linestyle="--", lw=1.8,
            label=f"{MISSING_THRESH}% drop threshold")
ax1.set_xlabel("Missing (%)", fontsize=11)
ax1.set_title("Missing Data by Feature (Top 25)", fontsize=13, fontweight="bold")
ax1.legend(fontsize=10)

for idx, col in enumerate([c for c in ["HR", "O2Sat", "SBP"] if c in df.columns]):
    ax = fig.add_subplot(gs[1, idx])
    for lbl, color, name in [(0, PALETTE["No Sepsis"], "No Sepsis"),
                               (1, PALETTE["Sepsis"],    "Sepsis")]:
        vals = df.loc[df[TARGET] == lbl, col].dropna()
        ax.hist(vals, bins=60, alpha=0.65, color=color, density=True, label=name)
    ax.set_title(f"{col} by Sepsis Label", fontsize=11, fontweight="bold")
    ax.set_xlabel(col); ax.set_ylabel("Density"); ax.legend(fontsize=9)

plt.suptitle("Sepsis Dataset — EDA Overview", fontsize=15, fontweight="bold", y=1.01)
fig.savefig(f"{OUT_DIR}/01_eda_overview.png", dpi=PLOT_DPI, bbox_inches="tight")
plt.close(); gc.collect()
print("  ✔  Saved: 01_eda_overview.png")

# Figure 2: Clinical Variables
plot_cols = [c for c in ["HR","O2Sat","SBP","MAP","Resp","Temp","BUN","WBC"]
             if c in df.columns]
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
axes = axes.flatten()
for i, col in enumerate(plot_cols[:8]):
    for lbl, color, name in [(0, PALETTE["No Sepsis"], "No Sepsis"),
                               (1, PALETTE["Sepsis"],    "Sepsis")]:
        vals = df.loc[df[TARGET] == lbl, col].dropna()
        q1, q99 = vals.quantile(0.01), vals.quantile(0.99)
        vals = vals[(vals >= q1) & (vals <= q99)]
        axes[i].hist(vals, bins=55, alpha=0.65, color=color, density=True, label=name)
    axes[i].set_title(col, fontsize=12, fontweight="bold")
    axes[i].legend(fontsize=9)
plt.suptitle("Clinical Variables — Sepsis vs No Sepsis", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/02_clinical_distributions.png", dpi=PLOT_DPI, bbox_inches="tight")
plt.close(); gc.collect()
print("  ✔  Saved: 02_clinical_distributions.png")


# ═════════════════════════════════════════════════════════════════════════
# 3.  FEATURE ENGINEERING & PREPROCESSING
# ═════════════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  [3/6]  Feature Engineering & Preprocessing …")
print("═"*65)

df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

hi_miss = miss_pct[miss_pct > MISSING_THRESH].index.tolist()
df.drop(columns=[c for c in hi_miss if c in df.columns], inplace=True)
print(f"  Dropped {len(hi_miss)} features with >{MISSING_THRESH}% missing")

if "HR"  in df.columns and "SBP" in df.columns:
    df["ShockIndex"]    = (df["HR"] / (df["SBP"] + 1e-6)).astype(np.float32)
    print("  ✔  ShockIndex     = HR / SBP")
if "SBP" in df.columns and "DBP" in df.columns:
    df["PulsePressure"] = (df["SBP"] - df["DBP"]).astype(np.float32)
    print("  ✔  PulsePressure  = SBP - DBP")
if "MAP" in df.columns:
    df["MAP_dev"]       = (df["MAP"] - 93).abs().astype(np.float32)
    print("  ✔  MAP_dev        = |MAP - 93|")
if "Temp" in df.columns:
    df["TempAbnormal"]  = ((df["Temp"] > 38.3) | (df["Temp"] < 36.0)).astype(np.int8)
    print("  ✔  TempAbnormal   = fever/hypothermia flag")
if "HR"   in df.columns:
    df["Tachycardia"]   = (df["HR"] > 90).astype(np.int8)
    print("  ✔  Tachycardia    = HR > 90 bpm")
if "Resp" in df.columns:
    df["Tachypnea"]     = (df["Resp"] > 20).astype(np.int8)
    print("  ✔  Tachypnea      = Resp > 20")

sirs = pd.Series(np.int8(0), index=df.index)
for col in ["TempAbnormal", "Tachycardia", "Tachypnea"]:
    if col in df.columns: sirs = sirs + df[col]
if "WBC" in df.columns:
    sirs = sirs + ((df["WBC"] > 12) | (df["WBC"] < 4)).astype(np.int8)
df["SIRS_Score"] = sirs.astype(np.int8)
print("  ✔  SIRS_Score     = composite 0-4")

if "Age" in df.columns:
    df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 40, 60, 75, 100],
                             labels=[0, 1, 2, 3]).astype(np.float32)
    print("  ✔  AgeGroup       = <40 / 40-60 / 60-75 / 75+")

gc.collect()

feature_cols = [c for c in df.columns if c != TARGET]
X = df[feature_cols].values.astype(np.float32)
y = df[TARGET].values.astype(np.int32)
del df; gc.collect()

print(f"\n  Feature matrix : {X.shape[0]:,} rows × {X.shape[1]} features")
print(f"  {ram()}")

# ── Train / Val split  80 / 20 ───────────────────────────────────────────
# NOTE: test.py uses its own separate test data (the held-out 20%)
#       This split keeps 80% for training (60% train + 20% val internally)
X_tv, X_test_save, y_tv, y_test_save = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=0.25, stratify=y_tv, random_state=42)

# Save held-out test set for test.py
np.save(f"{MODEL_DIR}/X_test.npy",  X_test_save.astype(np.float32))
np.save(f"{MODEL_DIR}/y_test.npy",  y_test_save)
print(f"\n  ✔  Saved held-out test set → models/X_test.npy + y_test.npy")

del X_tv, y_tv, X_test_save, y_test_save; gc.collect()

print(f"\n  Train : {len(X_train):>10,}  rows  (60%)")
print(f"  Val   : {len(X_val):>10,}  rows  (20%)")
print(f"  Test  : held-out 20%  → loaded by test.py")

# ── Impute ───────────────────────────────────────────────────────────────
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train).astype(np.float32)
X_val   = imputer.transform(X_val).astype(np.float32)
gc.collect()

# ── Class weight ─────────────────────────────────────────────────────────
# IMPORTANT: Raw imbalance ratio is ~54 but using full value causes models
# to over-weight positive class and output saturated probabilities for
# even mild cases. Capping at 10 preserves sensitivity while keeping
# raw probability outputs spread across the full 0-1 range.
n_neg = int((y_train == 0).sum())
n_pos = int((y_train == 1).sum())
raw_spw = n_neg / n_pos
scale_pos_weight = min(raw_spw, 10.0)   # cap at 10 — prevents prob saturation
print(f"\n  Raw imbalance ratio : {raw_spw:.1f}")
print(f"  scale_pos_weight    : {scale_pos_weight:.2f}  (capped at 10 for calibration)")


# ═════════════════════════════════════════════════════════════════════════
# 4.  TRAIN — XGBoost
# ═════════════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  [4/6]  Training XGBoost …")
print("═"*65)

xgb_model = xgb.XGBClassifier(
    n_estimators          = 600,
    max_depth             = 7,          # +1 depth: capture organ-failure interactions
    learning_rate         = 0.05,
    subsample             = 0.80,
    colsample_bytree      = 0.80,
    min_child_weight      = 5,          # lower: allows rarer severe patterns to split
    gamma                 = 0.5,        # lower: less pruning, catches severe cases
    reg_alpha             = 0.05,
    reg_lambda            = 1.0,
    scale_pos_weight      = scale_pos_weight,
    tree_method           = "hist",
    device                = "cpu",
    n_jobs                = -1,
    early_stopping_rounds = 30,         # more patience
    eval_metric           = "aucpr",
    random_state          = 42,
    verbosity             = 0,
)

t1 = time.time()
xgb_model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)
xgb_time = time.time() - t1

xgb_val_prob = xgb_model.predict_proba(X_val)[:, 1]
xgb_val_auc  = roc_auc_score(y_val, xgb_val_prob)
xgb_val_ap   = average_precision_score(y_val, xgb_val_prob)

print(f"  ✔  Done in {xgb_time/60:.1f} min  |  "
      f"Best iteration: {xgb_model.best_iteration}")
print(f"  Val  AUC-ROC       : {xgb_val_auc:.4f}")
print(f"  Val  Avg Precision : {xgb_val_ap:.4f}")
print(f"  {ram()}")


# ═════════════════════════════════════════════════════════════════════════
# 5.  TRAIN — LightGBM
# ═════════════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  [5/6]  Training LightGBM …")
print("═"*65)

lgb_model = lgb.LGBMClassifier(
    n_estimators      = 600,
    max_depth         = 7,
    num_leaves        = 80,             # more leaves: richer splits for severity
    learning_rate     = 0.05,
    subsample         = 0.80,
    subsample_freq    = 1,
    colsample_bytree  = 0.80,
    min_child_samples = 20,             # lower: detect rare severe sepsis patterns
    reg_alpha         = 0.05,
    reg_lambda        = 1.0,
    scale_pos_weight  = scale_pos_weight,
    n_jobs            = -1,
    max_bin           = 255,
    random_state      = 42,
    verbose           = -1,
)

t2 = time.time()
lgb_model.fit(
    X_train, y_train,
    eval_set  = [(X_val, y_val)],
    callbacks = [
        lgb.early_stopping(stopping_rounds=20, verbose=False),
        lgb.log_evaluation(period=-1),
    ],
)
lgb_time = time.time() - t2

lgb_val_prob = lgb_model.predict_proba(X_val)[:, 1]
lgb_val_auc  = roc_auc_score(y_val, lgb_val_prob)
lgb_val_ap   = average_precision_score(y_val, lgb_val_prob)

print(f"  ✔  Done in {lgb_time/60:.1f} min  |  "
      f"Best iteration: {lgb_model.best_iteration_}")
print(f"  Val  AUC-ROC       : {lgb_val_auc:.4f}")
print(f"  Val  Avg Precision : {lgb_val_ap:.4f}")
print(f"  {ram()}")


# ═════════════════════════════════════════════════════════════════════════
# 6.  FIND ENSEMBLE WEIGHT + SAVE EVERYTHING
# ═════════════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  [6/6]  Optimising Ensemble & Saving Models …")
print("═"*65)

# Find best ensemble weight on validation set
# Search range 0.2-0.8 — prevents degenerate 1.0/0.0 weight that
# throws away one model entirely and kills probability diversity
best_w, best_auc_ens = 0.5, 0.0
for w in np.arange(0.20, 0.81, 0.05):
    auc = roc_auc_score(y_val, w * xgb_val_prob + (1 - w) * lgb_val_prob)
    if auc > best_auc_ens:
        best_auc_ens, best_w = auc, w

ens_val_prob = best_w * xgb_val_prob + (1 - best_w) * lgb_val_prob

# Compute optimal thresholds on validation set
def best_threshold(y_true, proba):
    prec, rec, thresholds = precision_recall_curve(y_true, proba)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    return float(thresholds[np.argmax(f1s[:-1])])

thresholds = {
    "XGBoost"  : best_threshold(y_val, xgb_val_prob),
    "LightGBM" : best_threshold(y_val, lgb_val_prob),
    "Ensemble" : best_threshold(y_val, ens_val_prob),
}

print(f"  Optimal XGB weight : {best_w:.2f}  (LGB: {1-best_w:.2f})")
print(f"  Val AUC (Ensemble) : {best_auc_ens:.4f}")
print(f"  Thresholds → XGBoost:{thresholds['XGBoost']:.4f}  "
      f"LightGBM:{thresholds['LightGBM']:.4f}  "
      f"Ensemble:{thresholds['Ensemble']:.4f}")

# ── Post-training probability distribution check ─────────────────────
# Verify raw probs span the full range — key sign of healthy calibration
print(f"\n  Raw probability distribution on val set:")
for model_nm, probs in [("XGBoost", xgb_val_prob), ("LightGBM", lgb_val_prob), ("Ensemble", ens_val_prob)]:
    p10, p25, p50, p75, p90 = np.percentile(probs, [10, 25, 50, 75, 90])
    pct_high = (probs > 0.5).mean() * 100
    print(f"  {model_nm:<12}: p10={p10:.3f} p25={p25:.3f} p50={p50:.3f} "
          f"p75={p75:.3f} p90={p90:.3f}  >0.5: {pct_high:.1f}%")

# Save all artefacts
xgb_model.save_model(f"{MODEL_DIR}/xgb_model.json")
lgb_model.booster_.save_model(f"{MODEL_DIR}/lgb_model.txt")
joblib.dump(imputer,      f"{MODEL_DIR}/imputer.pkl")
joblib.dump(feature_cols, f"{MODEL_DIR}/feature_cols.pkl")
joblib.dump(best_w,       f"{MODEL_DIR}/ensemble_weight.pkl")
joblib.dump(thresholds,   f"{MODEL_DIR}/thresholds.pkl")

print(f"\n  ✔  models/xgb_model.json")
print(f"  ✔  models/lgb_model.txt")
print(f"  ✔  models/imputer.pkl")
print(f"  ✔  models/feature_cols.pkl")
print(f"  ✔  models/ensemble_weight.pkl")
print(f"  ✔  models/thresholds.pkl")

# Feature importance plot
xgb_fi = pd.Series(xgb_model.feature_importances_, index=feature_cols)
lgb_fi = pd.Series(lgb_model.feature_importances_, index=feature_cols)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
for ax, fi, nm, color in [(ax1, xgb_fi, "XGBoost",  PALETTE["XGBoost"]),
                           (ax2, lgb_fi, "LightGBM", PALETTE["LightGBM"])]:
    top = fi.sort_values().tail(20)
    ax.barh(top.index, top.values, color=color, alpha=0.88, edgecolor="white")
    ax.set_title(f"Top 20 Feature Importances\n{nm} (Gain-based)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance")
plt.suptitle("Feature Importances — XGBoost vs LightGBM",
             fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/03_feature_importance.png", dpi=PLOT_DPI, bbox_inches="tight")
plt.close(); gc.collect()
print(f"\n  ✔  sepsis_train_outputs/03_feature_importance.png")

# Save feature importances CSV
fi_df = pd.DataFrame({
    "Feature"            : feature_cols,
    "XGBoost_Importance" : xgb_model.feature_importances_,
    "LightGBM_Importance": lgb_model.feature_importances_,
}).sort_values("XGBoost_Importance", ascending=False)
fi_df.to_csv(f"{OUT_DIR}/feature_importances.csv", index=False)
print(f"  ✔  sepsis_train_outputs/feature_importances.csv")

# Validation summary
val_summary = pd.DataFrame([
    {"Model": "XGBoost",  "Val_AUC": round(xgb_val_auc, 4), "Val_AP": round(xgb_val_ap, 4)},
    {"Model": "LightGBM", "Val_AUC": round(lgb_val_auc, 4), "Val_AP": round(lgb_val_ap, 4)},
    {"Model": "Ensemble", "Val_AUC": round(best_auc_ens, 4), "Val_AP": round(
        average_precision_score(y_val, ens_val_prob), 4)},
])
val_summary.to_csv(f"{OUT_DIR}/val_results.csv", index=False)
print(f"  ✔  sepsis_train_outputs/val_results.csv")

total_time = time.time() - t0
print("\n" + "═"*65)
print("  ✅  TRAINING COMPLETE")
print("═"*65)
print(val_summary.to_string(index=False))
print(f"\n  XGBoost  training time : {xgb_time/60:.1f} min")
print(f"  LightGBM training time : {lgb_time/60:.1f} min")
print(f"  Total time             : {total_time/60:.1f} min")
print(f"  {ram()}")
print(f"\n  ➜  Now run:  python test.py")
print("═"*65)