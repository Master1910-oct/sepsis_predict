"""
╔══════════════════════════════════════════════════════════════════════╗
║           SEPSIS PREDICTION — FLASK API SERVER                       ║
║           app.py  |  Serves predictions to the Dashboard             ║
╠══════════════════════════════════════════════════════════════════════╣
║  REQUIREMENT: Run train.py first to generate models/ folder.         ║
║                                                                      ║
║  RUN:                                                                ║
║    python app.py                                                     ║
║    Then open app.html in your browser.                               ║
║                                                                      ║
║  API ENDPOINT:                                                       ║
║    POST http://localhost:5000/predict                                ║
║    Body: JSON with patient clinical values                           ║
║    Returns: XGBoost, LightGBM, Ensemble probabilities + tier        ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os, warnings
import numpy  as np
import joblib
import xgboost  as xgb
import lightgbm as lgb
from flask import Flask, request, jsonify
from flask_cors import CORS

warnings.filterwarnings("ignore")

MODEL_DIR  = "models"

# ── Required model artefacts ─────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # allow app.html (opened as file://) to hit the API

# ═════════════════════════════════════════════════════════════════════
# LOAD MODELS AT STARTUP
# ═════════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  SepsisGuard Flask API — Loading Models …")
print("═"*65)

required = ["xgb_model.json", "lgb_model.txt", "imputer.pkl",
            "feature_cols.pkl", "ensemble_weight.pkl", "thresholds.pkl"]
missing  = [f for f in required if not os.path.exists(f"{MODEL_DIR}/{f}")]
if missing:
    print("\n  ❌  ERROR: Missing files in models/ folder:")
    for f in missing:
        print(f"       — {f}")
    print("\n  ➜  Please run train.py first!\n")
    raise SystemExit(1)

# XGBoost
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(f"{MODEL_DIR}/xgb_model.json")
print("  ✔  XGBoost  model loaded")

# LightGBM
lgb_booster = lgb.Booster(model_file=f"{MODEL_DIR}/lgb_model.txt")
print("  ✔  LightGBM model loaded")

# Preprocessing artefacts
imputer      = joblib.load(f"{MODEL_DIR}/imputer.pkl")
feature_cols = joblib.load(f"{MODEL_DIR}/feature_cols.pkl")
ensemble_w   = joblib.load(f"{MODEL_DIR}/ensemble_weight.pkl")
thresholds   = joblib.load(f"{MODEL_DIR}/thresholds.pkl")

print(f"  ✔  Feature columns  : {len(feature_cols)}")
print(f"  ✔  Ensemble weight  : XGBoost={ensemble_w:.2f}  LightGBM={1-ensemble_w:.2f}")
print(f"  ✔  Thresholds       : XGB={thresholds['XGBoost']:.4f}  "
      f"LGB={thresholds['LightGBM']:.4f}  "
      f"Ens={thresholds['Ensemble']:.4f}")
print("═"*65)
print("  ✅  Ready.  Open app.html in your browser.")
print("═"*65 + "\n")


# ═════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING  (mirrors train.py exactly)
# ═════════════════════════════════════════════════════════════════════
def build_feature_vector(data: dict) -> np.ndarray:
    """
    Takes a flat dict of raw clinical values, engineers the same
    derived features that were created during training, and returns
    a 2-D float32 numpy array shaped (1, n_features) ready for
    model.predict_proba().
    """
    def safe(key, default=np.nan):
        v = data.get(key)
        if v is None or v == "":
            return default
        try:
            return float(v)
        except (ValueError, TypeError):
            return default

    row = {}

    # ── Raw clinical fields ─────────────────────────────────────────
    for col in feature_cols:
        row[col] = safe(col)

    # ── Derived features (replicate train.py logic) ─────────────────
    HR   = safe("HR")
    SBP  = safe("SBP")
    DBP  = safe("DBP")
    MAP  = safe("MAP")
    Temp = safe("Temp")
    Resp = safe("Resp")
    WBC  = safe("WBC")
    Age  = safe("Age")

    if "ShockIndex" in feature_cols:
        row["ShockIndex"] = (HR / (SBP + 1e-6)
                             if not np.isnan(HR) and not np.isnan(SBP)
                             else np.nan)

    if "PulsePressure" in feature_cols:
        row["PulsePressure"] = (SBP - DBP
                                if not np.isnan(SBP) and not np.isnan(DBP)
                                else np.nan)

    if "MAP_dev" in feature_cols:
        row["MAP_dev"] = abs(MAP - 93) if not np.isnan(MAP) else np.nan

    temp_abn  = int((Temp > 38.3) or (Temp < 36.0)) if not np.isnan(Temp) else 0
    tachy_c   = int(HR   > 90)                       if not np.isnan(HR)   else 0
    tachy_p   = int(Resp > 20)                       if not np.isnan(Resp) else 0

    if "TempAbnormal" in feature_cols:
        row["TempAbnormal"] = temp_abn
    if "Tachycardia" in feature_cols:
        row["Tachycardia"] = tachy_c
    if "Tachypnea" in feature_cols:
        row["Tachypnea"] = tachy_p

    sirs = temp_abn + tachy_c + tachy_p
    if not np.isnan(WBC):
        sirs += int((WBC > 12) or (WBC < 4))
    if "SIRS_Score" in feature_cols:
        row["SIRS_Score"] = sirs

    if "AgeGroup" in feature_cols:
        if not np.isnan(Age):
            row["AgeGroup"] = (0 if Age < 40 else
                               1 if Age < 60 else
                               2 if Age < 75 else 3)
        else:
            row["AgeGroup"] = np.nan

    # Build ordered array in exactly the same column order as training
    vector = np.array([[row.get(c, np.nan) for c in feature_cols]],
                      dtype=np.float32)
    # Impute remaining NaNs
    vector = imputer.transform(vector).astype(np.float32)
    return vector


# ═════════════════════════════════════════════════════════════════════
# ROUTES
# ═════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Serve the dashboard HTML."""
    try:
        with open("app.html", "r", encoding="utf-8") as fh:
            return fh.read(), 200, {"Content-Type": "text/html"}
    except FileNotFoundError:
        return "app.html not found — make sure you run app.py from the project folder.", 404


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON body with patient clinical values.
    Returns a JSON response with:
      xgb_prob, lgb_prob, ens_prob
      xgb_pred, lgb_pred, ens_pred  (0/1 using optimal thresholds)
      tier  — 'HIGH' | 'MODERATE' | 'LOW'
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON body received"}), 400

        X = build_feature_vector(data)

        # Predictions
        xgb_prob = float(xgb_model.predict_proba(X)[0, 1])
        lgb_prob = float(lgb_booster.predict(X)[0])
        ens_prob = float(ensemble_w * xgb_prob + (1 - ensemble_w) * lgb_prob)

        # Binary decisions using training-optimised thresholds
        xgb_pred = int(xgb_prob >= thresholds["XGBoost"])
        lgb_pred = int(lgb_prob >= thresholds["LightGBM"])
        ens_pred = int(ens_prob >= thresholds["Ensemble"])

        # Clinical tier (mirrors app.html thresholds)
        if ens_prob >= 0.45:
            tier = "HIGH"
        elif ens_prob >= 0.20:
            tier = "MODERATE"
        else:
            tier = "LOW"

        return jsonify({
            "xgb_prob"  : round(xgb_prob, 4),
            "lgb_prob"  : round(lgb_prob, 4),
            "ens_prob"  : round(ens_prob, 4),
            "xgb_pred"  : xgb_pred,
            "lgb_pred"  : lgb_pred,
            "ens_pred"  : ens_pred,
            "tier"      : tier,
            "thresholds": {k: round(v, 4) for k, v in thresholds.items()},
            "ensemble_weight": {
                "XGBoost" : round(float(ensemble_w), 4),
                "LightGBM": round(1 - float(ensemble_w), 4),
            },
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Quick liveness check."""
    return jsonify({
        "status"       : "ok",
        "models_loaded": True,
        "n_features"   : len(feature_cols),
    })


# ═════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)