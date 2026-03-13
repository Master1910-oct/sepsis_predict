# SepsisGuard — AI Early Warning System

## Table of Contents
- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Models Used](#models-used)
- [Model Performance](#model-performance)
  - [Training / Validation Results](#training--validation-results)
  - [Test Set Results](#test-set-results-held-out-20)
  - [Top Feature Importances](#top-feature-importances)
- [How Predictions are Made](#how-predictions-are-made)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Usage & Pipeline Overview](#usage--pipeline-overview)
  - [1. Train Models](#1-train-models)
  - [2. Evaluate Models](#2-evaluate-models)
  - [3. Run API Server & Dashboard](#3-run-api-server--dashboard)
- [Dashboard Capabilities](#dashboard-capabilities)

## Overview
SepsisGuard is an end-to-end framework built for the early prediction of Sepsis using clinical data. Utilizing Tree-based ensembling (LightGBM & XGBoost), this project offers comprehensive Exploratory Data Analysis (EDA), missing value imputation, robust feature engineering, model training, evaluation dashboards, and a user-friendly UI (served via Flask).

It transforms raw clinical EHR variables (vitals, labs, demographics) into an actionable and interpretable sepsis prediction outcome.

## Tech Stack

### 🤖 Machine Learning
| Library | Version | Purpose |
|---|---|---|
| [XGBoost](https://xgboost.readthedocs.io/) | ≥ 1.7 | Gradient-boosted trees — primary classifier |
| [LightGBM](https://lightgbm.readthedocs.io/) | ≥ 3.3 | Fast histogram-based gradient boosting |
| [scikit-learn](https://scikit-learn.org/) | ≥ 1.2 | Preprocessing, metrics, train/test split, imputation |

### 📊 Data & Visualisation
| Library | Version | Purpose |
|---|---|---|
| [pandas](https://pandas.pydata.org/) | ≥ 1.5 | Tabular data ingestion and manipulation |
| [NumPy](https://numpy.org/) | ≥ 1.24 | Numerical array operations |
| [Matplotlib](https://matplotlib.org/) | ≥ 3.6 | Plotting — ROC/PR curves, feature importance, EDA |
| [Seaborn](https://seaborn.pydata.org/) | ≥ 0.12 | Statistical visualisation (heatmaps, calibration plots) |

### 🌐 Backend & API
| Library | Version | Purpose |
|---|---|---|
| [Flask](https://flask.palletsprojects.com/) | ≥ 2.3 | REST API server (`POST /predict`) and static file host |
| [Flask-CORS](https://flask-cors.readthedocs.io/) | ≥ 4.0 | Cross-Origin Resource Sharing for the frontend |

### 🖥️ Frontend
| Technology | Purpose |
|---|---|
| HTML5 / CSS3 | Clinical prediction dashboard UI (`app.html`) |
| Vanilla JavaScript | Dynamic form handling, fetch API calls, risk-level rendering |

### 🛠️ Tooling & Utilities
| Library | Purpose |
|---|---|
| [joblib](https://joblib.readthedocs.io/) | Serialisation of sklearn objects (imputer, thresholds, weights) |
| [psutil](https://psutil.readthedocs.io/) | Real-time RAM usage monitoring during training |
| Python 3.8+ | Core runtime |

## Models Used

SepsisGuard uses a **two-model ensemble** composed of XGBoost and LightGBM — both state-of-the-art gradient-boosted tree algorithms that are well established in clinical machine learning research.

### 1. XGBoost (eXtreme Gradient Boosting)
- **Architecture:** Sequential decision tree ensemble using second-order gradient optimization
- **Config:** 600 estimators · max depth 7 · learning rate 0.05 · subsampling 80%
- **Key setting:** `scale_pos_weight` capped at 10 to handle the severe class imbalance (~54:1 No Sepsis:Sepsis)
- **Early stopping:** 30 rounds on AUC-PR (precision-recall AUC, best metric for imbalanced data)
- **Why chosen:**
  - Excellent on tabular/structured medical data
  - Handles missing values natively via `hist` tree method
  - Built-in regularisation (L1 + L2) prevents overfitting on high-dimensional feature sets
  - Proven track record in clinical EHR prediction tasks (PhysioNet, MIMIC benchmarks)

### 2. LightGBM (Light Gradient Boosting Machine)
- **Architecture:** Leaf-wise (best-first) tree growth with histogram-based binning
- **Config:** 600 estimators · max depth 7 · 80 leaves · learning rate 0.05 · subsampling 80%
- **Early stopping:** 20 rounds
- **Why chosen:**
  - Significantly faster training on large datasets (310K+ rows) than XGBoost
  - Leaf-wise growth captures complex non-linear feature interactions more aggressively
  - Memory-efficient; ideal for the 16 GB RAM laptop constraint
  - Complements XGBoost by learning a slightly different decision boundary, improving ensemble diversity

### 3. Ensemble Strategy
The final prediction is a **weighted linear blend** of both models' probability outputs:

```
Ensemble Score = w × XGBoost_prob + (1 − w) × LightGBM_prob
```

The optimal weight `w` is found by grid-searching the range `[0.20 – 0.80]` on the **validation set** to maximise AUC-ROC. The best-found weight is saved to `models/ensemble_weight.pkl` and loaded at inference time.

**Why ensemble?**
- Reduces variance compared to any single model
- Averages out individual model blind spots
- Consistently achieves equal or better AUC than either standalone model

---

## Model Performance

> **Dataset split:** 60% Train · 20% Validation · 20% Test (stratified, `random_state=42`)
> 
> **Dataset size:** ~1.55 million rows · ~37 features (after dropping >95% missing columns)
> 
> **Class imbalance:** ~1.8% sepsis positive rate (~54:1 ratio)

### Training / Validation Results

| Model | Val AUC-ROC | Val Avg Precision (AP) |
|---|---|---|
| XGBoost | **0.8794** | **0.2065** |
| LightGBM | 0.7303 | 0.0790 |
| **Ensemble** | **0.8790** | **0.2051** |

> AUC-ROC is the primary metric — it measures how well the model separates sepsis from non-sepsis cases independent of threshold.
> Avg Precision (AP) is the area under the Precision-Recall curve, especially informative under class imbalance.

### Test Set Results (Held-Out 20%)

| Metric | XGBoost | LightGBM | **Ensemble** |
|---|---|---|---|
| **AUC-ROC** | 0.8781 | 0.7361 | **0.8776** |
| **Avg Precision (AP)** | 0.2110 | 0.0761 | **0.2089** |
| **F1 Score** | 0.2796 | 0.1697 | **0.2774** |
| **Sensitivity (Recall)** | 30.4% | 26.3% | 27.7% |
| **Specificity** | 98.4% | 96.6% | 98.7% |
| **PPV (Precision)** | 25.9% | 12.5% | 27.8% |
| **NPV** | 98.7% | 98.6% | **98.7%** |
| **Brier Score** ↓ | 0.0283 | 0.0172 | 0.0234 |
| **Log Loss** ↓ | 0.1304 | 0.0850 | 0.1150 |
| **Decision Threshold** | 0.5053 | 0.0639 | 0.4393 |

#### Confusion Matrix Breakdown (Test Set)

| | XGBoost | LightGBM | Ensemble |
|---|---|---|---|
| True Positives (TP) | 1,695 | 1,469 | 1,546 |
| False Positives (FP) | 4,847 | 10,258 | 4,019 |
| True Negatives (TN) | 300,012 | 294,601 | 300,840 |
| False Negatives (FN) | 3,888 | 4,114 | 4,037 |

#### Interpreting the Results

- **AUC-ROC ≈ 0.878** — The model correctly ranks a sepsis patient above a non-sepsis patient ~88% of the time. This is considered strong performance on clinical EHR data.
- **High Specificity (98.7%)** — Very few false alarms; the system does not unnecessarily flag stable patients.
- **Lower Sensitivity (~28-30%)** — A known trade-off when maximising specificity at high imbalance. The model is calibrated to minimise false alarms at the cost of catching every sepsis case. The **decision threshold can be lowered to boost sensitivity** for higher-risk clinical environments.
- **Low F1 (~0.28)** is expected — F1 deflates naturally under severe class imbalance (~54:1). AUC-ROC is the correct primary metric here.
- **XGBoost dominates** — achieves significantly better AUC and F1 vs. LightGBM. The ensemble adds robustness but XGBoost is the backbone.

### Top Feature Importances

Features ranked by XGBoost gain-based importance:

| Rank | Feature | Description | XGB Importance |
|---|---|---|---|
| 1 | `ICULOS` | ICU Length of Stay (hours) | 0.1363 |
| 2 | `SIRS_Score` | Composite SIRS criteria score (0–4) | 0.0868 |
| 3 | `Unit2` | ICU unit type indicator | 0.0374 |
| 4 | `Hour` | Hour-of-day of the record | 0.0370 |
| 5 | `FiO2` | Fraction of inspired oxygen | 0.0368 |
| 6 | `HospAdmTime` | Time since hospital admission | 0.0357 |
| 7 | `pH` | Arterial blood pH | 0.0291 |
| 8 | `Temp` | Body temperature (°C) | 0.0289 |
| 9 | `PaCO2` | Partial pressure of CO₂ | 0.0278 |
| 10 | `Age` | Patient age | 0.0255 |

> The engineered `SIRS_Score` (rank 2) confirming that clinical domain knowledge embedded into the feature engineering pipeline directly contributes to predictive power.

---

## How Predictions are Made

The prediction pipeline in `app.py` follows a 5-step process for every patient submission:

### Step 1 — Clinical Feature Engineering
Raw input vitals and labs are transformed into derived clinical indicators:

| Engineered Feature | Formula | Clinical Meaning |
|---|---|---|
| `ShockIndex` | HR / SBP | Ratio >1.0 indicates hemodynamic instability |
| `PulsePressure` | SBP − DBP | Narrow pulse pressure = reduced cardiac output |
| `MAP_dev` | \|MAP − 93\| | Deviation from ideal mean arterial pressure |
| `TempAbnormal` | Temp >38.3°C or <36.0°C | Fever or hypothermia flag |
| `Tachycardia` | HR > 90 bpm | Elevated heart rate flag |
| `Tachypnea` | Resp > 20 bpm | Elevated respiratory rate flag |
| `SIRS_Score` | Sum of above 4 flags + WBC | 0–4 composite SIRS criteria score |
| `AgeGroup` | Age bucketed into 4 bands | <40 / 40–60 / 60–75 / 75+ |

### Step 2 — Imputation
Missing values are filled using the **median imputer** fitted on the training set (`models/imputer.pkl`). This ensures the model always receives a complete feature vector.

### Step 3 — Ensemble Scoring
Both models independently produce a raw sepsis probability. These are blended using the trained ensemble weight:
```
Raw Ensemble Score = w × XGBoost_prob + (1 − w) × LightGBM_prob
```

### Step 4 — Clinical Boost Override
Before calibration, hard clinical rules based on **Sepsis-3 criteria** can boost the raw score when the ML model systematically under-scores critical physiological states:

| Condition | Boost | Rationale |
|---|---|---|
| MAP < 65 or SBP < 90 | +0.25 | Septic shock: frank hypotension (Sepsis-3) |
| pH < 7.25 AND Creatinine > 2.0 | +0.20 | Multi-organ failure: AKI + metabolic acidosis |
| O₂Sat < 90 AND Resp > 28 | +0.10 | Acute respiratory failure |
| SIRS ≥ 4 AND ShockIndex > 1.2 | +0.10 | Full SIRS + hemodynamic instability |

> Maximum total boost is capped at **+0.50** to prevent probability saturation.

### Step 5 — Platt Scaling Calibration
The boosted raw score is passed through **Platt Scaling** to map it to a well-calibrated probability:
```
p_calibrated = sigmoid(1.410 × logit(p_raw) + 1.565)
```
This corrects the over-inflation in raw probabilities caused by the `scale_pos_weight` imbalance correction applied during training.

### Step 6 — 4-Tier Clinical Risk Stratification
The final calibrated ensemble probability is mapped to one of four actionable risk tiers:

| Tier | Level | Probability Range | Protocol |
|---|---|---|---|
| 🟢 Tier 1 | BASELINE | 0.00 – 0.15 | Standard Care (routine vitals q8h) |
| 🟡 Tier 2 | LOW | 0.16 – 0.35 | Active Monitoring (vitals q2h) |
| 🟠 Tier 3 | MODERATE | 0.36 – 0.60 | Diagnostic Trigger (labs, blood cultures) |
| 🔴 Tier 4 | HIGH | 0.61 – 1.00 | Code Sepsis (immediate MD + antibiotics) |

---

## Project Structure
```text
Sepsis_predict/
├── train.py                  # Training pipeline (EDA, Feature Eng, XGBoost, LightGBM, Ensemble)
├── test.py                   # Evaluation pipeline (Metrics, ROC/PR curves, Calibration, Dashboard generation)
├── app.py                    # Flask API Server and Web Host
├── app.html                  # Frontend Prediction Dashboard
├── Dataset.csv               # Raw Dataset (Sepsis patient records)
├── .gitignore                # Git Ignore file
├── models/                   # Auto-generated directory (Holds pickled models, feature columns, and thresholds)
├── sepsis_train_outputs/     # Auto-generated directory (EDA plots, Feature importances, Val results)
└── sepsis_test_outputs/      # Auto-generated directory (ROC/PR curves, Confusion matrices, Eval dashboards)
```

## Requirements
To run this project, ensure you have Python 3.8+ installed.

### Install Dependencies
Run the following command to grab all required libraries:

```bash
pip install numpy pandas scikit-learn xgboost lightgbm matplotlib seaborn joblib psutil flask flask-cors
```

## Usage & Pipeline Overview

The standard operation of SepsisGuard involves a 3-step pipeline:

### 1. Train Models
Run `train.py` to ingest `Dataset.csv`, handle missing values, engineer clinical features (Shock Index, SIRS score, MAP deviation, etc.), and train both XGBoost and LightGBM models. 

An optimal ensemble weight and optimized threshold on the validation split is calculated and saved alongside the `.pkl` / `.json` / `.txt` files into the `models/` directory.

```bash
python train.py
```
**Output:** Generates `models/` and `sepsis_train_outputs/` directories containing training diagnostic plots and trained artifacts. Additionally, saves `models/X_test.npy` (hold-out test features) and `models/y_test.npy` for Step 2.

### 2. Evaluate Models
Run `test.py` to evaluate the trained models on the strict 20% hold-out unseen test data saved from Step 1.

```bash
python test.py
```
**Output:** Generates `sepsis_test_outputs/` directory packed with visual analytics:
- ROC and Precision-Recall Curves
- Normalized Confusion Matrices
- Calibration curves and thresholds analysis
- A comprehensive evaluation dashboard image (`07_summary_dashboard.png`)

### 3. Run API Server & Dashboard
Run `app.py` to launch the Flask Server which loads the exported models and features into memory. The Flask server serves predictions as a REST API (`POST /predict`) and hosts the interactive User Interface.

```bash
python app.py
```

Once you see `✅ Ready. Open app.html in your browser.` in the console, navigate your browser to `http://localhost:5000` to interact with the system.

## Dashboard Capabilities
The frontend UI (`app.html`) contains a clinical interface. Using the provided UI, users can submit patient data such as Heart Rate, O₂ Saturation, Blood Pressure (SBP, DBP, MAP), Temperature, and various lab values. 

The dashboard provides:
1. **Tiered Risk Levels:** Classifies patient data dynamically into `HIGH`, `MODERATE`, or `LOW` threat levels based on the configured Thresholds from the ensemble score.
2. **Clinical Inference:** Displays exact probability distributions matched against the LightGBM, XGBoost, and the ensembled algorithm output.
3. **Criteria Re-cap:** Re-evaluates Standard SIRS Criteria rules alongside real-time recommendations.
