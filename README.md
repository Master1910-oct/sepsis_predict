# SepsisGuard — AI Early Warning System

## Table of Contents
- [Overview](#overview)
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
