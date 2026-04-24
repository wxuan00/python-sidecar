"""
churn_model.py — Dynamic XGBoost / Logistic Regression churn prediction.

Model selection:
  ≥ 100 customers  →  XGBoost (Pre-Trained)           + real SHAP TreeExplainer
  20–99 customers  →  Logistic Regression (Pre-Trained) + pseudo-SHAP (coef × value)
  < 20 customers   →  HTTP 422 (insufficient data)

Scaler:   models/rfm_scaler.pkl
Features: Frequency, Monetary, CancellationRate, AOV, Lifespan
Note:     Recency excluded — used only to define the churn label.
"""
from __future__ import annotations

from datetime import timedelta

import joblib
import numpy as np
import shap
import pandas as pd
from fastapi import HTTPException

from db import load_transactions

# ─── Feature order must match training exactly ───────────────────────────────
FEATURE_NAMES = ["Frequency", "Monetary", "CancellationRate", "AOV", "Lifespan"]

# ─── Minimum customer thresholds ─────────────────────────────────────────────
MIN_XGB = 100   # XGBoost needs enough rows for reliable tree splits
MIN_LR  =  20   # Logistic Regression handles small samples well

# ─── Load pre-trained artefacts once at import time ──────────────────────────
try:
    SCALER    = joblib.load("models/rfm_scaler.pkl")
    print("✅ Churn scaler loaded.")
except Exception as _e:
    SCALER = None
    print(f"⚠️  Churn scaler not found: {_e}")

try:
    XGB_MODEL = joblib.load("models/xgboost_churn_model.pkl")
    print("✅ XGBoost churn model loaded.")
except Exception as _e:
    XGB_MODEL = None
    print(f"⚠️  XGBoost model not found: {_e}")

try:
    LR_MODEL = joblib.load("models/logistic_churn_model.pkl")
    print("✅ Logistic Regression churn model loaded.")
except Exception as _e:
    LR_MODEL = None
    print(f"⚠️  Logistic Regression model not found: {_e}")


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the 5 Kaggle features per customer (card_no):
      Frequency, Monetary, CancellationRate, AOV, Lifespan
    """
    df = df.copy()
    df["is_cancelled"] = df["status"].isin(["REFUNDED", "REFUND_REQUESTED"]).astype(int)

    agg = df.groupby("card_no").agg(
        first_txn    =("txn_date",       "min"),
        last_txn     =("txn_date",       "max"),
        Frequency    =("transaction_id", "nunique"),
        Monetary     =("amount",         "sum"),
        Cancellations=("is_cancelled",   "sum"),
    ).reset_index()

    agg["Lifespan"]         = (agg["last_txn"] - agg["first_txn"]).dt.days
    agg["AOV"]              = agg["Monetary"] / agg["Frequency"]
    agg["CancellationRate"] = agg["Cancellations"] / agg["Frequency"]
    agg = agg.fillna(0)

    return agg[["card_no", "Frequency", "Monetary", "CancellationRate", "AOV", "Lifespan"]]


def _select_model(n: int) -> tuple:
    """
    Dynamically pick the best available model based on customer count.
    Returns (model, model_name, use_xgb, reason).
    """
    if n >= MIN_XGB and XGB_MODEL is not None:
        return (XGB_MODEL,
                "XGBoost (Pre-Trained)",
                True,
                f"XGBoost selected — {n} customers ≥ threshold of {MIN_XGB}")
    if n >= MIN_LR and LR_MODEL is not None:
        return (LR_MODEL,
                "Logistic Regression (Pre-Trained)",
                False,
                f"Logistic Regression selected — {n} customers is below XGBoost threshold of {MIN_XGB}")
    if n < MIN_LR:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Insufficient data: only {n} unique customers found. "
                f"Minimum required: {MIN_LR} (Logistic Regression) "
                f"or {MIN_XGB} (XGBoost). "
                f"Please widen the date range."
            ),
        )
    raise HTTPException(status_code=503, detail="No churn model is loaded.")


def run_churn_prediction(merchant_id: int | None,
                         churn_days: int,
                         start_date: str | None,
                         end_date: str | None) -> dict:
    """
    Predict customer churn with dynamic model selection:
      ≥100 customers → XGBoost + real SHAP
      20–99          → Logistic Regression + pseudo-SHAP
      <20            → HTTP 422
    """
    if SCALER is None:
        raise HTTPException(status_code=503, detail="Churn scaler not loaded.")

    df = load_transactions(merchant_id, start_date, end_date)
    if df.empty:
        raise HTTPException(status_code=404,
                            detail="No transactions found for the selected date range.")

    df = df.dropna(subset=["card_no"])

    try:
        max_date   = df["txn_date"].max()
        cutoff     = max_date - timedelta(days=churn_days)
        historical = df[df["txn_date"] <= cutoff]
        future     = df[df["txn_date"] >  cutoff]

        if historical.empty:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Date range too narrow for churn prediction. "
                    f"A minimum span of {churn_days} days is needed. "
                    f"Please widen the date range or reduce the churn window."
                ),
            )

        # 1. Feature engineering
        rfm = _build_features(historical)
        n   = len(rfm)

        # 2. Dynamic model selection
        model, model_name, use_xgb, model_reason = _select_model(n)
        print(f"ℹ️  {model_reason}")

        # 3. Churn label: not seen after cutoff = churned
        active_customers = set(future["card_no"].unique())
        rfm["isChurned"] = (~rfm["card_no"].isin(active_customers)).astype(bool)

        # 4. Scale + predict (inference only — no .fit())
        X        = rfm[FEATURE_NAMES].copy()
        X_scaled = SCALER.transform(X)
        rfm["churnProbability"] = model.predict_proba(X_scaled)[:, 1]

        # 5. SHAP values
        shap_base_value   = 0.0
        global_importance = {}
        try:
            if use_xgb:
                # Real SHAP via TreeExplainer
                explainer     = shap.TreeExplainer(model)
                shap_values   = explainer.shap_values(X_scaled)
                shap_base_value   = round(float(explainer.expected_value), 4)
                global_importance = {
                    name: round(float(np.abs(shap_values[:, i]).mean()), 4)
                    for i, name in enumerate(FEATURE_NAMES)
                }
                for i, fname in enumerate(FEATURE_NAMES):
                    rfm[f"shap_{fname}"] = shap_values[:, i]
            else:
                # Pseudo-SHAP via LR coefficients
                coefs             = model.coef_[0]
                shap_base_value   = round(float(model.intercept_[0]), 4)
                global_importance = {
                    name: round(float(abs(val)), 4)
                    for name, val in zip(FEATURE_NAMES, coefs)
                }
                for i, fname in enumerate(FEATURE_NAMES):
                    rfm[f"shap_{fname}"] = X_scaled[:, i] * coefs[i]
        except Exception as shap_err:
            print(f"⚠️  SHAP computation failed: {shap_err}")

        # 6. Build response list
        predictions = []
        shap_cols   = [c for c in rfm.columns if c.startswith("shap_")]
        for _, row in rfm.iterrows():
            pred = {
                "cardNo":           row["card_no"],
                "frequency":        int(row["Frequency"]),
                "monetary":         round(float(row["Monetary"]), 2),
                "cancellationRate": round(float(row["CancellationRate"]), 4),
                "aov":              round(float(row["AOV"]), 2),
                "lifespan":         int(row["Lifespan"]),
                "churnProbability": round(float(row["churnProbability"]), 4),
                "isChurned":        bool(row["isChurned"]),
            }
            if shap_cols:
                pred["shapBreakdown"] = {
                    col.replace("shap_", ""): round(float(row[col]), 4)
                    for col in shap_cols
                }
            predictions.append(pred)

        predictions.sort(key=lambda p: p["churnProbability"], reverse=True)

        high_risk          = sum(1 for p in predictions if p["churnProbability"] > 0.7)
        predicted_churners = sum(1 for p in predictions if p["churnProbability"] > 0.5)
        churn_rate         = round((predicted_churners / len(predictions)) * 100, 2) if predictions else 0

        return {
            "predictions":             predictions,
            "totalCustomers":          n,
            "highRiskCount":           high_risk,
            "churnRate":               churn_rate,
            "churnDays":               churn_days,
            "globalFeatureImportance": global_importance,
            "shapBaseValue":           shap_base_value,
            "modelUsed":               model_name,
            "modelSelectedReason":     model_reason,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Churn prediction failed: {exc}. Try widening the date range (churn window: {churn_days} days).",
        ) from exc
