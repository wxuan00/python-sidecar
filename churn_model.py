"""
churn_model.py — XGBoost churn prediction.

Model is always trained / scored on the FULL historical dataset so churn
probabilities are stable.  The optional start_date / end_date parameters
only filter which customer rows are returned for *display* in the UI.
Logistic Regression fallback is no longer needed because the full dataset
always provides enough customers for XGBoost.

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

from db import load_transactions_all

# ─── Feature order must match training exactly ───────────────────────────────
FEATURE_NAMES = ["Frequency", "Monetary", "CancellationRate", "AOV", "Lifespan"]

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


def run_churn_prediction(merchant_id: int | None,
                         churn_days: int,
                         start_date: str | None,
                         end_date: str | None) -> dict:
    """
    Predict customer churn using XGBoost on the FULL historical dataset.
    start_date / end_date only filter which predictions are returned for display.
    """
    if SCALER is None:
        raise HTTPException(status_code=503, detail="Churn scaler not loaded.")
    if XGB_MODEL is None:
        raise HTTPException(status_code=503, detail="XGBoost churn model not loaded.")

    # ── Always load ALL data for model scoring ───────────────────────────────
    df = load_transactions_all(merchant_id)
    if df.empty:
        raise HTTPException(status_code=404,
                            detail="No transactions found in the database.")

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
                    f"Not enough history for churn prediction. "
                    f"Database needs at least {churn_days} days of transactions."
                ),
            )

        # 1. Feature engineering on all customers
        rfm = _build_features(historical)
        n   = len(rfm)
        if n < 1:
            raise HTTPException(status_code=422, detail="No customers found after feature engineering.")

        model_name   = "XGBoost (Pre-Trained)"
        model_reason = f"XGBoost selected — full dataset ({n} customers)"

        # 2. Churn label: not seen after cutoff = churned
        active_customers = set(future["card_no"].unique())
        rfm["isChurned"] = (~rfm["card_no"].isin(active_customers)).astype(bool)

        # 3. Scale + predict (inference only — no .fit())
        X        = rfm[FEATURE_NAMES].copy()
        X_scaled = SCALER.transform(X)
        rfm["churnProbability"] = XGB_MODEL.predict_proba(X_scaled)[:, 1]

        # 4. SHAP values
        shap_base_value   = 0.0
        global_importance = {}
        try:
            explainer         = shap.TreeExplainer(XGB_MODEL)
            shap_values       = explainer.shap_values(X_scaled)
            shap_base_value   = round(float(explainer.expected_value), 4)
            global_importance = {
                name: round(float(np.abs(shap_values[:, i]).mean()), 4)
                for i, name in enumerate(FEATURE_NAMES)
            }
            for i, fname in enumerate(FEATURE_NAMES):
                rfm[f"shap_{fname}"] = shap_values[:, i]
        except Exception as shap_err:
            print(f"⚠️  SHAP computation failed: {shap_err}")

        # 5. Attach last-seen date for display filtering
        last_seen = df.groupby("card_no")["txn_date"].max().reset_index()
        last_seen.columns = ["card_no", "lastSeen"]
        rfm = rfm.merge(last_seen, on="card_no", how="left")

        # 6. Build full predictions list
        shap_cols    = [c for c in rfm.columns if c.startswith("shap_")]
        predictions_all = []
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
                "lastSeen":         row["lastSeen"].date().isoformat() if pd.notna(row["lastSeen"]) else None,
            }
            if shap_cols:
                pred["shapBreakdown"] = {
                    col.replace("shap_", ""): round(float(row[col]), 4)
                    for col in shap_cols
                }
            predictions_all.append(pred)

        # 7. Apply date filter for display only
        predictions = predictions_all
        if start_date:
            predictions = [p for p in predictions if p["lastSeen"] and p["lastSeen"] >= start_date]
        if end_date:
            predictions = [p for p in predictions if p["lastSeen"] and p["lastSeen"] <= end_date]

        predictions.sort(key=lambda p: p["churnProbability"], reverse=True)

        # 8. Summary stats always based on full dataset
        high_risk          = sum(1 for p in predictions_all if p["churnProbability"] > 0.7)
        predicted_churners = sum(1 for p in predictions_all if p["churnProbability"] > 0.5)
        churn_rate         = round((predicted_churners / len(predictions_all)) * 100, 2) if predictions_all else 0

        db_min = df["txn_date"].min().date().isoformat()
        db_max = df["txn_date"].max().date().isoformat()

        return {
            "predictions":             predictions,
            "totalCustomers":          n,
            "displayCustomers":        len(predictions),
            "highRiskCount":           high_risk,
            "churnRate":               churn_rate,
            "churnDays":               churn_days,
            "globalFeatureImportance": global_importance,
            "shapBaseValue":           shap_base_value,
            "modelUsed":               model_name,
            "modelSelectedReason":     model_reason,
            "dateRange":               {"from": db_min, "to": db_max},
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Churn prediction failed: {exc}.",
        ) from exc
