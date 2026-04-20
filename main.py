"""
MSP AI Sidecar — FastAPI service that reads transaction data from PostgreSQL,
runs XGBoost / Logistic Regression churn prediction (Kaggle-benchmarked),
Prophet cash-flow forecasting, and K-Means RFM segmentation,
then returns JSON results to the Spring Boot backend.

Start:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Environment variables (or .env file):
    DB_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/msp_db
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prophet import Prophet
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sqlalchemy import create_engine, text
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")
load_dotenv()

# ─── Database ───────────────────────────────────────────────────────────────

DB_URL = os.getenv("DB_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/msp_db")

def get_engine():
    return create_engine(DB_URL)


XGBOOST_THRESHOLD = 200  # min customers to use XGBoost; below this → Logistic Regression


def load_transactions(merchant_id: int | None = None,
                      start_date: str | None = None,
                      end_date: str | None = None) -> pd.DataFrame:
    """Load transactions from the MSP database (all terminal statuses)."""
    engine = get_engine()
    query = """
        SELECT
            t.transaction_id,
            t.merchant_id,
            t.card_no,
            t.amount,
            t.nett_amount,
            t.status,
            t.txn_date,
            t.payment_channel,
            t.currency
        FROM transactions t
        WHERE t.status IN ('APPROVED', 'REFUNDED', 'REFUND_REQUESTED')
          AND t.txn_date IS NOT NULL
    """
    if merchant_id:
        query += f" AND t.merchant_id = {merchant_id}"
    if start_date:
        query += f" AND t.txn_date >= '{start_date}'"
    if end_date:
        query += f" AND t.txn_date <= '{end_date} 23:59:59'"
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, parse_dates=["txn_date"])
    return df


def _build_rfm(df: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    """
    Build classic RFM features per customer (card_no):
      - Recency:   days since last transaction
      - Frequency: number of distinct transactions
      - Monetary:  total spend
    """
    agg = df.groupby("card_no").agg(
        last_txn=("txn_date", "max"),
        frequency=("transaction_id", "nunique"),
        monetary=("amount", "sum"),
    ).reset_index()

    agg["recency"] = (snapshot - agg["last_txn"]).dt.days

    return agg[["card_no", "recency", "frequency", "monetary"]]


# ─── App ────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MSP AI Sidecar",
    description="XGBoost Churn · Prophet Forecasting · K-Means RFM Segmentation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8001", "http://localhost:4201"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


# ─── RFM Segmentation (K-Means) ─────────────────────────────────────────────

@app.get("/rfm")
def rfm_segmentation(merchant_id: int | None = None,
                     start_date: str | None = None,
                     end_date: str | None = None):
    """
    Compute RFM scores and K-Means customer segments.
    Uses card_no as a proxy for customer ID.

    Returns:
      - segments: list of {cardNo, recency, frequency, monetary, cluster, label}
      - clusterSummary: aggregate stats per cluster
      - silhouetteScore: clustering quality (0–1)
      - totalCustomers: int
    """
    df = load_transactions(merchant_id, start_date, end_date)
    if df.empty:
        raise HTTPException(status_code=404, detail="No transactions found for the selected date range.")

    # Drop rows without card number
    df = df.dropna(subset=["card_no"])
    if len(df) < 10:
        raise HTTPException(status_code=422, detail=f"Not enough data for segmentation — only {len(df)} transaction(s) found in the selected range. Please widen the date range.")

    snapshot = df["txn_date"].max() + timedelta(days=1)

    rfm = df.groupby("card_no").agg(
        recency=("txn_date", lambda x: (snapshot - x.max()).days),
        frequency=("transaction_id", "nunique"),
        monetary=("amount", "sum"),
    ).reset_index()

    # Log-transform + scale
    rfm_log = np.log1p(rfm[["recency", "frequency", "monetary"]])
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)

    n_clusters = min(4, len(rfm))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

    # Silhouette (only if enough data)
    sil_score = None
    if len(rfm) > n_clusters:
        from sklearn.metrics import silhouette_score
        sil_score = round(float(silhouette_score(rfm_scaled, rfm["cluster"])), 3)

    # Label each cluster based on centroid characteristics
    cluster_stats = rfm.groupby("cluster").agg(
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
        count=("card_no", "count"),
    ).reset_index()

    # Sort: low recency + high frequency + high monetary = best
    cluster_stats["score"] = (
        -cluster_stats["avg_recency"] * 0.3
        + cluster_stats["avg_frequency"] * 0.35
        + cluster_stats["avg_monetary"] * 0.35
    )
    cluster_stats_sorted = cluster_stats.sort_values("score", ascending=False).reset_index(drop=True)
    rank_labels = ["Champions", "Loyal Customers", "At Risk", "Lost Customers"]
    label_map = {
        int(row["cluster"]): rank_labels[min(i, len(rank_labels) - 1)]
        for i, row in cluster_stats_sorted.iterrows()
    }
    rfm["label"] = rfm["cluster"].map(label_map)

    # Build response
    segments = rfm.rename(columns={"card_no": "cardNo"}).to_dict(orient="records")
    for s in segments:
        s["recency"] = int(s["recency"])
        s["frequency"] = int(s["frequency"])
        s["monetary"] = round(float(s["monetary"]), 2)
        s["cluster"] = int(s["cluster"])

    cluster_summary = []
    for _, row in cluster_stats.iterrows():
        cluster_summary.append({
            "cluster": int(row["cluster"]),
            "label": label_map[int(row["cluster"])],
            "count": int(row["count"]),
            "avgRecency": round(float(row["avg_recency"]), 1),
            "avgFrequency": round(float(row["avg_frequency"]), 1),
            "avgMonetary": round(float(row["avg_monetary"]), 2),
        })

    return {
        "segments": segments,
        "clusterSummary": cluster_summary,
        "silhouetteScore": sil_score,
        "totalCustomers": len(rfm),
        "snapshotDate": df["txn_date"].max().date().isoformat(),
    }


# ─── Churn Prediction (XGBoost / Logistic Regression — Kaggle Benchmarked) ──
# Kaggle benchmark (Online Retail II UCI):
#   LR  → 72.27% accuracy, 0.7922 ROC-AUC
#   XGB → 74.79% accuracy, 0.8086 ROC-AUC
# Split: 80 train / 10 validation / 10 test (stratified)
# Features: Recency, Frequency, Monetary (classic RFM)

FEATURE_NAMES = ["Recency", "Frequency", "Monetary"]

@app.get("/churn")
def churn_prediction(merchant_id: int | None = None, churn_days: int = 90,
                     start_date: str | None = None,
                     end_date: str | None = None):
    """
    Predict customer churn probability using classic RFM features.
    Training features: Recency, Frequency, Monetary.

    Model selection (Kaggle-benchmarked, 80:10:10 split):
      ≥ 200 customers → XGBoost (early stopping on validation set)
      20-199           → Logistic Regression (80:10:10 split)
      5-19             → Logistic Regression (Leave-One-Out CV)
      < 5 or 1 class   → Not enough data (error returned)

    Returns: predictions, modelAccuracy, rocAuc, classificationReport,
             confusionMatrix, globalFeatureImportance, shapBaseValue,
             modelUsed, dataSize, modelThreshold
    """
    df = load_transactions(merchant_id, start_date, end_date)
    if df.empty:
        raise HTTPException(status_code=404, detail="No transactions found for the selected date range.")

    df = df.dropna(subset=["card_no"])

    try:
        max_date = df["txn_date"].max()
        cutoff = max_date - timedelta(days=churn_days)

        historical = df[df["txn_date"] <= cutoff]
        future = df[df["txn_date"] > cutoff]

        # ── Fallback: insufficient date spread ──
        if historical.empty:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"The selected date range is too narrow for churn prediction. "
                    f"A minimum span of {churn_days} days is needed to identify churned customers. "
                    f"Please widen the date range or reduce the churn window."
                ),
            )

        # ── Build RFM features on historical data ──
        rfm = _build_rfm(historical, cutoff)

        active_customers = set(future["card_no"].unique())
        rfm["churn"] = (~rfm["card_no"].isin(active_customers)).astype(int)

        feature_cols = ["recency", "frequency", "monetary"]
        X = rfm[feature_cols].copy()
        y = rfm["churn"]
        n = len(rfm)

        accuracy = None
        roc_auc = None
        global_importance = None
        shap_base_value = None
        used_model = None
        cls_report = None
        cm = None

        if y.nunique() < 2:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"All {n} customers in this period are {'all active' if int(y.iloc[0]) == 0 else 'all churned'} — "
                    f"churn prediction requires a mix of active and churned customers. "
                    f"Try widening the date range or increasing the churn window (currently {churn_days} days)."
                ),
            )

        if n < 5:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Not enough data for prediction — only {n} unique customer(s) found in the historical window. "
                    f"At least 5 customers are needed. "
                    f"Note: the model uses unique customers from transactions older than {churn_days} days, "
                    f"not total transaction count."
                ),
            )

        elif n < 20:
            # ── Tiny dataset (5-19): LR + Leave-One-Out CV ──
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            loo = LeaveOneOut()
            loo_preds = np.zeros(len(y))
            loo_probas = np.zeros(len(y))
            for train_idx, test_idx in loo.split(X_scaled):
                lr_cv = LogisticRegression(
                    class_weight="balanced", random_state=67, max_iter=1000,
                )
                lr_cv.fit(X_scaled[train_idx], y.iloc[train_idx])
                loo_preds[test_idx] = lr_cv.predict(X_scaled[test_idx])
                loo_probas[test_idx] = lr_cv.predict_proba(X_scaled[test_idx])[:, 1]

            accuracy = round(float(accuracy_score(y, loo_preds)), 4)
            if y.nunique() > 1:
                roc_auc = round(float(roc_auc_score(y, loo_probas)), 4)

            report_dict = classification_report(y, loo_preds, output_dict=True, zero_division=0)
            cls_report = {k: v for k, v in report_dict.items() if k not in ("accuracy",)}
            cm_raw = confusion_matrix(y, loo_preds, labels=[0, 1])
            cm = {"tn": int(cm_raw[0][0]), "fp": int(cm_raw[0][1]),
                  "fn": int(cm_raw[1][0]), "tp": int(cm_raw[1][1])}

            lr_model = LogisticRegression(
                class_weight="balanced", random_state=67, max_iter=1000,
            )
            lr_model.fit(X_scaled, y)
            rfm["churnProbability"] = lr_model.predict_proba(X_scaled)[:, 1]
            used_model = "Logistic Regression (LOO)"

            coefs = np.abs(lr_model.coef_[0])
            global_importance = {
                name: round(float(val), 4)
                for name, val in zip(FEATURE_NAMES, coefs)
            }
            for i, fname in enumerate(FEATURE_NAMES):
                rfm[f"shap_{fname}"] = X_scaled[:, i] * lr_model.coef_[0][i]
            shap_base_value = round(float(lr_model.intercept_[0]), 4)

        elif n < XGBOOST_THRESHOLD:
            # ── Medium dataset (20-199): LR + 80:10:10 split ──
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            can_stratify = y.nunique() > 1 and y.value_counts().min() >= 2
            X_train, X_temp, y_train, y_temp = train_test_split(
                X_scaled, y, test_size=0.2, random_state=67,
                stratify=y if can_stratify else None,
            )
            can_stratify_temp = y_temp.nunique() > 1 and y_temp.value_counts().min() >= 2 and len(y_temp) >= 4
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=67,
                stratify=y_temp if can_stratify_temp else None,
            )

            lr_model = LogisticRegression(
                class_weight="balanced", random_state=67, max_iter=1000,
            )
            lr_model.fit(X_train, y_train)

            y_pred = lr_model.predict(X_test)
            y_proba = lr_model.predict_proba(X_test)[:, 1]
            accuracy = round(float(accuracy_score(y_test, y_pred)), 4)
            if y_test.nunique() > 1:
                roc_auc = round(float(roc_auc_score(y_test, y_proba)), 4)

            report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            cls_report = {k: v for k, v in report_dict.items() if k not in ("accuracy",)}
            cm_raw = confusion_matrix(y_test, y_pred, labels=[0, 1])
            cm = {"tn": int(cm_raw[0][0]), "fp": int(cm_raw[0][1]),
                  "fn": int(cm_raw[1][0]), "tp": int(cm_raw[1][1])}

            rfm["churnProbability"] = lr_model.predict_proba(X_scaled)[:, 1]
            used_model = "Logistic Regression"

            coefs = np.abs(lr_model.coef_[0])
            global_importance = {
                name: round(float(val), 4)
                for name, val in zip(FEATURE_NAMES, coefs)
            }
            for i, fname in enumerate(FEATURE_NAMES):
                rfm[f"shap_{fname}"] = X_scaled[:, i] * lr_model.coef_[0][i]
            shap_base_value = round(float(lr_model.intercept_[0]), 4)

        else:
            # ── Large dataset (≥ 200): XGBoost with 80:10:10 + early stopping ──
            can_stratify = y.nunique() > 1 and y.value_counts().min() >= 2
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.2, random_state=67,
                stratify=y if can_stratify else None,
            )
            can_stratify_temp = y_temp.nunique() > 1 and y_temp.value_counts().min() >= 2 and len(y_temp) >= 4
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=67,
                stratify=y_temp if can_stratify_temp else None,
            )

            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_val_sc = scaler.transform(X_val)
            X_test_sc = scaler.transform(X_test)
            X_all_sc = scaler.transform(X)

            model = xgb.XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                eval_metric="logloss",
                random_state=67,
                early_stopping_rounds=10,
                verbosity=0,
            )
            model.fit(
                X_train_sc, y_train,
                eval_set=[(X_val_sc, y_val)],
                verbose=False,
            )

            y_pred = model.predict(X_test_sc)
            y_proba = model.predict_proba(X_test_sc)[:, 1]
            accuracy = round(float(accuracy_score(y_test, y_pred)), 4)
            if y_test.nunique() > 1:
                roc_auc = round(float(roc_auc_score(y_test, y_proba)), 4)

            report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            cls_report = {k: v for k, v in report_dict.items() if k not in ("accuracy",)}
            cm_raw = confusion_matrix(y_test, y_pred, labels=[0, 1])
            cm = {"tn": int(cm_raw[0][0]), "fp": int(cm_raw[0][1]),
                  "fn": int(cm_raw[1][0]), "tp": int(cm_raw[1][1])}

            rfm["churnProbability"] = model.predict_proba(X_all_sc)[:, 1]
            used_model = "XGBoost"

            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_all_sc)
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                global_importance = {
                    name: round(float(val), 4)
                    for name, val in zip(FEATURE_NAMES, mean_abs_shap)
                }
                for i, fname in enumerate(FEATURE_NAMES):
                    rfm[f"shap_{fname}"] = shap_values[:, i]
                shap_base_value = round(float(explainer.expected_value), 4)
            except Exception:
                pass

        # ── Build predictions list ──
        predictions = []
        for _, row in rfm.iterrows():
            pred = {
                "cardNo": row["card_no"],
                "recency": int(row["recency"]),
                "frequency": int(row["frequency"]),
                "monetary": round(float(row["monetary"]), 2),
                "churnProbability": round(float(row["churnProbability"]), 4),
                "isChurned": bool(row["churn"]),
            }
            shap_cols = [c for c in rfm.columns if c.startswith("shap_")]
            if shap_cols:
                pred["shapBreakdown"] = {
                    col.replace("shap_", ""): round(float(row[col]), 4)
                    for col in shap_cols
                }
            predictions.append(pred)

        predictions.sort(key=lambda p: p["churnProbability"], reverse=True)

        high_risk = sum(1 for p in predictions if p["churnProbability"] > 0.7)
        predicted_churners = sum(1 for p in predictions if p["churnProbability"] > 0.5)
        computed_churn_rate = round((predicted_churners / len(predictions)) * 100, 2) if predictions else 0

        return {
            "predictions": predictions,
            "modelAccuracy": accuracy,
            "rocAuc": roc_auc,
            "churnRate": computed_churn_rate,
            "highRiskCount": high_risk,
            "churnDays": churn_days,
            "totalCustomers": len(predictions),
            "globalFeatureImportance": global_importance,
            "shapBaseValue": shap_base_value,
            "classificationReport": cls_report,
            "confusionMatrix": cm,
            "modelUsed": used_model,
            "dataSize": n,
            "modelThreshold": XGBOOST_THRESHOLD,
        }

    except HTTPException:
        raise  # re-raise intentional HTTP errors (404, 422) as-is
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Churn prediction failed for the selected date range: {exc}. "
                f"Try widening the date range (current churn window: {churn_days} days)."
            ),
        ) from exc


# ─── Cash-Flow Forecasting (Prophet) ────────────────────────────────────────

@app.get("/forecast")
def cash_flow_forecast(merchant_id: int | None = None, horizon_days: int = 30,
                       start_date: str | None = None,
                       end_date: str | None = None):
    """
    Forecast daily revenue for the next `horizon_days` days using Prophet.

    Returns:
      - actual: list of {ds, y} — last 90 days of real data
      - forecast: list of {ds, yhat, yhat_lower, yhat_upper}
      - totalPredicted: sum of yhat over horizon
      - changePercent: % change vs same-length period before forecast
    """
    df = load_transactions(merchant_id, start_date, end_date)
    if df.empty:
        raise HTTPException(status_code=404, detail="No transactions found for the selected date range.")

    # Aggregate to daily
    df["date"] = df["txn_date"].dt.date
    daily = df.groupby("date")["amount"].sum().reset_index()
    daily.columns = ["ds", "y"]
    daily["ds"] = pd.to_datetime(daily["ds"])

    # Fill missing days
    date_range = pd.date_range(daily["ds"].min(), daily["ds"].max(), freq="D")
    daily = daily.set_index("ds").reindex(date_range, fill_value=0).reset_index()
    daily.columns = ["ds", "y"]

    if len(daily) < 14:
        # ── Fallback: flat projection based on daily average ──
        avg_daily = float(daily["y"].mean()) if len(daily) > 0 else 0
        last_date = daily["ds"].max()

        actual = [
            {"ds": row["ds"].date().isoformat(), "y": round(float(row["y"]), 2)}
            for _, row in daily.iterrows()
        ]

        forecast = []
        for i in range(1, horizon_days + 1):
            d = last_date + timedelta(days=i)
            forecast.append({
                "ds": d.date().isoformat(),
                "yhat": round(avg_daily, 2),
                "yhat_lower": round(avg_daily * 0.7, 2),
                "yhat_upper": round(avg_daily * 1.3, 2),
            })

        total_predicted = round(avg_daily * horizon_days, 2)
        return {
            "actual": actual,
            "forecast": forecast,
            "totalPredicted": total_predicted,
            "changePercent": 0.0,
            "horizonDays": horizon_days,
            "lastActualDate": last_date.date().isoformat(),
            "fallbackMode": True,
            "fallbackReason": f"Only {len(daily)} day(s) of data — using daily-average projection instead of Prophet",
        }

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.80,
    )
    model.fit(daily)

    future = model.make_future_dataframe(periods=horizon_days, freq="D")
    forecast_df = model.predict(future)

    # Clip negatives
    forecast_df["yhat"] = forecast_df["yhat"].clip(lower=0)
    forecast_df["yhat_lower"] = forecast_df["yhat_lower"].clip(lower=0)
    forecast_df["yhat_upper"] = forecast_df["yhat_upper"].clip(lower=0)

    # Actual (last 90 days for display)
    cutoff_actual = daily["ds"].max() - timedelta(days=89)
    actual_display = daily[daily["ds"] >= cutoff_actual].copy()
    actual = [
        {"ds": row["ds"].date().isoformat(), "y": round(float(row["y"]), 2)}
        for _, row in actual_display.iterrows()
    ]

    # Forecast rows (only the future part)
    future_rows = forecast_df[forecast_df["ds"] > daily["ds"].max()]
    forecast = [
        {
            "ds": row["ds"].date().isoformat(),
            "yhat": round(float(row["yhat"]), 2),
            "yhat_lower": round(float(row["yhat_lower"]), 2),
            "yhat_upper": round(float(row["yhat_upper"]), 2),
        }
        for _, row in future_rows.iterrows()
    ]

    total_predicted = round(float(future_rows["yhat"].sum()), 2)

    # Change vs previous same-length window
    prev_window = daily[
        daily["ds"] >= (daily["ds"].max() - timedelta(days=horizon_days * 2))
    ].tail(horizon_days)
    prev_total = float(prev_window["y"].sum())
    change_pct = (
        round((total_predicted - prev_total) / prev_total * 100, 2)
        if prev_total > 0 else None
    )

    return {
        "actual": actual,
        "forecast": forecast,
        "totalPredicted": total_predicted,
        "changePercent": change_pct,
        "horizonDays": horizon_days,
        "lastActualDate": daily["ds"].max().date().isoformat(),
    }
