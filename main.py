"""
MSP AI Sidecar — FastAPI entry point.

Delegates all model logic to dedicated modules:
  rfm_model.py      — K-Means RFM customer segmentation
    churn_model.py    — Pre-trained XGBoost churn prediction
  forecast_model.py — Prophet cash-flow forecasting

Start:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Environment variables (or .env file):
    DB_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/msp_db
"""

from __future__ import annotations

import warnings
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rfm_model      import run_rfm_segmentation
from churn_model    import run_churn_prediction
from forecast_model import run_cash_flow_forecast

warnings.filterwarnings("ignore")
load_dotenv()

# ─── App ────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MSP AI Sidecar",
    description="K-Means RFM Segmentation · XGBoost Churn · Prophet Forecasting",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8001", "http://localhost:4201"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ─── Routes ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/rfm")
def rfm_segmentation(merchant_id: int | None = None,
                     start_date:  str | None = None,
                     end_date:    str | None = None):
    """K-Means RFM customer segmentation grouped by card_no."""
    return run_rfm_segmentation(merchant_id, start_date, end_date)


@app.get("/churn")
def churn_prediction(merchant_id: int | None = None,
                     churn_days:  int         = 90,
                     start_date:  str | None = None,
                     end_date:    str | None = None):
    """Pre-trained XGBoost churn prediction (5-feature Kaggle model)."""
    return run_churn_prediction(merchant_id, churn_days, start_date, end_date)


@app.get("/forecast")
def cash_flow_forecast(merchant_id:  int | None = None,
                       horizon_days: int         = 30,
                       start_date:   str | None = None,
                       end_date:     str | None = None):
    """Prophet cash-flow forecast for the next `horizon_days` days."""
    return run_cash_flow_forecast(merchant_id, horizon_days, start_date, end_date)
