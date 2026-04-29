"""
forecast_model.py — Prophet cash-flow forecasting.

Always trains on the FULL historical dataset for accuracy.
start_date / end_date only control which portion of the actual
revenue series is shown in the chart (viewing filter).
"""
from __future__ import annotations

from datetime import timedelta

import pandas as pd
from fastapi import HTTPException
from prophet import Prophet

from db import load_transactions_all


def run_cash_flow_forecast(merchant_id: int | None,
                           horizon_days: int,
                           start_date: str | None,
                           end_date: str | None) -> dict:
    """
    Forecast daily revenue for the next `horizon_days` days using Prophet,
    trained on ALL historical data.
    start_date / end_date only filter the `actual` series returned for display.

    Returns:
      actual, forecast, totalPredicted, changePercent, horizonDays, lastActualDate
    """
    df = load_transactions_all(merchant_id)
    if df.empty:
        raise HTTPException(status_code=404,
                            detail="No transactions found in the database.")

    # Aggregate to daily revenue (full history)
    df["date"] = df["txn_date"].dt.date
    daily = df.groupby("date")["amount"].sum().reset_index()
    daily.columns = ["ds", "y"]
    daily["ds"] = pd.to_datetime(daily["ds"])

    # Fill missing days with 0
    date_range = pd.date_range(daily["ds"].min(), daily["ds"].max(), freq="D")
    daily = daily.set_index("ds").reindex(date_range, fill_value=0).reset_index()
    daily.columns = ["ds", "y"]

    # ── Fallback: too few days for Prophet ──────────────────────────────────
    if len(daily) < 14:
        avg_daily = float(daily["y"].mean()) if len(daily) > 0 else 0
        last_date = daily["ds"].max()

        actual_all = [
            {"ds": row["ds"].date().isoformat(), "y": round(float(row["y"]), 2)}
            for _, row in daily.iterrows()
        ]
        actual = _filter_actual(actual_all, start_date, end_date)

        forecast = [
            {
                "ds":         (last_date + timedelta(days=i)).date().isoformat(),
                "yhat":       round(avg_daily, 2),
                "yhat_lower": round(avg_daily * 0.7, 2),
                "yhat_upper": round(avg_daily * 1.3, 2),
            }
            for i in range(1, horizon_days + 1)
        ]
        return {
            "actual":          actual,
            "forecast":        forecast,
            "totalPredicted":  round(avg_daily * horizon_days, 2),
            "changePercent":   0.0,
            "horizonDays":     horizon_days,
            "lastActualDate":  last_date.date().isoformat(),
            "fallbackMode":    True,
            "fallbackReason":  f"Only {len(daily)} day(s) of data — using daily-average projection instead of Prophet",
        }

    # ── Prophet forecast (trained on all data) ───────────────────────────────
    model = Prophet(
        yearly_seasonality =True,
        weekly_seasonality =True,
        daily_seasonality  =False,
        interval_width     =0.80,
    )
    model.fit(daily)

    future      = model.make_future_dataframe(periods=horizon_days, freq="D")
    forecast_df = model.predict(future)

    # Clip negative predictions
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        forecast_df[col] = forecast_df[col].clip(lower=0)

    # ── Build actual series: last 90 days by default, then apply view filter ─
    cutoff_actual  = daily["ds"].max() - timedelta(days=89)
    actual_display = daily[daily["ds"] >= cutoff_actual]
    actual_all = [
        {"ds": row["ds"].date().isoformat(), "y": round(float(row["y"]), 2)}
        for _, row in actual_display.iterrows()
    ]
    actual = _filter_actual(actual_all, start_date, end_date)

    # Future-only forecast rows
    future_rows = forecast_df[forecast_df["ds"] > daily["ds"].max()]
    forecast = [
        {
            "ds":         row["ds"].date().isoformat(),
            "yhat":       round(float(row["yhat"]), 2),
            "yhat_lower": round(float(row["yhat_lower"]), 2),
            "yhat_upper": round(float(row["yhat_upper"]), 2),
        }
        for _, row in future_rows.iterrows()
    ]

    total_predicted = round(float(future_rows["yhat"].sum()), 2)

    # % change vs same-length window before forecast
    prev_window = daily[
        daily["ds"] >= (daily["ds"].max() - timedelta(days=horizon_days * 2))
    ].tail(horizon_days)
    prev_total  = float(prev_window["y"].sum())
    change_pct  = (
        round((total_predicted - prev_total) / prev_total * 100, 2)
        if prev_total > 0 else None
    )

    db_min = df["txn_date"].min().date().isoformat()
    db_max = df["txn_date"].max().date().isoformat()

    return {
        "actual":          actual,
        "forecast":        forecast,
        "totalPredicted":  total_predicted,
        "changePercent":   change_pct,
        "horizonDays":     horizon_days,
        "lastActualDate":  daily["ds"].max().date().isoformat(),
        "dateRange":       {"from": db_min, "to": db_max},
    }


def _filter_actual(actual: list[dict], start_date: str | None, end_date: str | None) -> list[dict]:
    """Filter the actual revenue list by date range (display-only)."""
    if start_date:
        actual = [a for a in actual if a["ds"] >= start_date]
    if end_date:
        actual = [a for a in actual if a["ds"] <= end_date]
    return actual
