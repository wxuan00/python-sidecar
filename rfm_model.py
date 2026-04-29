"""
rfm_model.py — K-Means RFM customer segmentation.

Model is always trained on the FULL historical dataset so segments are
stable and meaningful.  The optional start_date / end_date parameters
only filter which customer rows are returned for *display* in the UI.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from db import load_transactions_all


def run_rfm_segmentation(merchant_id: int | None,
                         start_date: str | None,
                         end_date: str | None) -> dict:
    """
    Compute RFM scores and K-Means customer segments using ALL historical data.
    start_date / end_date are used only to filter the segments list returned
    to the frontend (viewing filter), not for model training.

    Returns:
      segments, clusterSummary, silhouetteScore, totalCustomers, snapshotDate, dateRange
    """
    # ── Always compute on full dataset ───────────────────────────────────────
    df = load_transactions_all(merchant_id)
    if df.empty:
        raise HTTPException(status_code=404,
                            detail="No transactions found in the database.")

    df = df.dropna(subset=["card_no"])
    if len(df) < 10:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Not enough data for segmentation — only {len(df)} transaction(s) found."
            ),
        )

    snapshot = df["txn_date"].max() + timedelta(days=1)

    rfm = df.groupby("card_no").agg(
        recency  =("txn_date",       lambda x: (snapshot - x.max()).days),
        frequency=("transaction_id", "nunique"),
        monetary =("amount",         "sum"),
    ).reset_index()

    # Log-transform + StandardScale
    rfm_log    = np.log1p(rfm[["recency", "frequency", "monetary"]])
    scaler     = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)

    n_clusters     = min(4, len(rfm))
    kmeans         = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

    sil_score = None
    if len(rfm) > n_clusters:
        sil_score = round(float(silhouette_score(rfm_scaled, rfm["cluster"])), 3)

    cluster_stats = rfm.groupby("cluster").agg(
        avg_recency  =("recency",   "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary =("monetary",  "mean"),
        count        =("card_no",   "count"),
    ).reset_index()

    cluster_stats["score"] = (
        -cluster_stats["avg_recency"]   * 0.3
        + cluster_stats["avg_frequency"] * 0.35
        + cluster_stats["avg_monetary"]  * 0.35
    )
    cluster_stats_sorted = cluster_stats.sort_values("score", ascending=False).reset_index(drop=True)
    rank_labels = ["Champions", "Loyal Customers", "At Risk", "Lost Customers"]
    label_map = {
        int(row["cluster"]): rank_labels[min(i, len(rank_labels) - 1)]
        for i, row in cluster_stats_sorted.iterrows()
    }
    rfm["label"] = rfm["cluster"].map(label_map)

    # ── Attach last-seen date per customer for display filtering ─────────────
    last_seen = df.groupby("card_no")["txn_date"].max().reset_index()
    last_seen.columns = ["card_no", "lastSeen"]
    rfm = rfm.merge(last_seen, on="card_no", how="left")

    # ── Apply date filter for display only ───────────────────────────────────
    rfm_display = rfm.copy()
    if start_date:
        rfm_display = rfm_display[rfm_display["lastSeen"] >= pd.Timestamp(start_date)]
    if end_date:
        rfm_display = rfm_display[rfm_display["lastSeen"] <= pd.Timestamp(end_date + " 23:59:59")]

    segments = rfm_display.rename(columns={"card_no": "cardNo"}).to_dict(orient="records")
    for s in segments:
        s["recency"]   = int(s["recency"])
        s["frequency"] = int(s["frequency"])
        s["monetary"]  = round(float(s["monetary"]), 2)
        s["cluster"]   = int(s["cluster"])
        s["lastSeen"]  = s["lastSeen"].date().isoformat() if pd.notna(s["lastSeen"]) else None

    cluster_summary = [
        {
            "cluster":      int(row["cluster"]),
            "label":        label_map[int(row["cluster"])],
            "count":        int(row["count"]),
            "avgRecency":   round(float(row["avg_recency"]), 1),
            "avgFrequency": round(float(row["avg_frequency"]), 1),
            "avgMonetary":  round(float(row["avg_monetary"]), 2),
        }
        for _, row in cluster_stats.iterrows()
    ]

    db_min = df["txn_date"].min().date().isoformat()
    db_max = df["txn_date"].max().date().isoformat()

    return {
        "segments":        segments,
        "clusterSummary":  cluster_summary,
        "silhouetteScore": sil_score,
        "totalCustomers":  len(rfm),
        "displayCustomers": len(rfm_display),
        "snapshotDate":    df["txn_date"].max().date().isoformat(),
        "dateRange":       {"from": db_min, "to": db_max},
    }
