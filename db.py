"""
db.py — Database connection and shared transaction loader.
"""
from __future__ import annotations

import os
import pandas as pd
from sqlalchemy import create_engine, text

DB_URL = os.getenv("DB_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/msp_db")


def get_engine():
    return create_engine(DB_URL)


def load_transactions(merchant_id: int | None = None,
                      start_date: str | None = None,
                      end_date: str | None = None) -> pd.DataFrame:
    """Load approved/refunded transactions from the MSP database."""
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
