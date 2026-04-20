# MSP AI Sidecar

Python FastAPI microservice that provides three AI-powered analytics endpoints consumed by the Spring Boot backend.

## Models

| Endpoint | Model | Purpose |
|---|---|---|
| `GET /rfm` | K-Means (sklearn) | Customer segmentation by Recency, Frequency, Monetary |
| `GET /churn` | XGBoost | Predict which customers are likely to stop transacting |
| `GET /forecast` | Facebook Prophet | Daily cash-flow forecast for the next N days |

## Setup

```bash
cd python-sidecar
python -m venv venv
.\venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Create a `.env` file:
```
DB_URL=postgresql+psycopg2://postgres:YOUR_PASSWORD@localhost:5432/msp
```

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs: http://localhost:8000/docs

## Query Parameters

All endpoints accept an optional `merchant_id` (integer) to scope results to a single merchant.

| Parameter | Default | Description |
|---|---|---|
| `merchant_id` | *(all merchants)* | Scope to a specific merchant |
| `churn_days` (churn only) | `90` | Days of inactivity that defines churn |
| `horizon_days` (forecast only) | `30` | How many days ahead to forecast |
