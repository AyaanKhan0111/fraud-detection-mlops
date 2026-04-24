from fastapi import FastAPI, Response
import joblib
import pandas as pd
import time
import json
import os

from prometheus_client import Counter, Histogram, generate_latest

app = FastAPI()

MODEL_PATH = "models/model.pkl"
SCHEMA_PATH = "models/preprocess_schema.json"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run training first.")

if not os.path.exists(SCHEMA_PATH):
    raise FileNotFoundError(f"Schema not found at {SCHEMA_PATH}. Run training first.")

model = joblib.load(MODEL_PATH)

with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    schema = json.load(f)

FEATURE_COLUMNS = schema["feature_columns"]
NUMERIC_COLUMNS = set(schema["numeric_columns"])
CATEGORICAL_COLUMNS = set(schema["categorical_columns"])
NUMERIC_DEFAULTS = schema["numeric_defaults"]

REQUEST_COUNT = Counter("api_requests_total", "Total API Requests")
LATENCY = Histogram("api_latency_seconds", "API Latency")
FRAUD_PRED = Counter("fraud_predictions_total", "Fraud Predictions")


def build_input_frame(payload: dict) -> pd.DataFrame:
    row = {}

    for col in FEATURE_COLUMNS:
        if col in payload:
            row[col] = payload[col]
        else:
            if col in NUMERIC_COLUMNS:
                row[col] = NUMERIC_DEFAULTS.get(col, 0.0)
            else:
                row[col] = "missing"

    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(NUMERIC_DEFAULTS.get(col, 0.0))

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("missing").astype(str)

    return df


@app.post("/predict")
def predict(data: dict):
    start = time.time()

    X_input = build_input_frame(data)

    pred = int(model.predict(X_input)[0])
    prob = None
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X_input)[0, 1])

    REQUEST_COUNT.inc()
    if pred == 1:
        FRAUD_PRED.inc()

    LATENCY.observe(time.time() - start)

    response = {"fraud": pred}
    if prob is not None:
        response["probability"] = prob
    return response


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")