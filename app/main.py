from fastapi import FastAPI
import joblib
import pandas as pd
import time

from prometheus_client import Counter, Histogram, generate_latest

app = FastAPI()

model = joblib.load("models/model.pkl")

# -----------------------
# METRICS
# -----------------------
REQUEST_COUNT = Counter("api_requests_total", "Total API Requests")
LATENCY = Histogram("api_latency_seconds", "API Latency")
FRAUD_PRED = Counter("fraud_predictions_total", "Fraud Predictions")

# -----------------------
# PREDICT
# -----------------------
@app.post("/predict")
def predict(data: dict):
    start = time.time()

    df = pd.DataFrame([data])
    pred = model.predict(df)[0]

    REQUEST_COUNT.inc()
    if pred == 1:
        FRAUD_PRED.inc()

    LATENCY.observe(time.time() - start)

    return {"fraud": int(pred)}

# -----------------------
# METRICS ENDPOINT
# -----------------------
from fastapi import Response

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")