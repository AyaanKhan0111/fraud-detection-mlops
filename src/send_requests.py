import pandas as pd
import requests
import time

from src.data_prep import load_data, preprocess

# Load real dataset
df = load_data()
X, y = preprocess(df)

# Convert to dictionary records
records = X.to_dict(orient="records")

url = "http://127.0.0.1:8000/predict"

print("Sending real data to API...")

for i, row in enumerate(records[:500]):  # limit for speed
    try:
        response = requests.post(url, json=row)
        print(i, response.json())
    except Exception as e:
        print("Error:", e)

    time.sleep(0.05)  # simulate real-time traffic