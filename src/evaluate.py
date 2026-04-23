import os
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.data_prep import load_data, preprocess

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("fraud-detection")

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def save_confusion_matrix(cm, path):
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Not Fraud", "Fraud"],
        yticklabels=["Not Fraud", "Fraud"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def evaluate():
    df = load_data()
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model_path = "models/model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError("models/model.pkl not found. Run training first.")

    model = joblib.load(model_path)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict(X_test)

    y_pred = (y_prob >= 0.5).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    metrics_dict = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "confusion_matrix": cm.tolist(),
    }

    metrics_json_path = os.path.join(ARTIFACT_DIR, "metrics.json")
    report_path = os.path.join(ARTIFACT_DIR, "classification_report.txt")
    cm_path = os.path.join(ARTIFACT_DIR, "confusion_matrix.png")

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    save_confusion_matrix(cm, cm_path)

    with mlflow.start_run(run_name="final_evaluation"):
        mlflow.log_metrics(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "auc": auc,
            }
        )
        mlflow.log_artifact(metrics_json_path)
        mlflow.log_artifact(report_path)
        mlflow.log_artifact(cm_path)

    print("Evaluation complete")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("AUC:", auc)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)


if __name__ == "__main__":
    evaluate()