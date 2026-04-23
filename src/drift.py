import os
import json
import joblib
import numpy as np
import pandas as pd
import mlflow
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("fraud-drift-simulation")

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def load_merged_data(nrows=12000):
    trans = pd.read_csv("data/train_transaction.csv", nrows=nrows)
    identity = pd.read_csv("data/train_identity.csv", nrows=nrows)
    df = trans.merge(identity, on="TransactionID", how="left")
    return df


def simulate_time_drift(df):
    """
    Simulate later-period drift:
    - transaction amount inflation
    - extra missingness
    - slight category shifts
    """
    drifted = df.copy()

    if "TransactionAmt" in drifted.columns:
        drifted["TransactionAmt"] = drifted["TransactionAmt"] * np.random.uniform(1.15, 1.45)

    # Make missingness worse in later period
    for col in ["dist1", "dist2", "P_emaildomain", "R_emaildomain", "DeviceInfo"]:
        if col in drifted.columns:
            mask = np.random.rand(len(drifted)) < 0.12
            drifted.loc[mask, col] = np.nan

    # Inject some new category patterns
    if "P_emaildomain" in drifted.columns:
        drifted.loc[drifted.sample(frac=0.05, random_state=42).index, "P_emaildomain"] = "newdomain.com"

    if "DeviceType" in drifted.columns:
        drifted.loc[drifted.sample(frac=0.03, random_state=42).index, "DeviceType"] = "tablet"

    return drifted


def compute_psi(expected, actual, buckets=10):
    """
    Population Stability Index for one feature.
    expected: training values
    actual: drifted/test values
    """
    expected = pd.Series(expected).replace([np.inf, -np.inf], np.nan).dropna()
    actual = pd.Series(actual).replace([np.inf, -np.inf], np.nan).dropna()

    if expected.empty or actual.empty:
        return np.nan

    try:
        breakpoints = np.unique(np.quantile(expected, np.linspace(0, 1, buckets + 1)))
        if len(breakpoints) < 3:
            return 0.0
        expected_counts = pd.cut(expected, bins=breakpoints, include_lowest=True).value_counts(normalize=True).sort_index()
        actual_counts = pd.cut(actual, bins=breakpoints, include_lowest=True).value_counts(normalize=True).sort_index()
    except Exception:
        return np.nan

    expected_counts, actual_counts = expected_counts.align(actual_counts, fill_value=0.0001)

    expected_counts = expected_counts.replace(0, 0.0001)
    actual_counts = actual_counts.replace(0, 0.0001)

    psi = np.sum((actual_counts - expected_counts) * np.log(actual_counts / expected_counts))
    return float(psi)


def build_preprocessor(X):
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )

    return preprocessor, num_cols, cat_cols


def save_bar_plot(df_scores, path):
    plt.figure(figsize=(12, 6))
    top = df_scores.sort_values("psi", ascending=False).head(15)
    plt.bar(top["feature"], top["psi"])
    plt.xticks(rotation=75, ha="right")
    plt.title("Top PSI Drift Features")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_confusion_matrix_plot(cm, path, title):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = [0, 1]
    plt.xticks(ticks, ["Not Fraud", "Fraud"])
    plt.yticks(ticks, ["Not Fraud", "Fraud"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def evaluate_model(model, X_test, y_test):
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

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "cm": cm,
    }


def main():
    mlflow.set_experiment("fraud-drift-simulation")

    df = load_merged_data(nrows=12000)
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    if "isFraud" not in df.columns:
        raise ValueError("Target column isFraud not found.")

    split_idx = int(len(df) * 0.7)
    early = df.iloc[:split_idx].copy()
    late = df.iloc[split_idx:].copy()

    # Keep only rows where target exists and is valid
    early = early.dropna(subset=["isFraud"]).copy()
    late = late.dropna(subset=["isFraud"]).copy()

    # Simulate stronger later-period drift
    late_drifted = simulate_time_drift(late)

    # Prepare features
    drop_cols = ["TransactionID", "TransactionDT", "isFraud"]
    X_early = early.drop(columns=[c for c in drop_cols if c in early.columns])
    y_early = early["isFraud"].astype(int)

    X_late = late.drop(columns=[c for c in drop_cols if c in late.columns])
    y_late = late["isFraud"].astype(int)

    X_late_drifted = late_drifted.drop(columns=[c for c in drop_cols if c in late_drifted.columns])
    y_late_drifted = late_drifted["isFraud"].astype(int)

    preprocessor, num_cols, cat_cols = build_preprocessor(X_early)

    X_early_ready = preprocessor.fit_transform(X_early)
    X_late_ready = preprocessor.transform(X_late)
    X_late_drifted_ready = preprocessor.transform(X_late_drifted)

    # Cost-sensitive learning to reflect fraud importance
    class_counts = y_early.value_counts().to_dict()
    neg = class_counts.get(0, 1)
    pos = class_counts.get(1, 1)
    scale_pos_weight = neg / max(pos, 1)

    model = XGBClassifier(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )

    with mlflow.start_run(run_name="time_based_drift"):
        model.fit(X_early_ready, y_early)

        before_drift = evaluate_model(model, X_late_ready, y_late)
        after_drift = evaluate_model(model, X_late_drifted_ready, y_late_drifted)

        # PSI on selected interpretable features
        candidate_features = [
            "TransactionAmt", "card1", "card2", "card3", "card5",
            "addr1", "addr2", "dist1", "dist2", "C1", "C2", "D1"
        ]

        psi_rows = []
        for feature in candidate_features:
            if feature in early.columns and feature in late_drifted.columns:
                psi_val = compute_psi(early[feature], late_drifted[feature])
                psi_rows.append({"feature": feature, "psi": psi_val})

        psi_df = pd.DataFrame(psi_rows).sort_values("psi", ascending=False)
        psi_csv = os.path.join(ARTIFACT_DIR, "psi_scores.csv")
        psi_df.to_csv(psi_csv, index=False)

        psi_plot = os.path.join(ARTIFACT_DIR, "psi_drift_top_features.png")
        save_bar_plot(psi_df, psi_plot)

        cm_before_path = os.path.join(ARTIFACT_DIR, "confusion_matrix_before_drift.png")
        cm_after_path = os.path.join(ARTIFACT_DIR, "confusion_matrix_after_drift.png")
        save_confusion_matrix_plot(before_drift["cm"], cm_before_path, "Confusion Matrix - Before Drift")
        save_confusion_matrix_plot(after_drift["cm"], cm_after_path, "Confusion Matrix - After Drift")

        drift_report = {
            "train_rows": int(len(early)),
            "test_rows": int(len(late)),
            "drifted_test_rows": int(len(late_drifted)),
            "before_drift": {
                "precision": before_drift["precision"],
                "recall": before_drift["recall"],
                "f1": before_drift["f1"],
                "auc": before_drift["auc"],
                "confusion_matrix": before_drift["cm"].tolist(),
            },
            "after_drift": {
                "precision": after_drift["precision"],
                "recall": after_drift["recall"],
                "f1": after_drift["f1"],
                "auc": after_drift["auc"],
                "confusion_matrix": after_drift["cm"].tolist(),
            },
            "recall_drop": float(before_drift["recall"] - after_drift["recall"]),
            "retrain_trigger": bool((before_drift["recall"] - after_drift["recall"]) > 0.10 or after_drift["recall"] < 0.80),
        }

        drift_report_path = os.path.join(ARTIFACT_DIR, "drift_report.json")
        with open(drift_report_path, "w", encoding="utf-8") as f:
            json.dump(drift_report, f, indent=2)

        joblib.dump(model, os.path.join(ARTIFACT_DIR, "drift_model.pkl"))

        # Log to MLflow
        mlflow.log_param("train_rows", len(early))
        mlflow.log_param("test_rows", len(late))
        mlflow.log_param("scale_pos_weight", scale_pos_weight)

        mlflow.log_metric("precision_before_drift", before_drift["precision"])
        mlflow.log_metric("recall_before_drift", before_drift["recall"])
        mlflow.log_metric("f1_before_drift", before_drift["f1"])
        mlflow.log_metric("auc_before_drift", before_drift["auc"])

        mlflow.log_metric("precision_after_drift", after_drift["precision"])
        mlflow.log_metric("recall_after_drift", after_drift["recall"])
        mlflow.log_metric("f1_after_drift", after_drift["f1"])
        mlflow.log_metric("auc_after_drift", after_drift["auc"])

        mlflow.log_metric("recall_drop", drift_report["recall_drop"])
        mlflow.log_metric("retrain_trigger", int(drift_report["retrain_trigger"]))

        mlflow.log_artifact(drift_report_path)
        mlflow.log_artifact(psi_csv)
        mlflow.log_artifact(psi_plot)
        mlflow.log_artifact(cm_before_path)
        mlflow.log_artifact(cm_after_path)

    print("Drift simulation complete.")
    print("Before drift recall:", before_drift["recall"])
    print("After drift recall:", after_drift["recall"])
    print("Recall drop:", drift_report["recall_drop"])
    print("Retrain trigger:", drift_report["retrain_trigger"])
    print("\nTop PSI features:")
    print(psi_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()