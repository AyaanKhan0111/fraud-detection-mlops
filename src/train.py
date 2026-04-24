import os
import json
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
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
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import ConfusionMatrixDisplay
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.data_prep import load_data

# -----------------------------
# MLflow setup
# -----------------------------
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("fraud-detection")

ARTIFACT_DIR = "artifacts"
MODEL_DIR = "models"
SCHEMA_PATH = os.path.join(MODEL_DIR, "preprocess_schema.json")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# -----------------------------
# Helpers
# -----------------------------
def save_schema(X: pd.DataFrame, numeric_cols, categorical_cols) -> None:
    schema = {
        "feature_columns": X.columns.tolist(),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "numeric_defaults": {},
    }

    for col in numeric_cols:
        series = pd.to_numeric(X[col], errors="coerce")
        med = series.median()
        schema["numeric_defaults"][col] = 0.0 if pd.isna(med) else float(med)

    with open(SCHEMA_PATH, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)


def build_preprocessor(numeric_cols, categorical_cols):
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

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def evaluate_model(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_prob),
        "cm": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def log_run_artifacts(run_name, model, metrics, X_columns):
    mlflow.log_metrics(
        {
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
            "auc": float(metrics["auc"]),
        }
    )

    cm = metrics["cm"]
    cm_path = os.path.join(ARTIFACT_DIR, f"{run_name}_confusion_matrix.png")

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
    plt.title(f"Confusion Matrix — {run_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()
    mlflow.log_artifact(cm_path)

    # log feature importance when available and aligned
    estimator = model
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        estimator = model.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        importances = getattr(estimator, "feature_importances_", None)
        if importances is not None and len(importances) == len(X_columns):
            feat_df = pd.DataFrame(
                {
                    "feature": X_columns,
                    "importance": importances,
                }
            ).sort_values("importance", ascending=False)

            feat_path = os.path.join(ARTIFACT_DIR, f"{run_name}_feature_importance.csv")
            feat_df.to_csv(feat_path, index=False)
            mlflow.log_artifact(feat_path)

    mlflow.sklearn.log_model(model, artifact_path="model")


def prepare_data():
    df = load_data().copy()

    if "isFraud" not in df.columns:
        raise ValueError("Target column isFraud not found.")

    # Keep runtime manageable
    if len(df) > 12000:
        df = df.sample(n=12000, random_state=42).reset_index(drop=True)

    # Drop obvious ID column if present
    df = df.drop(columns=["TransactionID"], errors="ignore")

    y = df["isFraud"].astype(int)
    X = df.drop(columns=["isFraud"])

    # Reduce dimensionality a bit for faster training
    X = X.iloc[:, :100].copy()

    numeric_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    save_schema(X, numeric_cols, categorical_cols)
    return X, y, numeric_cols, categorical_cols


# -----------------------------
# Training
# -----------------------------
def train():
    X, y, numeric_cols, categorical_cols = prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    results = {}

    # 1) XGBoost cost-sensitive
    with mlflow.start_run(run_name="xgboost"):
        class_counts = y_train.value_counts().to_dict()
        neg = class_counts.get(0, 1)
        pos = class_counts.get(1, 1)
        scale_pos_weight = neg / max(pos, 1)

        model = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", XGBClassifier(
                    scale_pos_weight=scale_pos_weight,
                    n_estimators=120,
                    max_depth=5,
                    learning_rate=0.08,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    eval_metric="logloss",
                    random_state=42,
                )),
            ]
        )

        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_param("model_type", "xgboost_cost_sensitive")
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        log_run_artifacts("xgboost", model, metrics, X.columns)
        results["xgb"] = (model, metrics)

    # 2) LightGBM + SMOTE
    with mlflow.start_run(run_name="lightgbm_smote"):
        model = ImbPipeline(
            steps=[
                ("preprocess", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("model", LGBMClassifier(
                    n_estimators=150,
                    learning_rate=0.05,
                    num_leaves=31,
                    random_state=42,
                )),
            ]
        )

        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_param("model_type", "lightgbm_smote")
        log_run_artifacts("lightgbm_smote", model, metrics, X.columns)
        results["lgb"] = (model, metrics)

    # 3) Hybrid RF feature selection + LightGBM
    with mlflow.start_run(run_name="hybrid_rf"):
        model = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("selector", SelectFromModel(
                    estimator=RandomForestClassifier(
                        n_estimators=100,
                        random_state=42,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                    threshold="median",
                )),
                ("model", LGBMClassifier(
                    n_estimators=150,
                    learning_rate=0.05,
                    num_leaves=31,
                    random_state=42,
                )),
            ]
        )

        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_param("model_type", "hybrid_rf_lgbm")
        log_run_artifacts("hybrid_rf", model, metrics, X.columns)
        results["rf"] = (model, metrics)

    # Select best model by recall
    best_key = max(results, key=lambda k: results[k][1]["recall"])
    best_model = results[best_key][0]
    best_metrics = results[best_key][1]

    joblib.dump(best_model, MODEL_PATH)
    print(f"Best model: {best_key}  →  saved to {MODEL_PATH}")

    with mlflow.start_run(run_name="best_model"):
        mlflow.log_param("best_model_type", best_key)
        mlflow.log_param("feature_count", X.shape[1])
        mlflow.log_artifact(MODEL_PATH)
        mlflow.log_artifact(SCHEMA_PATH)
        log_run_artifacts("best_model", best_model, best_metrics, X.columns)


if __name__ == "__main__":
    train()