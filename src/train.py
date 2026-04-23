import os
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from src.data_prep import load_data, preprocess

# ── MLflow setup (declared once) ────────────────────────────────────────────
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("fraud-detection")

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def evaluate(y_true, y_pred):
    """Return scalar metrics + confusion matrix."""
    return {
        "precision": precision_score(y_true, y_pred),
        "recall":    recall_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred),
        "auc":       roc_auc_score(y_true, y_pred),
        "cm":        confusion_matrix(y_true, y_pred),
    }


def log_all_artifacts(model, metrics, X, run_name):
    """
    Log metrics, confusion-matrix image, feature-importance CSV,
    and the model itself — all inside the *already active* MLflow run.
    """
    # 1. Scalar metrics
    mlflow.log_metrics({
        "precision": metrics["precision"],
        "recall":    metrics["recall"],
        "f1":        metrics["f1"],
        "auc":       metrics["auc"],
    })

    # 2. Confusion-matrix heatmap
    cm_path = os.path.join(ARTIFACT_DIR, f"{run_name}_confusion_matrix.png")
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        metrics["cm"], annot=True, fmt="d",
        cmap="Blues", cbar=False,
        xticklabels=["Not Fraud", "Fraud"],
        yticklabels=["Not Fraud", "Fraud"],
    )
    plt.title(f"Confusion Matrix — {run_name}")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)

    # 3. Feature importance (tree-based models expose .feature_importances_)
    if hasattr(model, "feature_importances_"):
        feat_df = pd.DataFrame({
            "feature":    X.columns,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

        feat_path = os.path.join(ARTIFACT_DIR, f"{run_name}_feature_importance.csv")
        feat_df.to_csv(feat_path, index=False)
        mlflow.log_artifact(feat_path)

    # 4. Model (MLflow native format — queryable from the UI)
    mlflow.sklearn.log_model(model, artifact_path="model")


# ── Training ─────────────────────────────────────────────────────────────────

def train():
    df = load_data()
    X, y = preprocess(df)

    # Stratified split so class ratios are preserved in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    results = {}

    # ── XGBoost (cost-sensitive) ─────────────────────────────────────────────
    with mlflow.start_run(run_name="xgboost"):
        model = XGBClassifier(scale_pos_weight=10, random_state=42)
        model.fit(X_train, y_train)

        preds   = model.predict(X_test)
        metrics = evaluate(y_test, preds)
        log_all_artifacts(model, metrics, X_train, run_name="xgboost")

        results["xgb"] = (model, metrics)

    # ── LightGBM + SMOTE ─────────────────────────────────────────────────────
    with mlflow.start_run(run_name="lightgbm_smote"):
        X_sm, y_sm = SMOTE(random_state=42).fit_resample(X_train, y_train)

        model = LGBMClassifier(random_state=42)
        model.fit(X_sm, y_sm)

        preds   = model.predict(X_test)
        metrics = evaluate(y_test, preds)
        log_all_artifacts(model, metrics, X_train, run_name="lightgbm_smote")

        results["lgb"] = (model, metrics)

    # ── Random Forest (hybrid) ───────────────────────────────────────────────
    with mlflow.start_run(run_name="hybrid_rf"):
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        preds   = model.predict(X_test)
        metrics = evaluate(y_test, preds)
        log_all_artifacts(model, metrics, X_train, run_name="hybrid_rf")

        results["rf"] = (model, metrics)

    # ── Select & save best model (by recall) ────────────────────────────────
    best_key   = max(results, key=lambda k: results[k][1]["recall"])
    best_model = results[best_key][0]

    model_path = "models/model.pkl"
    joblib.dump(best_model, model_path)
    print(f"Best model: {best_key}  →  saved to {model_path}")

    # Log the winning model under a dedicated "best_model" run so it's
    # easy to find in the MLflow UI
    with mlflow.start_run(run_name="best_model"):
        best_metrics = results[best_key][1]
        mlflow.log_param("best_model_type", best_key)
        log_all_artifacts(best_model, best_metrics, X_train, run_name="best_model")
        mlflow.log_artifact(model_path)   # also attach the .pkl


if __name__ == "__main__":
    train()