import os
import json
import joblib
import numpy as np
import pandas as pd
import mlflow
import shap
import matplotlib.pyplot as plt

from src.data_prep import load_data

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("fraud-explainability")

ARTIFACT_DIR = "artifacts"
MODEL_PATH = "models/model.pkl"
SCHEMA_PATH = "models/preprocess_schema.json"

os.makedirs(ARTIFACT_DIR, exist_ok=True)


def load_schema():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_base_estimator_and_transformed_data(model, X_sample):
    """
    Handles:
    - Pipeline(preprocess -> model)
    - Pipeline(preprocess -> selector -> model)
    - plain estimator
    """
    feature_names = None
    X_transformed = X_sample.copy()

    if hasattr(model, "named_steps"):
        steps = list(model.named_steps.keys())

        # preprocess step
        if "preprocess" in model.named_steps:
            preprocessor = model.named_steps["preprocess"]
            X_transformed = preprocessor.transform(X_sample)

            try:
                feature_names = preprocessor.get_feature_names_out()
            except Exception:
                feature_names = np.array([f"f{i}" for i in range(X_transformed.shape[1])])

        # selector step
        if "selector" in model.named_steps:
            selector = model.named_steps["selector"]
            X_transformed = selector.transform(X_transformed)

            if feature_names is not None:
                try:
                    selected_mask = selector.get_support()
                    feature_names = np.array(feature_names)[selected_mask]
                except Exception:
                    feature_names = np.array([f"f{i}" for i in range(X_transformed.shape[1])])

        # final estimator
        estimator = model.named_steps[steps[-1]]
        return estimator, X_transformed, feature_names

    # plain model
    if hasattr(X_sample, "columns"):
        feature_names = np.array(X_sample.columns)
    else:
        feature_names = np.array([f"f{i}" for i in range(X_sample.shape[1])])

    return model, X_sample, feature_names


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train first.")

    if not os.path.exists(SCHEMA_PATH):
        raise FileNotFoundError(f"Schema not found at {SCHEMA_PATH}. Train first.")

    model = joblib.load(MODEL_PATH)
    df = load_data()

    # small sample for explainability
    if len(df) > 1000:
        df = df.sample(n=1000, random_state=42).reset_index(drop=True)

    # align with training prep
    df = df.drop(columns=["TransactionID"], errors="ignore")
    y = df["isFraud"].astype(int)
    X = df.drop(columns=["isFraud"])

    # keep same feature truncation as training
    X = X.iloc[:, :100].copy()

    # sample for SHAP speed
    X_sample = X.sample(n=min(200, len(X)), random_state=42)

    estimator, X_transformed, feature_names = get_base_estimator_and_transformed_data(model, X_sample)

    # Tree models
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_transformed)

    # output paths
    summary_png = os.path.join(ARTIFACT_DIR, "shap_summary.png")
    bar_png = os.path.join(ARTIFACT_DIR, "shap_bar.png")
    top_csv = os.path.join(ARTIFACT_DIR, "shap_top_features.csv")

    # SHAP summary
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        show=False,
        max_display=15,
    )
    plt.tight_layout()
    plt.savefig(summary_png, dpi=150, bbox_inches="tight")
    plt.close()

    # SHAP bar plot
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=15,
    )
    plt.tight_layout()
    plt.savefig(bar_png, dpi=150, bbox_inches="tight")
    plt.close()

    # Top feature importance table
    shap_abs_mean = np.abs(shap_values).mean(axis=0)
    top_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": shap_abs_mean,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    top_df.to_csv(top_csv, index=False)

    # log to MLflow
    with mlflow.start_run(run_name="shap_explainability"):
        mlflow.log_artifact(summary_png)
        mlflow.log_artifact(bar_png)
        mlflow.log_artifact(top_csv)

    print("SHAP explainability complete.")
    print("Saved:", summary_png)
    print("Saved:", bar_png)
    print("Saved:", top_csv)
    print("\nTop 10 features:")
    print(top_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()