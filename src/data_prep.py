import pandas as pd

def load_data():
    trans = pd.read_csv("data/train_transaction.csv", nrows=10000)
    identity = pd.read_csv("data/train_identity.csv", nrows=10000)

    df = trans.merge(identity, on="TransactionID", how="left")
    return df


def preprocess(df):
    df = df.drop(columns=["TransactionID"], errors="ignore")

    # Separate target
    y = df["isFraud"]
    X = df.drop(columns=["isFraud"])

    # Fill missing
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    X[cat_cols] = X[cat_cols].fillna("missing")

    # Encode categoricals
    for col in cat_cols:
        X[col] = X[col].astype("category").cat.codes

    # Reduce size (IMPORTANT)
    X = X.iloc[:, :100]

    return X, y