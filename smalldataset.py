import pandas as pd

trans = pd.read_csv(
    "data/train_transaction.csv",
    nrows=10000,
    low_memory=False
)

identity = pd.read_csv(
    "data/train_identity.csv",
    nrows=10000,
    low_memory=False
)

# Take only 10k rows
trans_small = trans.head(10000)
identity_small = identity.head(10000)

# Save back (overwrite or new folder)
trans_small.to_csv("data/train_transaction.csv", index=False)
identity_small.to_csv("data/train_identity.csv", index=False)