"""Make the train/val/test split for the CSC311 project.

Run: python split.py
Splits respondents 70/15/15 by id and saves the result to split.csv,
so every teammate uses the identical split.
"""
import numpy as np
import pandas as pd

DATA_FILE = "csc311_project_training_202609.csv"
SEED = 311


def make_split():
    """Split by respondent id (70/15/15) and save it to split.csv."""
    df = pd.read_csv(DATA_FILE)
    ids = df["id"].unique()
    print("Unique respondents:", len(ids))
    print("Rows per respondent:", df.groupby("id").size().value_counts().to_dict())

    rng = np.random.default_rng(SEED)
    ids = rng.permutation(ids)
    n_train = int(0.70 * len(ids))
    n_val = int(0.15 * len(ids))

    split = pd.DataFrame({"id": ids})
    split["split"] = "test"
    split.loc[:n_train - 1, "split"] = "train"
    split.loc[n_train:n_train + n_val - 1, "split"] = "val"
    split.sort_values("id").to_csv("split.csv", index=False)
    print("Saved split.csv")

    df = df.merge(split, on="id")
    print("\nRows per split:\n", df["split"].value_counts().to_string())
    print("\nRespondents per split:\n", split["split"].value_counts().to_string())
    print("\nClass counts in train:\n",
          df.loc[df["split"] == "train", "label"].value_counts().to_string())


if __name__ == "__main__":
    make_split()
