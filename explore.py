"""Simple data exploration for the CSC311 project.

Run: python split.py first (makes split.csv), then python explore.py
Prints counts/distributions to the terminal and saves a few plots to figs/.
"""
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_FILE = "csc311_project_training_202609.csv"

# Short names for the long question columns (same order as the CSV)
SHORT_NAMES = ["id", "ease", "strategy", "injury", "individual", "where",
               "seasons", "music", "food", "colour", "song", "feeling",
               "price", "label"]
LIKERT = ["strategy", "injury", "individual"]
LIKERT_ORDER = ["Strongly disagree", "Disagree", "Neutral", "Agree",
                "Strongly agree"]
MULTI = ["where", "seasons", "music"]
TEXT = ["food", "colour", "song", "feeling"]


def load_data():
    df = pd.read_csv(DATA_FILE)
    df.columns = SHORT_NAMES
    # Treat blank text answers as missing
    for col in TEXT + MULTI + LIKERT:
        df[col] = df[col].str.strip().replace("", np.nan)
    return df


def load_train(df):
    """Keep only the train rows, using the split saved by split.py."""
    split = pd.read_csv("split.csv")
    df = df.merge(split, on="id")
    return df[df["split"] == "train"].copy()


def parse_price(value):
    """Turn a free-text price into a number (ranges -> midpoint)."""
    if pd.isna(value):
        return np.nan
    text = str(value).lower().replace(",", "")
    numbers = re.findall(r"\d+\.?\d*", text)
    if numbers:
        return np.mean([float(n) for n in numbers[:2]])
    if any(w in text for w in ["free", "none", "nothing", "zero", "not"]):
        return 0.0
    return np.nan


def missing_values(train):
    print("\n=== Missing values (train) ===")
    missing = pd.DataFrame({
        "count": train.isna().sum(),
        "percent": (train.isna().mean() * 100).round(1),
    })
    print(missing.to_string())


def distributions(train):
    print("\n=== Label counts ===")
    print(train["label"].value_counts().to_string())

    print("\n=== Ease (0-10) ===")
    print(train["ease"].describe().round(2).to_string())
    print("\nMean ease by sport:")
    print(train.groupby("label")["ease"].mean().round(2).to_string())

    for col in LIKERT:
        print(f"\n=== {col} ===")
        print(train[col].value_counts().reindex(LIKERT_ORDER).to_string())

    for col in MULTI:
        print(f"\n=== {col} (each option, % of rows by sport) ===")
        options = train[col].str.get_dummies(sep=",")
        options.columns = options.columns.str.strip()
        pct = (options.groupby(train["label"]).mean() * 100).round(1)
        print(pct.T.to_string())

    print("\n=== colour ===")
    colour = train["colour"].str.lower()
    print("Distinct raw values:", train["colour"].nunique(),
          "| after lowercase:", colour.nunique())
    print(colour.value_counts().head(10).to_string())

    print("\n=== price ===")
    price = train["price"].apply(parse_price)
    print("Could not parse:", price.isna().sum(), "rows")
    print(price.describe().round(1).to_string())
    print("\nMedian price by sport:")
    print(price.groupby(train["label"]).median().to_string())
    return price


def plots(train, price):
    os.makedirs("figs", exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    axes[0, 0].hist(train["ease"].dropna(), bins=11, edgecolor="black")
    axes[0, 0].set_title("Ease (0-10)")
    axes[0, 0].set_xlabel("Rating")
    axes[0, 0].set_ylabel("Count")

    for ax, col in zip([axes[0, 1], axes[0, 2], axes[1, 0]], LIKERT):
        counts = train[col].value_counts().reindex(LIKERT_ORDER)
        ax.bar(range(5), counts.values)
        ax.set_xticks(range(5))
        ax.set_xticklabels(["SD", "D", "N", "A", "SA"])
        ax.set_title(col)
        ax.set_ylabel("Count")

    axes[1, 1].hist(price.dropna(), bins=30, edgecolor="black")
    axes[1, 1].set_title("Price (raw)")
    axes[1, 1].set_xlabel("$")

    axes[1, 2].hist(np.log1p(price.dropna()), bins=30, edgecolor="black")
    axes[1, 2].set_title("Price (log1p)")
    axes[1, 2].set_xlabel("log(1 + $)")

    plt.tight_layout()
    plt.savefig("figs/distributions.png", dpi=150)
    plt.close()

    # Mean ease by sport
    train.groupby("label")["ease"].mean().plot(kind="bar")
    plt.title("Mean ease by sport")
    plt.ylabel("Mean ease")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("figs/ease_by_sport.png", dpi=150)
    plt.close()
    print("\nSaved plots to figs/")


def build_features(train, price_clean):
    """Encode the train rows into one numeric matrix (only to count features)."""
    parts = [train[["ease"]]]

    # Likert -> 1 to 5
    likert = pd.DataFrame({f"{c}_n": train[c].map(
        {a: i + 1 for i, a in enumerate(LIKERT_ORDER)}) for c in LIKERT})
    parts.append(likert)

    # Multi-select -> one column per option
    for col in MULTI:
        parts.append(train[col].str.get_dummies(sep=",").add_prefix(col + "_"))

    # Colour -> top 7 colours plus "other"
    colour = train["colour"].str.lower().str.strip()
    top7 = colour.value_counts().index[:7]
    parts.append(pd.get_dummies(colour.where(colour.isin(top7), "other"),
                                prefix="colour"))

    parts.append(train["song"].isna().astype(int).rename("song_missing"))

    # Price -> clipped at the train 99th percentile, then log1p
    clip = price_clean.quantile(0.99)
    parts.append(np.log1p(price_clean.clip(upper=clip)).rename("price"))

    # Free text -> TF-IDF, fit on train only
    for col in ["food", "song", "feeling"]:
        tfidf = TfidfVectorizer(min_df=2)
        words = tfidf.fit_transform(train[col].fillna(""))
        print(f"TF-IDF vocab size for {col}: {len(tfidf.vocabulary_)}")
        parts.append(pd.DataFrame(words.toarray(), index=train.index)
                     .add_prefix(col + "_"))

    return pd.concat(parts, axis=1), likert


def report_numbers(train, price_clean):
    """Key numbers for the report."""
    print("\n=== Key numbers for the report ===")
    X_train, likert = build_features(train, price_clean)
    train = pd.concat([train, likert], axis=1)

    print("\nSong missing % by sport:")
    print(train.groupby("label")["song"].apply(lambda s: s.isna().mean() * 100).round(1))
    print("\nPrice clip value (train 99th percentile):", round(price_clean.quantile(0.99), 1))
    print("\nFinal feature count:", X_train.shape[1])
    print("\nCorrelation of ease and Likert features:")
    print(train[["ease", "strategy_n", "injury_n", "individual_n"]].corr().round(2))


def main():
    df = load_data()
    train = load_train(df)  # everything below uses the train split only
    missing_values(train)
    price = distributions(train)
    plots(train, price)
    report_numbers(train, price)


if __name__ == "__main__":
    main()
