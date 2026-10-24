"""Data exploration for the CSC311 project (train split only).

Run: python split.py first (makes split.csv), then python explore.py
Prints the numbers used in the report and saves the plots to figs/.
"""
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Text answers that mean "no answer" (these are the ones in the train data)
PLACEHOLDERS = {"", "none", "no", "idk", "nil", "n/a", "-", "no idea"}

# Price answers with no number that mean "would not pay" (-> 0)
ZERO_WORDS = r"free|zero|none|nothing|not|unwilling|no willing|don.?t|wouldn.?t"


def load_train():
    df = pd.read_csv(DATA_FILE)
    df.columns = SHORT_NAMES
    for col in MULTI + LIKERT:
        df[col] = df[col].str.strip().replace("", np.nan)
    for col in TEXT:
        df[col] = df[col].str.strip()
        df.loc[df[col].str.lower().isin(PLACEHOLDERS), col] = np.nan

    # Keep only the train rows, using the split saved by split.py
    split = pd.read_csv("split.csv")
    df = df.merge(split, on="id")
    return df[df["split"] == "train"].copy()


def parse_price(value):
    """Turn a free-text price into a number (ranges -> midpoint)."""
    if pd.isna(value):
        return np.nan
    text = str(value).lower().replace(",", "")
    text = re.sub(r"(?<=\d) (?=\d{3}\b)", "", text)  # "10 000" -> "10000"
    numbers = re.findall(r"\d+\.?\d*", text)
    if numbers:
        return np.mean([float(n) for n in numbers[:2]])
    if re.search(ZERO_WORDS, text):
        return 0.0
    return np.nan


def missing_values(train):
    print("\nMissing values:")
    missing = pd.DataFrame({
        "count": train.isna().sum(),
        "percent": (train.isna().mean() * 100).round(1),
    })
    print(missing.drop(["id", "label", "split"]).to_string())

    print("\nSong missing % by sport:")
    print(train["song"].isna().groupby(train["label"]).mean().mul(100).round(1)
          .to_string())


def numeric_and_likert(train):
    print(f"\nEase: mean {train['ease'].mean():.2f}, median {train['ease'].median()}")

    for col in LIKERT:
        print(f"\n{col}:")
        print(train[col].value_counts().reindex(LIKERT_ORDER).to_string())
        agree = train[col].isin(["Agree", "Strongly agree"]).sum()
        print(f"Agree or strongly agree: {agree / train[col].notna().sum() * 100:.0f}%")

    print("\nCorrelation of ease and Likert features:")
    likert = train[LIKERT].apply(lambda s: s.map(
        {a: i + 1 for i, a in enumerate(LIKERT_ORDER)}))
    print(pd.concat([train["ease"], likert], axis=1).corr().round(2).to_string())

    print("\nSeasons, % of rows selecting each option by sport:")
    seasons = train["seasons"].str.get_dummies(sep=",")
    print((seasons.groupby(train["label"]).mean() * 100).round(1).T.to_string())


def colour(train):
    lower = train["colour"].str.lower()
    clean = lower.str.replace(r"[^\w\s]", "", regex=True).str.strip().replace("", np.nan)
    print(f"\nColour: {train['colour'].nunique()} distinct raw, "
          f"{lower.nunique()} after lowercasing, "
          f"{clean.nunique()} after removing punctuation")
    top7 = clean.value_counts().head(7)
    print(top7.to_string())
    print(f"Top 7 cover {top7.sum() / clean.notna().sum() * 100:.1f}% of answers")


def price(train):
    price = train["price"].apply(parse_price)
    print(f"\nPrice: {train['price'].isna().sum()} blank, "
          f"{price.isna().sum()} with no usable number (incl. blanks)")
    print(f"Valid: {price.notna().sum()}, median {price.median()}, "
          f"mean {price.mean():.0f}, 75th percentile {price.quantile(0.75)}, "
          f"max {price.max()}")
    print(f"Exactly $0: {(price == 0).sum() / price.notna().sum() * 100:.1f}%")
    print(f"Clip value (99th percentile): {price.quantile(0.99):.0f}")
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

    clip = price.quantile(0.99)
    axes[1, 2].hist(np.log1p(price.dropna().clip(upper=clip)), bins=30,
                    edgecolor="black")
    axes[1, 2].set_title("Price (clipped at 99th percentile, log1p)")
    axes[1, 2].set_xlabel("log(1 + $)")

    plt.tight_layout()
    plt.savefig("figs/distributions.png", dpi=150)
    plt.close()

    train.groupby("label")["ease"].mean().plot(kind="bar")
    plt.title("Mean ease by sport")
    plt.ylabel("Mean ease")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("figs/ease_by_sport.png", dpi=150)
    plt.close()
    print("\nSaved plots to figs/")


def main():
    train = load_train()
    missing_values(train)
    numeric_and_likert(train)
    colour(train)
    plots(train, price(train))


if __name__ == "__main__":
    main()
