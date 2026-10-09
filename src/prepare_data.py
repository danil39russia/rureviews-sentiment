"""Build train/val/test splits for RuReviews.

Two split protocols are produced:
  * random  - stratified 70/15/15 split of all 90k rows, the protocol used by
              Smetanin & Komarov (2021);
  * dedup   - the same proportions, but rows with identical normalized text are
              kept inside one split, so no test text is seen during training.
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = ["negative", "neutral", "positive"]
# The original file contains a typo in the neutral label.
RAW2LABEL = {"negative": 0, "neautral": 1, "neutral": 1, "positive": 2}


def normalize(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"\W+", " ", text)
    return text.strip()


def split_random(df: pd.DataFrame, seed: int):
    train, rest = train_test_split(df, test_size=0.3, random_state=seed, stratify=df["label"])
    val, test = train_test_split(rest, test_size=0.5, random_state=seed, stratify=rest["label"])
    return train, val, test


def split_dedup(df: pd.DataFrame, seed: int):
    # Split unique normalized texts (stratified by majority label), then map rows back.
    groups = df.groupby("norm")["label"].agg(lambda s: s.value_counts().idxmax()).reset_index()
    g_train, g_rest = train_test_split(groups, test_size=0.3, random_state=seed, stratify=groups["label"])
    g_val, g_test = train_test_split(g_rest, test_size=0.5, random_state=seed, stratify=g_rest["label"])
    out = []
    for g in (g_train, g_val, g_test):
        out.append(df[df["norm"].isin(set(g["norm"]))])
    return out


def describe(df: pd.DataFrame) -> dict:
    lengths = df["text"].str.len()
    words = df["text"].str.split().str.len()
    return {
        "rows": int(len(df)),
        "label_counts": {LABELS[k]: int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "chars_mean": float(lengths.mean()),
        "chars_median": float(lengths.median()),
        "words_mean": float(words.mean()),
        "words_total": int(words.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-path", default="data/raw/rureviews.csv")
    parser.add_argument("--out-dir", default="data/splits")
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    df = pd.read_csv(args.raw_path, sep="\t").rename(columns={"review": "text"})
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["sentiment"].map(RAW2LABEL).astype(int)
    df["norm"] = df["text"].map(normalize)
    df = df[["text", "label", "norm"]].reset_index(drop=True)
    df["id"] = np.arange(len(df))

    stats = {
        "full": describe(df),
        "exact_duplicate_rows": int(df["text"].duplicated().sum()),
        "normalized_duplicate_rows": int(df["norm"].duplicated().sum()),
        "texts_with_conflicting_labels": int(df.groupby("text")["label"].nunique().gt(1).sum()),
        "norm_texts_with_conflicting_labels": int(df.groupby("norm")["label"].nunique().gt(1).sum()),
    }

    for name, fn in (("random", split_random), ("dedup", split_dedup)):
        train, val, test = fn(df, args.seed)
        out_dir = os.path.join(args.out_dir, name)
        os.makedirs(out_dir, exist_ok=True)
        for part, part_df in (("train", train), ("val", val), ("test", test)):
            part_df[["id", "text", "label"]].to_csv(os.path.join(out_dir, f"{part}.csv"), index=False)
        train_norm = set(train["norm"])
        stats[name] = {
            "train": describe(train),
            "val": describe(val),
            "test": describe(test),
            "test_rows_seen_in_train": int(test["norm"].isin(train_norm).sum()),
        }

    with open(os.path.join(args.out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
