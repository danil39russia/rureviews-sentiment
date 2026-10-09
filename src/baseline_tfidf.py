"""TF-IDF (word + char n-grams) + logistic regression baseline."""
import argparse
import os

import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from metrics import save_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="random", choices=["random", "dedup"])
    parser.add_argument("--data-dir", default="data/splits")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--C", default=4.0, type=float)
    args = parser.parse_args()

    d = os.path.join(args.data_dir, args.split)
    train, val, test = (pd.read_csv(os.path.join(d, f"{p}.csv")) for p in ("train", "val", "test"))
    for df in (train, val, test):
        df["text"] = df["text"].fillna("").astype(str)

    word = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=300_000, sublinear_tf=True)
    char = TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                           max_features=300_000, sublinear_tf=True)
    x_train = hstack([word.fit_transform(train["text"]), char.fit_transform(train["text"])]).tocsr()
    x_val = hstack([word.transform(val["text"]), char.transform(val["text"])]).tocsr()
    x_test = hstack([word.transform(test["text"]), char.transform(test["text"])]).tocsr()

    clf = LogisticRegression(C=args.C, max_iter=2000)
    clf.fit(x_train, train["label"])
    save_run(os.path.join(args.out_dir, args.split), "tfidf_logreg",
             clf.decision_function(x_val), clf.decision_function(x_test), val["label"], test["label"])


if __name__ == "__main__":
    main()
