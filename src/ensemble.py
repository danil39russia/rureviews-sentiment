"""Weighted ensemble of saved model logits.

Weights are fitted on the validation set only (minimising log-loss of the averaged
log-probabilities), the test set is used once for the final report.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_softmax

from metrics import save_run


def load(results_dir: str, name: str):
    val = np.load(os.path.join(results_dir, f"{name}.val.npy"))
    test = np.load(os.path.join(results_dir, f"{name}.test.npy"))
    return log_softmax(val, -1), log_softmax(test, -1)


def combine(logps, w):
    return sum(wi * lp for wi, lp in zip(w, logps))


def fit_weights(val_logps, labels):
    k = len(val_logps)

    def loss(raw):
        w = np.exp(raw) / np.exp(raw).sum()
        lp = log_softmax(combine(val_logps, w), -1)
        return -lp[np.arange(len(labels)), labels].mean()

    res = minimize(loss, np.zeros(k), method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-4})
    return np.exp(res.x) / np.exp(res.x).sum()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="random", choices=["random", "dedup"])
    parser.add_argument("--data-dir", default="data/splits")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--run-name", default="ensemble")
    parser.add_argument("--uniform", action="store_true", help="equal weights instead of fitted ones")
    args = parser.parse_args()

    d = os.path.join(args.data_dir, args.split)
    val_labels = pd.read_csv(os.path.join(d, "val.csv"))["label"].to_numpy()
    test_labels = pd.read_csv(os.path.join(d, "test.csv"))["label"].to_numpy()

    res_dir = os.path.join(args.results_dir, args.split)
    pairs = [load(res_dir, m) for m in args.models]
    val_logps = [p[0] for p in pairs]
    test_logps = [p[1] for p in pairs]

    w = np.ones(len(pairs)) / len(pairs) if args.uniform else fit_weights(val_logps, val_labels)
    weights = {m: round(float(x), 4) for m, x in zip(args.models, w)}
    print("weights:", json.dumps(weights))
    save_run(res_dir, args.run_name, combine(val_logps, w), combine(test_logps, w), val_labels, test_labels,
             {"members": args.models, "weights": weights})


if __name__ == "__main__":
    main()
