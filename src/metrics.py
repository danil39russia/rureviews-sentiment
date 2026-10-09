import json
import os

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def classification_report(labels, preds) -> dict:
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    per_class = f1_score(labels, preds, average=None, labels=[0, 1, 2])
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision_macro": float(precision_score(labels, preds, average="macro")),
        "recall_macro": float(recall_score(labels, preds, average="macro")),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "f1_negative": float(per_class[0]),
        "f1_neutral": float(per_class[1]),
        "f1_positive": float(per_class[2]),
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1, 2]).tolist(),
    }


def save_run(out_dir: str, name: str, val_logits, test_logits, val_labels, test_labels, extra=None) -> dict:
    # Every model stores raw logits so that ensembles can be built afterwards.
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{name}.val.npy"), np.asarray(val_logits, dtype=np.float32))
    np.save(os.path.join(out_dir, f"{name}.test.npy"), np.asarray(test_logits, dtype=np.float32))
    report = {
        "model": name,
        "val": classification_report(val_labels, np.argmax(val_logits, -1)),
        "test": classification_report(test_labels, np.argmax(test_logits, -1)),
    }
    if extra:
        report.update(extra)
    with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[{name}] val F1={report['val']['f1_macro']:.4f}  test F1={report['test']['f1_macro']:.4f}  "
          f"test acc={report['test']['accuracy']:.4f}")
    return report
