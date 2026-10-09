"""Fine-tune a transformer encoder for 3-class sentiment classification on RuReviews."""
import argparse
import os
import random
import shutil

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from metrics import save_run


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"f1_macro": f1_score(labels, preds, average="macro"), "accuracy": accuracy_score(labels, preds)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--split", default="random", choices=["random", "dedup"])
    parser.add_argument("--data-dir", default="data/splits")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--text-prefix", default="", help="e.g. 'query: ' for E5 models")
    parser.add_argument("--max-length", default=256, type=int)
    parser.add_argument("--epochs", default=3, type=float)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--grad-accum", default=1, type=int)
    parser.add_argument("--weight-decay", default=0.01, type=float)
    parser.add_argument("--warmup-ratio", default=0.1, type=float)
    parser.add_argument("--label-smoothing", default=0.0, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--optim", default="adamw_torch", help="adamw_bnb_8bit saves memory for large models")
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--keep-checkpoint", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    d = os.path.join(args.data_dir, args.split)
    frames = {p: pd.read_csv(os.path.join(d, f"{p}.csv")) for p in ("train", "val", "test")}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, revision=args.revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, revision=args.revision, num_labels=3
    )

    def encode(df: pd.DataFrame) -> Dataset:
        texts = [args.text_prefix + t for t in df["text"].fillna("").astype(str)]
        enc = tokenizer(texts, truncation=True, max_length=args.max_length)
        enc["labels"] = df["label"].tolist()
        return Dataset.from_dict(dict(enc))

    data = {p: encode(df) for p, df in frames.items()}

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    ckpt_dir = os.path.join(args.ckpt_dir, args.split, args.run_name)
    training_args = TrainingArguments(
        output_dir=ckpt_dir,
        overwrite_output_dir=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        label_smoothing_factor=args.label_smoothing,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=100,
        report_to="none",
        seed=args.seed,
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        dataloader_num_workers=0,
        optim=args.optim,
        gradient_checkpointing=args.grad_checkpointing,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["val"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    val_logits = trainer.predict(data["val"]).predictions
    test_logits = trainer.predict(data["test"]).predictions
    extra = {"hparams": {k: v for k, v in vars(args).items()}}
    save_run(os.path.join(args.out_dir, args.split), args.run_name, val_logits, test_logits,
             frames["val"]["label"], frames["test"]["label"], extra)

    if args.keep_checkpoint:
        trainer.save_model(os.path.join(ckpt_dir, "best"))
    else:
        shutil.rmtree(ckpt_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
