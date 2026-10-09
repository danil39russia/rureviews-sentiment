"""QLoRA fine-tuning of a decoder LLM (Qwen2.5) with a classification head.

The last-token hidden state is fed to a linear head (AutoModelForSequenceClassification),
the backbone is loaded in 4 bit and adapted with LoRA on all linear projections.
"""
import argparse
import os
import random
import shutil

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from metrics import save_run

PROMPT = "Определи тональность отзыва покупателя о товаре.\nОтзыв: {text}\nТональность:"


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
    parser.add_argument("--model-name", default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    parser.add_argument("--run-name", default="qwen2.5_7b_lora")
    parser.add_argument("--split", default="random", choices=["random", "dedup"])
    parser.add_argument("--data-dir", default="data/splits")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--max-length", default=256, type=int)
    parser.add_argument("--epochs", default=1, type=float)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--grad-accum", default=2, type=int)
    parser.add_argument("--lora-r", default=16, type=int)
    parser.add_argument("--lora-alpha", default=32, type=int)
    parser.add_argument("--lora-dropout", default=0.05, type=float)
    parser.add_argument("--eval-steps", default=500, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--keep-adapter", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    d = os.path.join(args.data_dir, args.split)
    frames = {p: pd.read_csv(os.path.join(d, f"{p}.csv")) for p in ("train", "val", "test")}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()

    def encode(df: pd.DataFrame) -> Dataset:
        texts = [PROMPT.format(text=t) for t in df["text"].fillna("").astype(str)]
        enc = tokenizer(texts, truncation=True, max_length=args.max_length)
        enc["labels"] = df["label"].tolist()
        return Dataset.from_dict(dict(enc))

    data = {p: encode(df) for p, df in frames.items()}

    ckpt_dir = os.path.join(args.ckpt_dir, args.split, args.run_name)
    training_args = TrainingArguments(
        output_dir=ckpt_dir,
        overwrite_output_dir=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=1,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=0.0,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=args.seed,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        dataloader_num_workers=0,
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

    if args.keep_adapter:
        trainer.save_model(os.path.join(ckpt_dir, "best"))
    else:
        shutil.rmtree(ckpt_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
