"""Zero-shot / few-shot sentiment classification with an instruction-tuned LLM.

The class score is the next-token logit of each label word (all three are single tokens for Qwen),
so no generation or answer parsing is needed.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from metrics import save_run

SYSTEM = ("Ты — эксперт по анализу тональности. Тебе дают отзыв покупателя о товаре из интернет-магазина "
          "одежды. Определи тональность отзыва и ответь одним английским словом: negative, neutral или positive.")
LABEL_WORDS = ["negative", "neutral", "positive"]
FEW_SHOT = [
    ("Ткань ужасная, швы кривые, размер не соответствует. Деньги на ветер.", 0),
    ("Нормально, за свою цену пойдёт. Цвет немного отличается от фото.", 1),
    ("Отличное платье, сидит идеально, доставка быстрая. Спасибо продавцу!", 2),
]


def build_messages(text: str, few_shot: bool):
    messages = [{"role": "system", "content": SYSTEM}]
    if few_shot:
        for ex, lab in FEW_SHOT:
            messages.append({"role": "user", "content": f"Отзыв: {ex}"})
            messages.append({"role": "assistant", "content": LABEL_WORDS[lab]})
    messages.append({"role": "user", "content": f"Отзыв: {text}"})
    return messages


@torch.no_grad()
def score(model, tokenizer, texts, label_ids, few_shot, batch_size):
    out = []
    for i in tqdm(range(0, len(texts), batch_size)):
        prompts = [tokenizer.apply_chat_template(build_messages(t, few_shot), tokenize=False,
                                                 add_generation_prompt=True) for t in texts[i:i + batch_size]]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        logits = model(**enc).logits[:, -1, :]
        out.append(torch.log_softmax(logits.float(), -1)[:, label_ids].cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    parser.add_argument("--run-name", default="qwen2.5_7b_zeroshot")
    parser.add_argument("--few-shot", action="store_true")
    parser.add_argument("--split", default="random", choices=["random", "dedup"])
    parser.add_argument("--data-dir", default="data/splits")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--batch-size", default=32, type=int)
    args = parser.parse_args()

    d = os.path.join(args.data_dir, args.split)
    val, test = (pd.read_csv(os.path.join(d, f"{p}.csv")) for p in ("val", "test"))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map={"": 0})
    model.eval()

    label_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in LABEL_WORDS]
    assert len(set(label_ids)) == 3, "label words must map to different tokens"

    val_scores = score(model, tokenizer, val["text"].astype(str).tolist(), label_ids, args.few_shot, args.batch_size)
    test_scores = score(model, tokenizer, test["text"].astype(str).tolist(), label_ids, args.few_shot, args.batch_size)
    save_run(os.path.join(args.out_dir, args.split), args.run_name, val_scores, test_scores,
             val["label"], test["label"], {"few_shot": args.few_shot})


if __name__ == "__main__":
    main()
