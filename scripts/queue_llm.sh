#!/usr/bin/env bash
# LLM experiments for one split: QLoRA classifier, then zero-/few-shot prompting.
SPLIT=${1:-random}
export PYTHONIOENCODING=utf-8
mkdir -p logs
python src/train_llm_lora.py --split "$SPLIT" --model-name unsloth/Qwen2.5-7B-Instruct-bnb-4bit \
    --run-name qwen2.5_7b_lora > "logs/${SPLIT}_qwen2.5_7b_lora.log" 2>&1; echo "done lora exit=$?"
python src/zero_shot_llm.py --split "$SPLIT" --run-name qwen2.5_7b_zeroshot > "logs/${SPLIT}_qwen2.5_7b_zeroshot.log" 2>&1; echo "done zs exit=$?"
python src/zero_shot_llm.py --split "$SPLIT" --run-name qwen2.5_7b_fewshot --few-shot > "logs/${SPLIT}_qwen2.5_7b_fewshot.log" 2>&1; echo "done fs exit=$?"
