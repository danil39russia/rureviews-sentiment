#!/usr/bin/env bash
# Runs all encoder experiments for one split sequentially, one log per model.
SPLIT=${1:-random}
export PYTHONIOENCODING=utf-8
mkdir -p logs
run() { name=$1; shift; python src/train_encoder.py --split "$SPLIT" --run-name "$name" "$@" > "logs/${SPLIT}_${name}.log" 2>&1; echo "done $SPLIT $name exit=$?"; }
# Large models: 8-bit AdamW and batch 8 x 4 (effective 32) to fit into 16 GB VRAM.
LARGE="--batch-size 8 --grad-accum 4 --optim adamw_bnb_8bit"
run rubert_base --model-name DeepPavlov/rubert-base-cased --lr 3e-5
run rumodernbert_base --model-name deepvk/RuModernBERT-base --revision patched-tokenizer --lr 5e-5
run xlmr_large --model-name FacebookAI/xlm-roberta-large --lr 1e-5 $LARGE
run ruroberta_large --model-name ai-forever/ruRoBERTa-large --lr 2e-5 --epochs 1 $LARGE
# Not included in the reported results (time budget):
# run me5_large --model-name intfloat/multilingual-e5-large --text-prefix "query: " --lr 1.5e-5 $LARGE
