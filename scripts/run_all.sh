#!/usr/bin/env bash
# Full reproduction of all experiments. Usage: bash scripts/run_all.sh [random|dedup]
set -euo pipefail
SPLIT=${1:-random}
export PYTHONIOENCODING=utf-8
mkdir -p logs

[ -f data/raw/rureviews.csv ] || bash scripts/download_data.sh
python src/prepare_data.py

python src/baseline_tfidf.py --split "$SPLIT"

# Fine-tuned encoders: previous art reproduction (RuBERT, XLM-R-large) and competitors.
bash scripts/queue_encoders.sh "$SPLIT"

# LLMs: QLoRA classifier, zero-shot and few-shot prompting.
# Optional, not included in the reported results (time budget): bash scripts/queue_llm.sh "$SPLIT"

# Final ensemble, weights are fitted on the validation set.
python src/ensemble.py --split "$SPLIT" --run-name ensemble \
    --models ruroberta_large xlmr_large rumodernbert_base rubert_base tfidf_logreg

python src/make_tables.py --split "$SPLIT"
