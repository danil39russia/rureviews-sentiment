# Анализ тональности отзывов на RuReviews

Финальный проект курса NLP (весна 2026). Решается задача трёхклассовой классификации тональности
(negative / neutral / positive) русскоязычных отзывов о товарах на открытом датасете
[RuReviews](https://github.com/sismetanin/rureviews) (Smetanin & Komarov, 2019).

Отчёт: [`report/main.pdf`](report/main.pdf).

## Основные результаты

Test macro-F1, протокол `random` (стратифицированный сплит 70/15/15, как в Smetanin & Komarov, 2021):

| Модель | Test macro-F1, % |
|---|---|
| CNN + Word2Vec (Smetanin & Komarov, 2019), опубл. | 75.45 |
| RuBERT (Smetanin & Komarov, 2021), опубл. | 77.44 |
| XLM-RoBERTa-Large (Smetanin, 2022), опубл., предыдущий SotA | 78.81 |
| TF-IDF + LogReg | 75.07 |
| RuBERT-base (воспроизведение) | 78.28 |
| RuModernBERT-base | 78.57 |
| XLM-RoBERTa-Large (воспроизведение) | 79.44 |
| ruRoBERTa-large | 79.12 |
| Ансамбль, равные веса (абляция) | 79.89 |
| **Ансамбль, веса по валидации (наш)** | **80.14** |

Полная таблица с precision, recall, accuracy и F1 по классам: `results/random/summary.md`.

## Что найдено в данных

- 2 679 точных дубликатов и 5 068 дубликатов после нормализации;
- 388 текстов встречаются с разными метками (например, «Товар не пришёл» — negative и neutral);
- при случайном сплите 830 тестовых отзывов (6.1%) встречаются в train, поэтому добавлен протокол
  `dedup` без пересечений между частями.

## Структура

```
src/
  prepare_data.py     # загрузка, нормализация, сплиты random/dedup, статистика
  baseline_tfidf.py   # TF-IDF (слова + символы) + логистическая регрессия
  train_encoder.py    # дообучение энкодеров (RuBERT, XLM-R, ruRoBERTa, RuModernBERT, mE5)
  train_llm_lora.py   # QLoRA-классификатор на Qwen2.5 (в отчётных результатах не используется)
  zero_shot_llm.py    # zero-/few-shot классификация LLM (в отчётных результатах не используется)
  ensemble.py         # взвешенный ансамбль, веса подбираются на валидации
  make_tables.py      # сводные таблицы (markdown и LaTeX)
  metrics.py          # метрики и сохранение логитов
scripts/
  download_data.sh    # загрузка RuReviews
  queue_encoders.sh   # все энкодеры для заданного сплита
  queue_llm.sh        # LLM-эксперименты
  run_all.sh          # полное воспроизведение
data/splits/stats.json  # статистика датасета и сплитов
results/<split>/      # метрики (*.json) и логиты (*.npy) каждой модели
report/               # исходники отчёта LaTeX
```

## Воспроизведение

```bash
pip install -r requirements.txt
bash scripts/run_all.sh random
```

Отдельные шаги:

```bash
bash scripts/download_data.sh
python src/prepare_data.py
python src/baseline_tfidf.py --split random
python src/train_encoder.py --split random --model-name ai-forever/ruRoBERTa-large --run-name ruroberta_large \
    --lr 2e-5 --epochs 1 --batch-size 8 --grad-accum 4 --optim adamw_bnb_8bit
python src/ensemble.py --split random --run-name ensemble \
    --models ruroberta_large xlmr_large rumodernbert_base rubert_base tfidf_logreg
python src/make_tables.py --split random
```

Все эксперименты проводились на одной RTX 5080 (16 ГБ). XLM-R-Large обучается ~30 минут (3 эпохи),
ruRoBERTa-large ~10 минут (1 эпоха), обе с 8-битным AdamW.

## Лицензии

Датасет RuReviews распространяется авторами по лицензии Apache-2.0.
