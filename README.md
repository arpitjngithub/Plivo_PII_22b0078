# PII NER Assignment — Noisy STT Entity Recognition

This repository implements a **token-level NER model** for detecting PII entities in **noisy Speech-to-Text (STT)** transcripts.  
The system outputs **character-level spans** with correct PII flags and follows the assignment constraints:

- Learned model (DistilBERT token classifier)
- High PII precision
- p95 CPU latency ≤ 20 ms
- Noisy, synthetic STT-style dataset generation included
- Post-processing for precision on PII entities (EMAIL / PHONE / CREDIT_CARD)

---

## 📁 Project Structure

src/
train.py # Training script (with collate + freeze_base)
predict.py # Inference + span decoding + confidence filtering
eval_span_f1.py # Span-level evaluation (PII + non-PII)
measure_latency.py # p50 / p95 CPU latency measurement
dataset.py # Tokenization + BIO encoding + custom collate
generate_data.py # Synthetic noisy-STT dataset generator

data/
train.jsonl # 1000 synthetic noisy-STT examples (generated)
dev.jsonl # 200 examples (generated)
stress.jsonl # Stress-test set (provided)
test.jsonl # Unlabeled

out/
(created after training)


---

## 🚀 Setup

```bash
pip install -r requirements.txt
🎯 Dataset Generation (Optional)

To regenerate noisy-STT synthetic train/dev sets:

python src/generate_data.py --train_out data/train.jsonl --dev_out data/dev.jsonl

🧠 Train the Model
python src/train.py \
  --model_name distilbert-base-uncased \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --out_dir out


Optional: freeze the transformer base for faster training & lower latency:

python src/train.py --freeze_base ...

🔎 Predict
python src/predict.py \
  --model_dir out \
  --input data/dev.jsonl \
  --output out/dev_pred.json

📊 Evaluate
Dev Set
python src/eval_span_f1.py \
  --gold data/dev.jsonl \
  --pred out/dev_pred.json

Stress Test (Optional)
python src/predict.py \
  --model_dir out \
  --input data/stress.jsonl \
  --output out/stress_pred.json

python src/eval_span_f1.py \
  --gold data/stress.jsonl \
  --pred out/stress_pred.json

⚡ Latency (Assignment Requirement)
python src/measure_latency.py \
  --model_dir out \
  --input data/dev.jsonl \
  --runs 50

✅ Final Latency Results (CPU, batch_size=1):
p50: 12.88 ms
p95: 19.25 ms


✔ Passes requirement: p95 ≤ 20 ms

📈 Final Metrics
Stress Set Metrics
Per-entity metrics:
CITY            P=0.706 R=0.600 F1=0.649
CREDIT_CARD     P=0.000 R=0.000 F1=0.000
DATE            P=0.471 R=0.600 F1=0.527
EMAIL           P=0.000 R=0.000 F1=0.000
PERSON_NAME     P=0.472 R=0.625 F1=0.538
PHONE           P=0.000 R=0.000 F1=0.000

Macro-F1: 0.286

PII-only metrics:
  Precision = 0.471
  Recall    = 0.245
  F1        = 0.322

Non-PII metrics:
  Precision = 0.706
  Recall    = 0.600
  F1        = 0.649
