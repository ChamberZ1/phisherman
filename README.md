Training transformer... [GUH](docs/training_transformer.png)

# Phisherman

A cascading phishing email detection engine combining rule-based heuristics, classical ML, a transformer, and anomaly detection.

---

## Cascading Detection Engine

Each layer is independent. When a layer flags an email as phishing, the cascade short-circuits and returns immediately — later layers are skipped.

1. **Rule-Based** — Weighted heuristic rules (IP URLs, domain mismatches, risky TLDs, etc.). Score ≥ 3 flags as phishing. Currently limited to text signals, but could be extended with SPF, DKIM, DMARC, and TLS checks.
2. **Supervised (SVM)** — TF-IDF + LinearSVC trained on labeled email data. Probability ≥ 0.7 flags as phishing.
3. **Transformer (DistilBERT)** — Fine-tuned HuggingFace transformer for deep semantic classification. Probability ≥ 0.7 flags as phishing. Modular — can be swapped for a heavier model.
4. **Isolation Forest** — Trained on benign-only data to detect anomalies. Achieved ROC-AUC ~0.61 — too weak to make reliable decisions without excessive false positives. Included as a **passive scoring layer only**: its score is reported in output but does not affect the final verdict.

---

## Project Structure

```
main.py                         # Data prep: loads raw CSVs, cleans, splits into train/val/test
app.py                          # Flask web app serving the detection UI
src/
  cascade.py                    # Inference: ties all four layers together
  data_loader.py                # Loads and normalises raw CSV datasets
  preprocessing.py              # Cleaning, deduplication, stratified splitting
  features.py                   # Feature engineering (text, URL, sender, typosquat)
  email_detection_constants.py  # Shared regex patterns and domain lists
  dataset_config.py             # Discovers raw CSVs and builds load configs
  filter_label.py               # Produces benign-only dataset for Isolation Forest training
  typosquat_risk.py             # Brand similarity and typosquatting detection
  layers/
    rules/baseline_rules.py     # Rule definitions and scoring logic
    supervised/train_ml.py      # Train SVM or Logistic Regression
    transformer/train_transformer.py  # Fine-tune DistilBERT
    unsupervised/train_isolation_forest.py  # Train Isolation Forest
web/
  index.html                    # Frontend UI
tests/                          # Pytest test suite
```

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e .
```

---

## Training from Scratch

Run these steps in order:

### 1. Prepare processed splits
Reads all CSVs from `data/raw/`, cleans and deduplicates, saves train/val/test to `data/processed/`.
```bash
python main.py
```

### 2. Generate benign-only dataset (for Isolation Forest)
Filters the raw data to label=0 emails and saves to `data/processed/benign_only.csv`.
```bash
python -m src.filter_label
```

### 3. Train each model

**Supervised (SVM):**
```bash
python -m src.layers.supervised.train_ml --model linear_svm
```

**Transformer (DistilBERT):**
```bash
python -m src.layers.transformer.train_transformer
```

**Isolation Forest:**
```bash
python -m src.layers.unsupervised.train_isolation_forest
```

Trained models are saved to `models/`.

---

## Running the Web App

```bash
python app.py
```

Open `http://127.0.0.1:5000`. The models load once at startup (transformer takes ~10 seconds). Enter a `from_address`, optional subject, and optional body to analyse an email.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Datasets

1. [Enron and Nazario combined](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)
2. [Email Text and Type](https://www.kaggle.com/datasets/subhajournal/phishingemails/data)

**Known limitation:** Training data is dominated by older corporate email (Enron, ~2001) and academic mailing lists. Modern marketing and transactional emails (newsletters, notifications) may be misclassified as phishing. Retraining with more diverse benign data would improve accuracy.

---

## Diagram

[Phisherman.pdf](docs/Phisherman.pdf)
