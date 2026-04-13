# Phisherman

A phishing email detection engine combining rule-based heuristics, classical ML, a transformer, and anomaly detection.

---

## Detection Architecture

The original intent was a strict cascade — each layer would only run if the previous one was inconclusive, keeping inference fast and cheap. In practice, the layers are too unreliable in isolation: the rule layer fires on legitimate bulk email, the SVM misfires on modern transactional language, and the transformer occasionally hallucinates high confidence on benign content. Running a single layer to a verdict produced too many false positives.

The architecture evolved into a consensus model: all four layers run on every email, and the verdict is determined by a priority-ordered set of decision rules. High-confidence signals (hard-block rules, near-certain transformer) can act alone; weaker signals require corroboration from a second layer. This trades some inference speed for a meaningful reduction in false positives without sacrificing recall on clear-cut cases.

**Decision logic (evaluated in order):**

1. **Hard-block rules** — If `ip_url`, `punycode_domain`, or `malicious_attachment_ext` fires, the email is flagged immediately. These fire regardless of sender trust — a verified sender with a raw IP URL or punycode domain is still suspicious.
2. **DKIM trusted sender exemption** — If the email has a valid DKIM signature from a domain in the trusted sender list (major brands and their known ESP relay domains), all ML-based verdict paths below are suppressed. A cryptographic DKIM signature is more reliable than models trained on 2001 data for distinguishing legitimate transactional email from phishing.
3. **Transformer near-certain** — If transformer probability ≥ 0.995, it acts alone (`triggered_by: "transformer_certain"`).
4. **Transformer corroborated** — If transformer ≥ 0.99 *and* supervised ≥ 0.55, the email is flagged (`triggered_by: "transformer_corroborated"`).
5. **Transformer + rule corroborated** — If transformer ≥ 0.95 *and* rule score ≥ 2, the email is flagged (`triggered_by: "transformer_rule_corroborated"`). Catches cases where supervised undershoots but transformer and rules agree.
6. **Transformer dominant** — If transformer ≥ 0.97 *and* supervised ≥ 0.40, the email is flagged (`triggered_by: "transformer_dominant"`). The supervised floor is a sanity check — it prevents firing when the supervised model actively disagrees, without requiring a full vote.
7. **ML consensus** — If both supervised (≥ 0.55) and transformer (≥ 0.75) independently exceed their thresholds, the email is flagged (`triggered_by: "ml_consensus"`).
8. **Rule-assisted** — If rule score ≥ 3 *and* at least one ML model exceeds its threshold, the email is flagged (`triggered_by: "rule_assisted"`). Soft rules alone are not sufficient without ML corroboration.
9. **Benign** — No verdict paths triggered.

**Layers:**

1. **Rule-Based** — Weighted heuristic rules (IP URLs, domain mismatches, risky TLDs, shorteners, malicious attachment extensions, etc.). Three rules (`ip_url`, `punycode_domain`, `malicious_attachment_ext`) act as hard blocks; the rest contribute to a vote score. Could be extended with SPF, DKIM, DMARC, and TLS checks.
2. **Supervised (Support Vector Machine)** — TF-IDF + LinearSVC trained on labeled email data. Probability ≥ 0.55 casts a phishing vote.
3. **Transformer (DistilBERT)** — Fine-tuned HuggingFace transformer for deep semantic classification. Probability ≥ 0.75 casts a phishing vote. Modular — can be swapped for a heavier model.
4. **Unsupervised (Isolation Forest)** — Trained on benign-only data to detect anomalies. Achieved ROC-AUC ~0.61 — too weak to vote reliably. Included as a **passive scoring layer only**: its score is reported in output but does not affect the final verdict.

---

## Project Structure

```
main.py                         # Data prep: loads raw CSVs, cleans, splits into train/val/test
app.py                          # Flask web app serving the detection UI
download_models.py              # Downloads pre-trained weights from Hugging Face
evaluate_batch.py               # Batch evaluation of a directory of .eml files
src/
  cascade.py                    # Inference: ties all four layers together
  eml_parser.py                 # Parses raw .eml files (MIME, href extraction, SafeLinks decoding)
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
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
python download_models.py
```

Model weights are hosted on Hugging Face: [zchf420/phisherman](https://huggingface.co/zchf420/phisherman). `download_models.py` fetches them automatically into the expected `models/` paths.

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

![Flask Web App](/docs/web-app.png)

```bash
python app.py
```

Open `http://127.0.0.1:5000`. The models load once at startup (transformer takes ~10 seconds).

**Uploading a `.eml` file is strongly recommended over pasting plain text.** The upload path parses the raw MIME, extracts real `href` URLs from HTML (rather than visible display text), decodes Microsoft SafeLinks-wrapped URLs, and detects attachment metadata — all of which are important phishing signals. Pasting plain text loses all of this. Most email clients let you export a `.eml` via *Save As* or *Download as EML*.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Datasets

**Dataset 1 — [Phishing Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)** (seven combined sources):
- `CEAS_08.csv` — CEAS 2008 spam corpus
- `Enron.csv` — Enron corporate email corpus (~2001)
- `Ling.csv` — Ling-Spam academic mailing list dataset
- `Nazario.csv` — Nazario phishing corpus
- `Nigerian_Fraud.csv` — Nigerian advance-fee fraud emails
- `SpamAssasin.csv` — SpamAssassin public corpus
- `body_only.csv` — Body-only variant of the above

**Dataset 2 — [Email Text and Type](https://www.kaggle.com/datasets/subhajournal/phishingemails/data)**  
Documented as an additional candidate corpus, but not currently included in the checked-in `data/raw` training set.

---

## Evaluation

**[phishing_pot](https://github.com/rf-peixoto/phishing_pot)** — a community-maintained repository of real-world phishing `.eml` files. Used to measure the false negative rate of the cascade and tune consensus thresholds. Not used for training.

```bash
git clone https://github.com/rf-peixoto/phishing_pot
python evaluate_batch.py --eml-dir phishing_pot/email --label phish --output results.csv --limit 0
```

For the phishing corpus snapshot used in this repo, a full run reported **94.7%** phishing detection rate (excluding parse errors).

**Benign evaluation** — a personal collection of real legitimate `.eml` files (transactional email, newsletters, account notifications from major services) used to measure the false positive rate.

```bash
python evaluate_batch.py --eml-dir data/benign_eml --label benign --output benign_results.csv --limit 0
```

For the current benign corpus snapshot in this repo, the latest full run reported **98.8%** benign pass rate (excluding parse errors).


The `evaluate_batch.py` script parses each `.eml` through the same pipeline as the web app (MIME parsing, SafeLinks decoding, href extraction) and outputs three files: `results.csv` (all verdicts), `results_review.csv` (incorrect verdicts — missed phishing when `--label phish`, false positives when `--label benign` — with body preview and layer scores for manual inspection), and `results_errors.csv` (parse failures).

---

## Limitations

### Outdated training data

The models were trained on seven combined Dataset 1 sources: Enron (~2001 corporate email), Ling-Spam, Nazario, CEAS 2008, Nigerian fraud emails, SpamAssassin, and `body_only`. While these datasets yield high held-out accuracy (SVM test F1 ~0.993), that accuracy reflects performance on the same distribution they were trained on — not modern email.

In practice, modern legitimate emails (transactional notifications, newsletters, security alerts) use vocabulary that heavily overlaps with what phishing looked like in 2001. Words like *"password"*, *"account"*, *"verify"*, *"click here"*, and *"secure your account"* appear in both phishing emails from the training data and routine legitimate emails from services like Google, banks, and SaaS providers today. As a result, the supervised model in particular is prone to false positives on modern benign email.

A concrete example: a standard Google security alert ("App password created — check activity at myaccount.google.com") contains credential terms and a call-to-action URL, which the rule layer scores as suspicious and the SVM may flag outright. The consensus architecture introduced in this project mitigates but does not eliminate this problem. Retraining on a diverse, recent dataset that includes modern legitimate transactional email would be the correct long-term fix.

### URL visibility and input quality

When email content is pasted as plain text, all HTML structure is lost and **actual link targets are never seen by the system** — only visible display text. This is significant because the most reliable phishing signal is a mismatch between the displayed URL and the actual `href` attribute (e.g. displaying `https://myaccount.google.com` while the href points to a malicious IP).

The web interface addresses this partially via `.eml` upload: raw MIME is parsed, `href` attributes are extracted from the HTML part, and real link targets are appended to the body before analysis. Microsoft SafeLinks-wrapped URLs are also decoded to their true destination. However, some signal is still lost — email clients that wrap all outbound links through their own tracking or security proxies (common in enterprise environments) obscure the final destination URL, and the system has no way to follow redirect chains to determine where a link ultimately leads.

Additional metadata available in raw `.eml` files — `Reply-To`, `Return-Path`, `Received` chain, `X-Originating-IP` — are currently unused. These headers are strong phishing signals in a production detector but are not extracted or fed to any model layer in this system.

### English-only training data

All training datasets are composed almost entirely of English-language email. This has two consequences for non-English input:

The transformer (DistilBERT) has no meaningful representation of non-English text. Chinese, Arabic, Russian, or other non-Latin-script content is tokenised into rare or unknown subword tokens that the model has never associated with benign email. Because non-English text was essentially absent from the benign training distribution, the model's uncertainty tends to resolve toward the phishing class — meaning any non-English email is likely to receive an artificially elevated phishing score regardless of its actual content.

The supervised SVM is similarly affected: non-English vocabulary falls outside its TF-IDF feature space entirely, leaving the model with near-zero input signal and unpredictable outputs.

In practice, the system is English-only. Non-English emails should be treated as unanalysable by the ML layers; any verdict on them would be driven primarily by URL and sender heuristics rather than content understanding.

### Rule sensitivity on common email patterns

Several heuristic rules fire on patterns that are routine in legitimate bulk email:

- `sender_url_domain_mismatch` — triggers whenever the sending domain differs from the first URL domain. This is normal for newsletters sent via Mailchimp, Sendgrid, or similar services, where the sender is `news@mailchimp.com` but links point to the company's own domain.
- `urgent_or_action` — matches words like *"click"*, *"confirm"*, *"update"*, *"login"*, which appear in almost every marketing and transactional email.
- `credentials_with_url` — matches *"account"*, *"password"*, *"billing"* alongside any URL, which describes the majority of bank, SaaS, and e-commerce emails.

These rules are only dangerous in combination with a hard-block signal or ML agreement (the consensus architecture prevents them from triggering alone), but they inflate the rule vote score and increase the probability of a `rule_assisted` false positive when an ML model is also uncertain.

### Transformer sensitivity to call-to-action language

The transformer is heavily conditioned to treat "click here", "click this button", "click to verify", and similar CTA phrases as strong phishing signals. In the training data this was a reliable indicator — phishing emails from 2001–2008 used aggressive CTAs while legitimate email largely did not. Modern legitimate email (marketing, transactional, SaaS notifications) uses this language constantly. The result is that the transformer's confidence can be disproportionately elevated by a single CTA phrase even when the rest of the email content is benign, contributing to false positives on newsletters and promotional email.

---

## Diagram

[Phisherman.pdf](docs/Phisherman.pdf)
