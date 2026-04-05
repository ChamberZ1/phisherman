Training transformer... [GUH](docs/training_transformer.png)

# Phisherman

## Cascading Detection Engine
1) Rule-Based - Currently limited to text, but if given full email details could use SPF, DKIM, SMARC, and TLS as signals.
2) Classical ML Layer - Linear Support Vector Model (SVM) - It was either this or Logistic Regression. Both models had good performance but SVM was slightly better. 
3) Unsupervised Layer - Isolation Forest. Trained on benign-only data to detect anomalies. Achieved ROC-AUC of ~0.61 — too weak to make reliable decisions without causing false positives. Included in the cascade as a passive scoring layer only (score is reported in output but does not affect the final verdict).
4) Deep Learning Layer - Lightweight transformer model DistilBERT. Keep system modular so it can be swapped out for a heavier more accurate model if desired in the future.

### Transformer (DistilBERT) training
This project uses Hugging Face Transformers for the deep learning layer. Example:

```bash
.\.venv\Scripts\python -m src.layers.transformer.train_transformer --model-name distilbert-base-uncased
```

### Datasets:
1) [Enron and Nazario in one](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)
2) [Kaggle dataset with only Email Text and Type](https://www.kaggle.com/datasets/subhajournal/phishingemails/data)

### Diagram
[Phisherman.pdf](docs/Phisherman.pdf)
