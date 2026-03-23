# Phisherman

## Cascading Detection Engine
1) Rule-Based
2) Classical ML Layer - Linear Support Vector Model (SVM) - It was either this or Logistic Regression. Both models had good performance but SVM was slightly better. 
3) Unsupervised Layer - Isolation Forest
4) Deep Learning Layer - DistilBERT first, keep system modular, so it can be swapped out for a heavier more accurate model if desired in the future.

### Datasets:
1) [Enron and Nazario in one](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)
2) [Kaggle dataset with only Email Text and Type](https://www.kaggle.com/datasets/subhajournal/phishingemails/data)

### Diagram
[Phisherman.pdf](docs/Phisherman.pdf)
