import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)


class TextDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        self.labels = labels

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self) -> int:
        return len(self.labels)


def _load_split(path: Path, text_col: str, target_col: str) -> tuple[list[str], list[int]]:
    df = pd.read_csv(path)
    if text_col not in df.columns:
        raise ValueError(f"Missing text column '{text_col}' in {path}")
    if target_col not in df.columns:
        raise ValueError(f"Missing target column '{target_col}' in {path}")
    texts = df[text_col].fillna("").astype(str).tolist()
    labels = df[target_col].astype(int).tolist()
    return texts, labels


def _compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, preds)

    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = exp / exp.sum(axis=-1, keepdims=True)
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a DistilBERT (or other HF transformer) classifier."
    )
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--train-file", type=str, default="train.csv")
    parser.add_argument("--val-file", type=str, default="val.csv")
    parser.add_argument("--test-file", type=str, default="test.csv")
    parser.add_argument("--text-col", type=str, default="text_combined")
    parser.add_argument("--target-col", type=str, default="label")
    parser.add_argument(
        "--model-name",
        type=str,
        default="distilbert-base-uncased",
        help="Any Hugging Face model name or local path.",
    )
    parser.add_argument("--output-dir", type=str, default="models/distilbert")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    train_path = Path(args.processed_dir) / args.train_file
    val_path = Path(args.processed_dir) / args.val_file
    test_path = Path(args.processed_dir) / args.test_file

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )

    train_texts, train_labels = _load_split(train_path, args.text_col, args.target_col)
    val_texts, val_labels = _load_split(val_path, args.text_col, args.target_col)
    test_texts, test_labels = _load_split(test_path, args.text_col, args.target_col)

    train_ds = TextDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_ds = TextDataset(val_texts, val_labels, tokenizer, args.max_length)
    test_ds = TextDataset(test_texts, test_labels, tokenizer, args.max_length)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=args.seed,
        logging_steps=50,
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=_compute_metrics,
    )

    # Resume from the latest checkpoint if one exists (makes training resume-safe
    # after an unexpected shutdown).
    latest_checkpoint = None
    checkpoints = []
    for p in output_dir.glob("checkpoint-*"):
        try:
            checkpoints.append((int(p.name.split("-")[-1]), p))
        except ValueError:
            pass
    if checkpoints:
        latest_checkpoint = str(max(checkpoints, key=lambda x: x[0])[1])
        print(f"Resuming from checkpoint: {latest_checkpoint}")

    print(f"Training device: {training_args.device}")
    trainer.train(resume_from_checkpoint=latest_checkpoint)
    metrics = {
        "val": trainer.evaluate(),
        "test": trainer.evaluate(test_ds),
    }

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model: {output_dir}")
    print(f"Saved metrics: {metrics_path}")
    return metrics


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
