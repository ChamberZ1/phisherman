from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


# allows direct execution of this script for training after preprocessing in main.py
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train phishing detector from processed train/val/test CSV splits."
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--train-file", type=str, default="train.csv")
    parser.add_argument("--val-file", type=str, default="val.csv")
    parser.add_argument("--test-file", type=str, default="test.csv")
    parser.add_argument("--target-col", type=str, default="label")
    parser.add_argument("--text-col", type=str, default="text_combined")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--class-weight", type=str, default="balanced")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--model",
        type=str,
        default="logreg",
        choices=["logreg", "linear_svm"],
        help="Which model to train",
    )
    return parser.parse_args()


def _load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return pd.read_csv(path)


def _to_xy(
    df: pd.DataFrame,
    target_col: str,
    text_col: str,
) -> tuple[pd.Series, pd.Series]:
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe columns")

    if text_col not in df.columns:
        raise KeyError(f"Text column '{text_col}' not found in dataframe columns")

    y = pd.to_numeric(df[target_col], errors="coerce")
    if y.isna().any():
        bad = int(y.isna().sum())
        raise ValueError(f"Target column '{target_col}' contains {bad} non-numeric values")
    y = y.astype(int)

    X = df[text_col].fillna("").astype(str)
    return X, y


def _evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    y_pred = model.predict(X)
    metrics: dict[str, Any] = {
        "rows": int(len(y)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
    }

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y, y_prob))

    return metrics


def run_training(args: argparse.Namespace) -> tuple[Path, Path]:
    train_path = args.processed_dir / args.train_file
    val_path = args.processed_dir / args.val_file
    test_path = args.processed_dir / args.test_file

    train_df = _load_split(train_path)
    val_df = _load_split(val_path)
    test_df = _load_split(test_path)

    X_train, y_train = _to_xy(train_df, target_col=args.target_col, text_col=args.text_col)
    X_val, y_val = _to_xy(val_df, target_col=args.target_col, text_col=args.text_col)
    X_test, y_test = _to_xy(test_df, target_col=args.target_col, text_col=args.text_col)

    class_weight = None if args.class_weight.lower() == "none" else args.class_weight

    model_name = getattr(args, "model", "logreg")

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=20000,
        min_df=3,
    )

    if model_name == "logreg":
        clf = LogisticRegression(
            max_iter=args.max_iter,
            class_weight=class_weight,
            random_state=args.random_state,
        )
    else:
        clf = LinearSVC(
            max_iter=args.max_iter,
            class_weight=class_weight,
            random_state=args.random_state,
        )

    model = Pipeline(
        steps=[
            ("tfidf", vectorizer),
            ("clf", clf),
        ]
    )

    model.fit(X_train, y_train)

    metrics = {
        "train": _evaluate(model, X_train, y_train),
        "val": _evaluate(model, X_val, y_val),
        "test": _evaluate(model, X_test, y_test),
        "num_features": int(model.named_steps["tfidf"].get_feature_names_out().shape[0]),
    }

    args.models_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.models_dir / f"phish_{model_name}.joblib"
    metrics_path = args.models_dir / f"phish_{model_name}_metrics.json"

    dump(
        {
            "model": model,
            "target_col": args.target_col,
            "text_col": args.text_col,
        },
        model_path,
    )

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print("Validation metrics:")
    print(json.dumps(metrics["val"], indent=2))
    print("Test metrics:")
    print(json.dumps(metrics["test"], indent=2))

    return model_path, metrics_path


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
