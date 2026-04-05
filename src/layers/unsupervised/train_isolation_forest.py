from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.features import build_features


# Numerical feature columns produced by build_features() — these are the only
# columns fed to the Isolation Forest (no raw text).
FEATURE_COLS: list[str] = [
    # text stats
    "text_len", "body_len", "subject_len", "subject_missing",
    "num_words_text", "num_digits_text", "num_currency_symbols_text",
    "has_crypto_terms", "num_crypto_term_hits", "char_entropy_text",
    "num_exclamations_body", "num_question_marks_body",
    "num_uppercase_chars_body", "uppercase_ratio_body",
    "body_to_subject_len_ratio",
    "num_html_tags_body", "has_html_tags_body",
    "num_bare_domain_links_body", "has_bare_domain_links_body",
    "has_urgent_terms", "has_action_terms", "has_credential_terms",
    "num_suspicious_keyword_hits",
    # URL stats
    "num_urls_in_text", "num_short_urls",
    "first_url_has_https", "first_url_len",
    "first_url_domain_len", "first_url_domain_num_dots",
    "first_url_domain_num_hyphens", "first_url_domain_is_ip",
    "has_url", "has_url_was_missing", "tld_risk_flag",
    # sender stats
    "sender_missing", "sender_domain_len", "sender_local_len",
    "sender_has_digits", "sender_is_free_email", "sender_url_domain_mismatch",
    # typosquatting
    "sender_brand_similarity", "sender_brand_edit_distance",
    "sender_domain_punycode", "sender_domain_non_ascii", "sender_typosquat_flag",
    "url_brand_similarity", "url_brand_edit_distance",
    "url_domain_punycode", "url_domain_non_ascii", "url_typosquat_flag",
    # metadata
    "source_missing", "from_subject_missing",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an Isolation Forest anomaly detector on engineered email features."
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--train-file", type=str, default="benign_only.csv")  # changed to benign_only
    parser.add_argument("--val-file", type=str, default="val.csv")
    parser.add_argument("--test-file", type=str, default="test.csv")
    parser.add_argument("--target-col", type=str, default="label")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument(
        "--contamination",
        type=lambda x: float(x) if x != "auto" else "auto",
        default="auto",
        help="Expected fraction of outliers in the training set, or 'auto'.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _load_features(path: Path, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    df = build_features(df)
    # Keep only the engineered feature columns that were actually produced.
    present = [c for c in FEATURE_COLS if c in df.columns]
    # X is the numeric feature matrix used to train/evaluate the model.
    X = df[present].fillna(0).astype(float)
    # y is the ground-truth label column converted to integer classes.
    y = df[target_col].astype(int)
    return X, y


def _if_predict_binary(model: IsolationForest, X: pd.DataFrame) -> np.ndarray:
    # IF returns 1 (inlier/legit) and -1 (outlier/phishing). Map to 0/1.
    raw = model.predict(X)
    return np.where(raw == -1, 1, 0)


def _evaluate(model: IsolationForest, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    preds = _if_predict_binary(model, X)
    # score_samples: lower = more anomalous. Negate so higher = more phishing.
    scores = -model.score_samples(X)
    metrics: dict[str, Any] = {
        "rows": int(len(y)),
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y, scores))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def run_training(args: argparse.Namespace) -> tuple[Path, Path]:
    X_train, y_train = _load_features(args.processed_dir / args.train_file, args.target_col)
    X_val, y_val = _load_features(args.processed_dir / args.val_file, args.target_col)
    X_test, y_test = _load_features(args.processed_dir / args.test_file, args.target_col)

    model = IsolationForest(
        n_estimators=args.n_estimators,
        contamination=args.contamination,
        random_state=args.random_state,
        n_jobs=-1,
    )
    model.fit(X_train)  # y_train (label) is not used since it's unsupervised

    metrics = {
        "train": _evaluate(model, X_train, y_train),
        "val": _evaluate(model, X_val, y_val),
        "test": _evaluate(model, X_test, y_test),
        "feature_cols": list(X_train.columns),
    }

    args.models_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.models_dir / "isolation_forest.joblib"
    metrics_path = args.models_dir / "isolation_forest_metrics.json"

    dump({"model": model, "feature_cols": list(X_train.columns)}, model_path)
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
