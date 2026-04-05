from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from joblib import load
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.features import build_features
from src.layers.rules.baseline_rules import classify_with_rules

import pandas as pd


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_supervised(model_path: str | Path) -> Any:
    """Load a joblib-serialised sklearn pipeline (e.g. textcombined_svm)."""
    bundle = load(model_path)
    return bundle


def load_isolation_forest(model_path: str | Path) -> Any:
    """Load a joblib-serialised Isolation Forest bundle."""
    return load(model_path)


def load_transformer(model_dir: str | Path) -> tuple[Any, Any]:
    """Load a fine-tuned HuggingFace transformer and its tokenizer."""
    model_dir = str(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Per-layer inference helpers
# ---------------------------------------------------------------------------

def _predict_supervised(bundle: dict, record: dict) -> float:
    """Return phishing probability from the supervised sklearn pipeline."""
    text_col = bundle.get("text_col", "text_combined")
    text = record.get(text_col) or record.get("text_combined", "")
    model = bundle["model"]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([text])[0][1]
    else:
        # LinearSVC has no predict_proba — use decision_function and sigmoid
        score = model.decision_function([text])[0]
        proba = float(1 / (1 + np.exp(-score)))
    return float(proba)


def _predict_isolation_forest(bundle: dict, record: dict) -> float:
    """Return a normalised anomaly score (0–1, higher = more anomalous)."""
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    df = pd.DataFrame([record])
    df = build_features(df)
    present = [c for c in feature_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    X = df[present].fillna(0).astype(float)

    # score_samples returns negative values; more negative = more anomalous.
    # Negate and apply min-max normalisation using the model's threshold as anchor.
    raw_score = -model.score_samples(X)[0]
    # Sigmoid to squash into 0–1
    normalised = float(1 / (1 + np.exp(-raw_score)))
    return normalised


def _predict_transformer(
    model: Any,
    tokenizer: Any,
    record: dict,
    max_length: int = 256,
    temperature: float = 2.0,
) -> float:
    """Return phishing probability from the transformer model.

    temperature > 1 softens overconfident logits, producing probabilities
    that better reflect actual uncertainty rather than collapsing to 0/1.
    """
    text = record.get("text_combined", "")
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits / temperature, dim=-1)
    return float(probs[0][1].item())


# ---------------------------------------------------------------------------
# Cascade
# ---------------------------------------------------------------------------

class PhishingCascade:
    """Four-layer phishing detection cascade.

    Layers (in order):
      1. Rule-based  — weighted heuristic rules, fast pre-filter
      2. Supervised  — TF-IDF + LinearSVC, strong baseline classifier
      3. Transformer — DistilBERT fine-tuned classifier, deep semantic layer
      4. Isolation Forest — anomaly detection, weak last-resort layer

    Args:
        supervised_model_path: Path to the joblib file for the supervised model.
        isolation_forest_path: Path to the joblib file for the Isolation Forest.
        transformer_dir: Directory containing the fine-tuned transformer.
        rule_threshold: Minimum rule score to immediately flag as phishing.
        supervised_threshold: Probability above which supervised flags as phishing.
        transformer_threshold: Probability above which transformer flags as phishing.
        isolation_threshold: Anomaly score above which Isolation Forest flags as phishing.
        transformer_max_length: Max token length for the transformer.
    """

    def __init__(
        self,
        supervised_model_path: str | Path = Path("models/textcombined_svm.joblib"),
        isolation_forest_path: str | Path = Path("models/isolation_forest.joblib"),
        transformer_dir: str | Path = Path("models/distilbert"),
        rule_threshold: int = 3,
        supervised_threshold: float = 0.7,
        transformer_threshold: float = 0.7,
        isolation_threshold: float = 0.75,
        transformer_max_length: int = 256,
    ) -> None:
        self.rule_threshold = rule_threshold
        self.supervised_threshold = supervised_threshold
        self.transformer_threshold = transformer_threshold
        self.isolation_threshold = isolation_threshold
        self.transformer_max_length = transformer_max_length

        self._supervised = load_supervised(supervised_model_path)
        self._if_bundle = load_isolation_forest(isolation_forest_path)
        self._transformer, self._tokenizer = load_transformer(transformer_dir)

    def predict(self, record: dict) -> dict[str, Any]:
        """Run the cascade on a single email record and return a detailed result.

        Args:
            record: Dict with keys ``from_address``, ``subject``, ``body``,
                    and optionally ``text_combined`` and ``has_url``.

        Returns:
            Dict with ``is_phish``, ``confidence``, ``triggered_by``, and
            per-layer scores.
        """
        # Ensure text_combined exists for layers that need it
        if "text_combined" not in record or not record["text_combined"]:
            record = dict(record)
            record["text_combined"] = (
                "FROM: " + str(record.get("from_address", ""))
                + " SUBJECT: " + str(record.get("subject", ""))
                + " BODY: " + str(record.get("body", ""))
            )

        result: dict[str, Any] = {
            "is_phish": 0,
            "confidence": 0.0,
            "triggered_by": None,
            "rule_score": None,
            "rule_matches": [],
            "supervised_proba": None,
            "transformer_proba": None,
            "isolation_score": None,
        }

        # --- Layer 1: Rules ---
        rule_result = classify_with_rules(record, threshold=self.rule_threshold)
        result["rule_score"] = rule_result["score"]
        result["rule_matches"] = [m["rule_id"] for m in rule_result["matches"]]

        if rule_result["is_phish"]:
            result["is_phish"] = 1
            result["confidence"] = 1.0
            result["triggered_by"] = "rules"
            return result

        # --- Layer 2: Supervised ---
        supervised_proba = _predict_supervised(self._supervised, record)
        result["supervised_proba"] = supervised_proba

        if supervised_proba >= self.supervised_threshold:
            result["is_phish"] = 1
            result["confidence"] = supervised_proba
            result["triggered_by"] = "supervised"
            return result

        # --- Layer 3: Transformer ---
        transformer_proba = _predict_transformer(
            self._transformer, self._tokenizer, record, self.transformer_max_length
        )
        result["transformer_proba"] = transformer_proba

        if transformer_proba >= self.transformer_threshold:
            result["is_phish"] = 1
            result["confidence"] = transformer_proba
            result["triggered_by"] = "transformer"
            return result

        # --- Layer 4: Isolation Forest (scoring only) ---
        # Isolation Forest achieved ROC-AUC of ~0.61 — too weak to make final decisions
        # without causing excessive false positives. Its score is included in the output
        # for informational purposes only and does not affect the final verdict.
        isolation_score = _predict_isolation_forest(self._if_bundle, record)
        result["isolation_score"] = isolation_score

        # Benign — fill in remaining scores and set confidence to how far below threshold
        result["is_phish"] = 0
        result["confidence"] = 1.0 - max(supervised_proba, transformer_proba)
        result["triggered_by"] = None
        return result


__all__ = ["PhishingCascade"]
