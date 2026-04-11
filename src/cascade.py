from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from joblib import load
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.features import build_features
from src.layers.rules.baseline_rules import evaluate_baseline_rules, _root_domain
from src.email_detection_constants import TRUSTED_SENDING_DOMAINS, TRUSTED_FROM_SUBDOMAINS

import pandas as pd


# Rules that are high-confidence technical signals strong enough to trigger a
# phishing verdict on their own, without requiring ML agreement.
HARD_BLOCK_RULES: frozenset[str] = frozenset({"ip_url", "punycode_domain", "malicious_attachment_ext"})


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
    """Phishing detection cascade with consensus voting.

    All four layers run on every email. A phishing verdict requires agreement
    from at least two independent sources, which dramatically reduces false
    positives from any single over-eager model.

    Decision logic (evaluated in order):
      1. Hard-block rules (ip_url, punycode_domain) — fire immediately, no
         ML agreement needed. These are narrow, high-precision technical
         signals with almost no legitimate use.
      2. ML consensus — supervised AND transformer both exceed their threshold.
      3. Rule-assisted — rule score >= rule_vote_threshold AND at least one
         ML model exceeds its threshold.
      4. Benign — fewer than two sources agree.

    The Isolation Forest (ROC-AUC ~0.61) is too weak to vote but its score is
    included in every result for diagnostic purposes.

    Args:
        supervised_model_path: Path to the joblib file for the supervised model.
        isolation_forest_path: Path to the joblib file for the Isolation Forest.
        transformer_dir: Directory containing the fine-tuned transformer.
        rule_vote_threshold: Minimum rule score (from non-hard-block rules) for
            the rule layer to contribute a vote toward a rule-assisted verdict.
        supervised_threshold: Probability above which supervised casts a phish vote.
        transformer_threshold: Probability above which transformer casts a phish vote.
        transformer_certainty_threshold: Probability above which the transformer
            triggers a verdict alone, without requiring a second vote. Reserved for
            cases where the transformer is near-certain (default 0.995).
        transformer_max_length: Max token length for the transformer.
    """

    def __init__(
        self,
        supervised_model_path: str | Path = Path("models/textcombined_svm.joblib"),
        isolation_forest_path: str | Path = Path("models/isolation_forest.joblib"),
        transformer_dir: str | Path = Path("models/distilbert"),
        rule_vote_threshold: int = 3,
        supervised_threshold: float = 0.55,
        transformer_threshold: float = 0.75,
        transformer_certainty_threshold: float = 0.995,
        transformer_max_length: int = 256,
    ) -> None:
        self.rule_vote_threshold = rule_vote_threshold
        self.supervised_threshold = supervised_threshold
        self.transformer_threshold = transformer_threshold
        self.transformer_certainty_threshold = transformer_certainty_threshold
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
            per-layer scores. ``triggered_by`` is one of ``"rules"``,
            ``"ml_consensus"``, ``"rule_assisted"``, or ``None`` (benign).
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
        rule_result = evaluate_baseline_rules(record)
        result["rule_score"] = rule_result["score"]
        matched_ids = {m["rule_id"] for m in rule_result["matches"]}
        result["rule_matches"] = list(matched_ids)

        # Hard-block: narrow, high-precision technical signals that require no
        # ML confirmation (raw IP URLs, punycode/homograph domains).
        if matched_ids & HARD_BLOCK_RULES:
            result["is_phish"] = 1
            result["confidence"] = 1.0
            result["triggered_by"] = "rules"
            return result

        # --- Layer 2: Supervised ---
        supervised_proba = _predict_supervised(self._supervised, record)
        result["supervised_proba"] = supervised_proba

        # --- Layer 3: Transformer ---
        transformer_proba = _predict_transformer(
            self._transformer, self._tokenizer, record, self.transformer_max_length
        )
        result["transformer_proba"] = transformer_proba

        # --- Layer 4: Isolation Forest (scoring only) ---
        # ROC-AUC ~0.61 — too weak to vote on its own. Included for diagnostics.
        isolation_score = _predict_isolation_forest(self._if_bundle, record)
        result["isolation_score"] = isolation_score

        # --- DKIM trusted sender check ---
        # If the email has a verified DKIM signature from a known-trusted domain,
        # suppress all ML verdict paths to avoid false positives on legitimate
        # transactional email. Hard-block rules still fire regardless.
        dkim_pass = bool(record.get("dkim_pass", False))
        dkim_domain = record.get("dkim_domain") or ""
        dkim_root = _root_domain(dkim_domain)

        # Broad trust: DKIM root domain is in the trusted sending list
        broad_trusted = dkim_pass and bool(dkim_domain) and dkim_root in TRUSTED_SENDING_DOMAINS

        # Subdomain trust: platform whose broad domain is abusable (e.g. google.com
        # via Forms/Sites) but specific From subdomains are not (e.g. accounts.google.com)
        from_domain = ""
        from_addr = record.get("from_address") or ""
        if "@" in from_addr:
            from_domain = from_addr.split("@")[-1].lower()
        subdomain_trusted = (
            dkim_pass
            and dkim_root in TRUSTED_FROM_SUBDOMAINS
            and from_domain in TRUSTED_FROM_SUBDOMAINS[dkim_root]
        )

        dkim_trusted = broad_trusted or subdomain_trusted

        # --- Consensus voting ---
        supervised_votes_phish = supervised_proba >= self.supervised_threshold
        transformer_votes_phish = transformer_proba >= self.transformer_threshold
        rules_vote = rule_result["score"] >= self.rule_vote_threshold

        # For DKIM-verified trusted senders, all ML-based verdict paths are suppressed.
        # A cryptographic DKIM signature from a known domain is more reliable than
        # models trained on 2001 data that cannot distinguish legitimate transactional
        # email ("verify your account") from phishing with the same language.
        # Hard-block rules still fire regardless — a verified sender with a raw IP URL
        # or punycode domain is genuinely suspicious.

        # Transformer near-certainty override — at ≥0.995 it acts alone
        if not dkim_trusted and transformer_proba >= self.transformer_certainty_threshold:
            result["is_phish"] = 1
            result["confidence"] = transformer_proba
            result["triggered_by"] = "transformer_certain"
            return result

        # Transformer high-confidence + supervised corroboration (0.99 ≤ trf < 0.995)
        if not dkim_trusted and transformer_proba >= 0.99 and supervised_votes_phish:
            result["is_phish"] = 1
            result["confidence"] = (transformer_proba + supervised_proba) / 2
            result["triggered_by"] = "transformer_corroborated"
            return result

        # Transformer near-certain + rule corroboration (0.95 ≤ trf < 0.99, rules ≥ 2)
        if not dkim_trusted and transformer_proba >= 0.95 and rule_result["score"] >= 2:
            result["is_phish"] = 1
            result["confidence"] = transformer_proba
            result["triggered_by"] = "transformer_rule_corroborated"
            return result

        # Transformer dominant — very high confidence with weak supervised corroboration.
        # Supervised ≥ 0.40 acts as a sanity check (not a full vote): ensures the supervised
        # model isn't actively calling the email benign before trusting the transformer alone.
        if not dkim_trusted and transformer_proba >= 0.97 and supervised_proba >= 0.40:
            result["is_phish"] = 1
            result["confidence"] = transformer_proba
            result["triggered_by"] = "transformer_dominant"
            return result

        # Both ML models independently agree → high confidence verdict
        if not dkim_trusted and supervised_votes_phish and transformer_votes_phish:
            result["is_phish"] = 1
            result["confidence"] = (supervised_proba + transformer_proba) / 2
            result["triggered_by"] = "ml_consensus"
            return result

        # Rule signal + one ML model → rule-assisted verdict
        if not dkim_trusted and rules_vote and (supervised_votes_phish or transformer_votes_phish):
            result["is_phish"] = 1
            result["confidence"] = max(supervised_proba, transformer_proba)
            result["triggered_by"] = "rule_assisted"
            return result

        # Benign — confidence reflects how far below threshold the strongest signal is
        result["is_phish"] = 0
        result["confidence"] = 1.0 - max(supervised_proba, transformer_proba)
        result["triggered_by"] = None
        return result


__all__ = ["PhishingCascade", "HARD_BLOCK_RULES"]
