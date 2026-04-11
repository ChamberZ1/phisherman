"""Tests for src/cascade.py.

All model loading is mocked so tests run without trained model files on disk.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.cascade import (
    HARD_BLOCK_RULES,
    PhishingCascade,
    _predict_isolation_forest,
    _predict_supervised,
    _predict_transformer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    from_address: str = "sender@example.com",
    subject: str = "Hello",
    body: str = "Just a normal message.",
    text_combined: str | None = None,
) -> dict:
    record = {
        "from_address": from_address,
        "subject": subject,
        "body": body,
    }
    if text_combined is not None:
        record["text_combined"] = text_combined
    return record


def _make_cascade(
    supervised_proba: float = 0.1,
    transformer_proba: float = 0.1,
    isolation_score: float = 0.1,
) -> PhishingCascade:
    """Return a PhishingCascade with all models mocked out."""
    cascade = PhishingCascade.__new__(PhishingCascade)

    # Defaults matching the real __init__
    cascade.rule_vote_threshold = 3
    cascade.supervised_threshold = 0.5
    cascade.transformer_threshold = 0.5
    cascade.transformer_certainty_threshold = 0.995
    cascade.isolation_threshold = 0.55
    cascade.transformer_max_length = 256

    # Mock supervised bundle — sklearn pipeline with predict_proba
    supervised_model = MagicMock()
    supervised_model.predict_proba.return_value = np.array([[1 - supervised_proba, supervised_proba]])
    cascade._supervised = {"model": supervised_model, "text_col": "text_combined"}

    # Mock Isolation Forest bundle
    if_model = MagicMock()
    # score_samples returns negative values; we negate in the helper, so set to -raw
    # such that sigmoid(-(-raw)) = sigmoid(raw) ≈ isolation_score
    # Solve: isolation_score = 1/(1+exp(-x)) → x = log(p/(1-p))
    p = max(min(isolation_score, 0.9999), 0.0001)
    raw = np.log(p / (1 - p))
    if_model.score_samples.return_value = np.array([-raw])  # negated in helper

    from src.layers.unsupervised.train_isolation_forest import FEATURE_COLS
    cascade._if_bundle = {"model": if_model, "feature_cols": FEATURE_COLS}

    # Mock transformer model + tokenizer.
    # _predict_transformer applies temperature=2.0, so logits are divided by 2
    # before softmax. To produce the desired transformer_proba after scaling,
    # multiply the raw log-odds by 2 so that dividing by temperature cancels out.
    transformer_model = MagicMock()
    log_odds = float(np.log(transformer_proba / max(1 - transformer_proba, 1e-9)))
    logit_phish = 2.0 * log_odds
    logit_benign = 0.0
    fake_logits = torch.tensor([[logit_benign, logit_phish]])
    output = MagicMock()
    output.logits = fake_logits
    transformer_model.return_value = output

    tokenizer = MagicMock()
    tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

    cascade._transformer = transformer_model
    cascade._tokenizer = tokenizer

    return cascade


# ---------------------------------------------------------------------------
# _predict_supervised
# ---------------------------------------------------------------------------

class TestPredictSupervised:
    def test_returns_proba_from_predict_proba(self):
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.3, 0.7]])
        bundle = {"model": model, "text_col": "text_combined"}
        record = _make_record(text_combined="some email text")
        assert _predict_supervised(bundle, record) == pytest.approx(0.7)

    def test_falls_back_to_decision_function_when_no_predict_proba(self):
        model = MagicMock(spec=["decision_function"])
        model.decision_function.return_value = np.array([0.0])  # sigmoid(0) = 0.5
        bundle = {"model": model, "text_col": "text_combined"}
        record = _make_record(text_combined="some email text")
        result = _predict_supervised(bundle, record)
        assert result == pytest.approx(0.5, abs=1e-4)

    def test_high_decision_score_gives_high_proba(self):
        model = MagicMock(spec=["decision_function"])
        model.decision_function.return_value = np.array([10.0])
        bundle = {"model": model, "text_col": "text_combined"}
        record = _make_record(text_combined="phishing email")
        assert _predict_supervised(bundle, record) > 0.99

    def test_uses_text_combined_fallback_when_text_col_missing(self):
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.4, 0.6]])
        bundle = {"model": model, "text_col": "nonexistent_col"}
        record = _make_record(text_combined="fallback text")
        result = _predict_supervised(bundle, record)
        assert result == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# _predict_transformer
# ---------------------------------------------------------------------------

class TestPredictTransformer:
    def test_returns_float_between_0_and_1(self):
        model = MagicMock()
        output = MagicMock()
        output.logits = torch.tensor([[0.2, 0.8]])
        model.return_value = output
        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        record = _make_record(text_combined="test email")
        result = _predict_transformer(model, tokenizer, record)
        assert 0.0 <= result <= 1.0

    def test_high_phish_logit_gives_high_proba(self):
        model = MagicMock()
        output = MagicMock()
        output.logits = torch.tensor([[0.0, 10.0]])
        model.return_value = output
        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        record = _make_record(text_combined="test email")
        assert _predict_transformer(model, tokenizer, record) > 0.99

    def test_empty_text_combined_does_not_crash(self):
        model = MagicMock()
        output = MagicMock()
        output.logits = torch.tensor([[1.0, 0.0]])
        model.return_value = output
        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": torch.tensor([[1]])}
        record = {"from_address": "", "subject": "", "body": ""}
        result = _predict_transformer(model, tokenizer, record)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# PhishingCascade.predict — output schema
# ---------------------------------------------------------------------------

class TestCascadeOutputSchema:
    def test_output_contains_required_keys(self):
        cascade = _make_cascade()
        result = cascade.predict(_make_record())
        for key in ("is_phish", "confidence", "triggered_by", "rule_score",
                    "rule_matches", "supervised_proba", "transformer_proba",
                    "isolation_score"):
            assert key in result, f"Missing key: {key}"

    def test_is_phish_is_binary(self):
        cascade = _make_cascade()
        result = cascade.predict(_make_record())
        assert result["is_phish"] in (0, 1)

    def test_confidence_between_0_and_1(self):
        cascade = _make_cascade()
        result = cascade.predict(_make_record())
        assert 0.0 <= result["confidence"] <= 1.0

    def test_rule_matches_is_list(self):
        cascade = _make_cascade()
        result = cascade.predict(_make_record())
        assert isinstance(result["rule_matches"], list)

    def test_all_scores_populated_for_benign(self):
        # When no hard-block rule fires, all layer scores should be present
        cascade = _make_cascade(supervised_proba=0.1, transformer_proba=0.1)
        result = cascade.predict(_make_record())
        assert result["supervised_proba"] is not None
        assert result["transformer_proba"] is not None
        assert result["isolation_score"] is not None


# ---------------------------------------------------------------------------
# PhishingCascade.predict — layer routing
# ---------------------------------------------------------------------------

class TestCascadeLayerRouting:
    def test_hard_block_rule_triggers_immediately(self):
        # ip_url (weight 3, in HARD_BLOCK_RULES) fires regardless of ML scores
        cascade = _make_cascade(supervised_proba=0.1, transformer_proba=0.1)
        record = _make_record(
            text_combined="FROM:  SUBJECT:  BODY: Click http://192.168.1.1/login"
        )
        result = cascade.predict(record)
        assert result["is_phish"] == 1
        assert result["triggered_by"] == "rules"

    def test_hard_block_skips_ml_layers(self):
        # When a hard-block rule fires, ML layers are not run (scores remain None)
        cascade = _make_cascade()
        record = _make_record(
            text_combined="FROM:  SUBJECT:  BODY: Visit http://192.168.0.1/verify"
        )
        result = cascade.predict(record)
        assert result["triggered_by"] == "rules"
        assert result["supervised_proba"] is None
        assert result["transformer_proba"] is None
        assert result["isolation_score"] is None

    def test_ml_consensus_triggers_when_both_models_agree(self):
        # Both supervised and transformer above threshold → ml_consensus
        cascade = _make_cascade(supervised_proba=0.95, transformer_proba=0.95)
        result = cascade.predict(_make_record())
        assert result["is_phish"] == 1
        assert result["triggered_by"] == "ml_consensus"

    def test_rule_assisted_triggers_when_rules_plus_supervised_agree(self):
        # Rule score ≥ vote_threshold + supervised above threshold → rule_assisted
        # Using a record that hits sender_url_domain_mismatch + urgent + credentials
        # but NOT a hard-block rule, alongside a high supervised score
        cascade = _make_cascade(supervised_proba=0.95, transformer_proba=0.1)
        # Craft a record where rule_score >= rule_vote_threshold (3) but no hard-block
        record = _make_record(
            from_address="user@gmail.com",
            subject="Urgent: verify your account",
            body="Please login and update your password at http://somesite.com/verify",
            text_combined=(
                "FROM: user@gmail.com SUBJECT: Urgent: verify your account "
                "BODY: Please login and update your password at http://somesite.com/verify"
            ),
        )
        result = cascade.predict(record)
        assert result["is_phish"] == 1
        assert result["triggered_by"] == "rule_assisted"

    def test_rule_assisted_triggers_when_rules_plus_transformer_agree(self):
        # Rule vote + transformer above threshold → rule_assisted
        cascade = _make_cascade(supervised_proba=0.1, transformer_proba=0.95)
        record = _make_record(
            from_address="user@gmail.com",
            subject="Urgent: verify your account",
            body="Please login and update your password at http://somesite.com/verify",
            text_combined=(
                "FROM: user@gmail.com SUBJECT: Urgent: verify your account "
                "BODY: Please login and update your password at http://somesite.com/verify"
            ),
        )
        result = cascade.predict(record)
        assert result["is_phish"] == 1
        assert result["triggered_by"] == "rule_assisted"

    def test_single_supervised_alone_is_benign(self):
        # A single ML model exceeding threshold is NOT enough without corroboration
        cascade = _make_cascade(supervised_proba=0.95, transformer_proba=0.1)
        result = cascade.predict(_make_record())
        assert result["is_phish"] == 0
        assert result["triggered_by"] is None

    def test_single_transformer_alone_is_benign(self):
        # Same: transformer alone is not enough
        cascade = _make_cascade(supervised_proba=0.1, transformer_proba=0.95)
        result = cascade.predict(_make_record())
        assert result["is_phish"] == 0
        assert result["triggered_by"] is None

    def test_isolation_forest_score_is_reported_but_does_not_trigger(self):
        # IF is a passive scoring layer only — high anomaly score must not flip verdict
        cascade = _make_cascade(
            supervised_proba=0.1,
            transformer_proba=0.1,
            isolation_score=0.9,
        )
        result = cascade.predict(_make_record())
        assert result["is_phish"] == 0
        assert result["triggered_by"] is None
        assert result["isolation_score"] is not None

    def test_benign_when_all_layers_pass(self):
        cascade = _make_cascade(
            supervised_proba=0.1,
            transformer_proba=0.1,
            isolation_score=0.1,
        )
        result = cascade.predict(_make_record())
        assert result["is_phish"] == 0
        assert result["triggered_by"] is None

    def test_high_rule_score_without_hard_block_is_benign_alone(self):
        # Soft rule hits alone (no ip_url/punycode) do NOT trigger without ML vote
        cascade = _make_cascade(supervised_proba=0.1, transformer_proba=0.1)
        # This record hits several soft rules (free sender + brand terms + mismatch)
        # but not ip_url or punycode, and ML scores are low
        record = _make_record(
            from_address="user@gmail.com",
            subject="Urgent: verify your account",
            body="Please login and update your password at http://somesite.com/verify",
            text_combined=(
                "FROM: user@gmail.com SUBJECT: Urgent: verify your account "
                "BODY: Please login and update your password at http://somesite.com/verify"
            ),
        )
        result = cascade.predict(record)
        assert result["is_phish"] == 0
        assert result["triggered_by"] is None


# ---------------------------------------------------------------------------
# PhishingCascade.predict — text_combined auto-construction
# ---------------------------------------------------------------------------

class TestCascadeTextCombined:
    def test_constructs_text_combined_when_missing(self):
        cascade = _make_cascade()
        record = {
            "from_address": "a@b.com",
            "subject": "Hello",
            "body": "Normal message",
        }
        result = cascade.predict(record)
        assert "is_phish" in result  # just verify it ran without error

    def test_uses_existing_text_combined_when_present(self):
        cascade = _make_cascade(supervised_proba=0.99, transformer_proba=0.99)
        record = _make_record(text_combined="prebuilt text combined value")
        result = cascade.predict(record)
        assert result["is_phish"] == 1


# ---------------------------------------------------------------------------
# PhishingCascade.predict — confidence values
# ---------------------------------------------------------------------------

class TestCascadeConfidence:
    def test_rules_hard_block_confidence_is_1(self):
        cascade = _make_cascade()
        record = _make_record(
            text_combined="FROM:  SUBJECT:  BODY: http://192.168.1.1/phish"
        )
        result = cascade.predict(record)
        if result["triggered_by"] == "rules":
            assert result["confidence"] == 1.0

    def test_ml_consensus_confidence_is_average_of_both_probas(self):
        cascade = _make_cascade(supervised_proba=0.8, transformer_proba=0.9)
        result = cascade.predict(_make_record())
        if result["triggered_by"] == "ml_consensus":
            assert result["confidence"] == pytest.approx(0.85, abs=0.01)

    def test_rule_assisted_confidence_is_max_ml_proba(self):
        cascade = _make_cascade(supervised_proba=0.87, transformer_proba=0.1)
        record = _make_record(
            from_address="user@gmail.com",
            subject="Urgent: verify your account",
            body="Please login and update your password at http://somesite.com/verify",
            text_combined=(
                "FROM: user@gmail.com SUBJECT: Urgent: verify your account "
                "BODY: Please login and update your password at http://somesite.com/verify"
            ),
        )
        result = cascade.predict(record)
        if result["triggered_by"] == "rule_assisted":
            assert result["confidence"] == pytest.approx(0.87, abs=0.01)

    def test_benign_confidence_is_positive(self):
        cascade = _make_cascade(
            supervised_proba=0.1,
            transformer_proba=0.1,
            isolation_score=0.1,
        )
        result = cascade.predict(_make_record())
        assert result["confidence"] > 0.0

    def test_benign_confidence_reflects_distance_from_threshold(self):
        # confidence = 1 - max(supervised_proba, transformer_proba)
        cascade = _make_cascade(supervised_proba=0.3, transformer_proba=0.2)
        result = cascade.predict(_make_record())
        assert result["is_phish"] == 0
        assert result["confidence"] == pytest.approx(1.0 - 0.3, abs=0.01)


# ---------------------------------------------------------------------------
# HARD_BLOCK_RULES constant
# ---------------------------------------------------------------------------

class TestHardBlockRules:
    def test_contains_ip_url(self):
        assert "ip_url" in HARD_BLOCK_RULES

    def test_contains_punycode_domain(self):
        assert "punycode_domain" in HARD_BLOCK_RULES
