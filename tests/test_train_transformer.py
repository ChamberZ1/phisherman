import numpy as np
import pandas as pd
import pytest

pytest.importorskip("transformers")
pytest.importorskip("torch")
import torch
from transformers import AutoTokenizer, EvalPrediction

from src.layers.transformer.train_transformer import (
    TextDataset,
    _compute_metrics,
    _load_split,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TOKENIZER_NAME = "distilbert-base-uncased"


@pytest.fixture(scope="module")
def tokenizer():
    """Download once per test session; skip if no network."""
    try:
        return AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    except Exception:
        pytest.skip("Could not load tokenizer (no network?)")


def _make_csv(tmp_path, rows: list[dict], filename: str = "data.csv") -> str:
    path = tmp_path / filename
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# TextDataset
# ---------------------------------------------------------------------------

class TestTextDataset:
    def test_len(self, tokenizer):
        texts = ["hello phish", "safe email"]
        labels = [1, 0]
        ds = TextDataset(texts, labels, tokenizer, max_length=32)
        assert len(ds) == 2

    def test_getitem_keys(self, tokenizer):
        ds = TextDataset(["click here now"], [1], tokenizer, max_length=32)
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item

    def test_getitem_tensors(self, tokenizer):
        ds = TextDataset(["hello"], [0], tokenizer, max_length=32)
        item = ds[0]
        for v in item.values():
            assert isinstance(v, torch.Tensor)

    def test_label_dtype(self, tokenizer):
        ds = TextDataset(["test"], [1], tokenizer, max_length=32)
        assert ds[0]["labels"].dtype == torch.long

    def test_max_length_respected(self, tokenizer):
        long_text = "word " * 300
        ds = TextDataset([long_text], [0], tokenizer, max_length=16)
        assert ds[0]["input_ids"].shape[0] <= 16


# ---------------------------------------------------------------------------
# _load_split
# ---------------------------------------------------------------------------

class TestLoadSplit:
    def test_basic_load(self, tmp_path):
        path = _make_csv(tmp_path, [
            {"text": "hello", "label": 0},
            {"text": "click here", "label": 1},
        ])
        texts, labels = _load_split(path, "text", "label")
        assert texts == ["hello", "click here"]
        assert labels == [0, 1]

    def test_fillna_on_text(self, tmp_path):
        path = _make_csv(tmp_path, [
            {"text": None, "label": 0},
            {"text": "real text", "label": 1},
        ])
        texts, labels = _load_split(path, "text", "label")
        assert texts[0] == ""

    def test_missing_text_col_raises(self, tmp_path):
        path = _make_csv(tmp_path, [{"body": "hi", "label": 0}])
        with pytest.raises(ValueError, match="Missing text column"):
            _load_split(path, "text", "label")

    def test_missing_target_col_raises(self, tmp_path):
        path = _make_csv(tmp_path, [{"text": "hi", "target": 0}])
        with pytest.raises(ValueError, match="Missing target column"):
            _load_split(path, "text", "label")


# ---------------------------------------------------------------------------
# _compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def _pred(self, logits, labels):
        return EvalPrediction(predictions=np.array(logits), label_ids=np.array(labels))

    def test_perfect_predictions(self):
        # logits strongly predict correct class
        logits = [[10.0, -10.0], [-10.0, 10.0], [10.0, -10.0], [-10.0, 10.0]]
        labels = [0, 1, 0, 1]
        m = _compute_metrics(self._pred(logits, labels))
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)
        assert m["roc_auc"] == pytest.approx(1.0)

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning")
    def test_all_wrong(self):
        logits = [[10.0, -10.0], [10.0, -10.0]]  # always predicts 0
        labels = [1, 1]
        m = _compute_metrics(self._pred(logits, labels))
        assert m["accuracy"] == pytest.approx(0.0)
        assert m["f1"] == pytest.approx(0.0)

    def test_keys_present(self):
        logits = [[-1.0, 1.0], [1.0, -1.0]]
        labels = [1, 0]
        m = _compute_metrics(self._pred(logits, labels))
        assert set(m.keys()) == {"accuracy", "precision", "recall", "f1", "roc_auc"}

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning")
    def test_auc_nan_on_single_class(self):
        # roc_auc_score requires both classes; should fall back to nan
        logits = [[1.0, -1.0], [1.0, -1.0]]
        labels = [0, 0]
        m = _compute_metrics(self._pred(logits, labels))
        assert np.isnan(m["roc_auc"])


# ---------------------------------------------------------------------------
# Integration: run_training (slow, requires network + transformers)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_run_training_end_to_end(tmp_path):
    """Full training loop on a tiny synthetic dataset. Requires network access."""
    import argparse
    from src.layers.transformer.train_transformer import run_training
    from transformers import AutoTokenizer

    processed = tmp_path / "processed"
    processed.mkdir()
    output = tmp_path / "model"

    rows = [{"text_combined": f"sample text {i}", "label": i % 2} for i in range(20)]
    for split in ("train", "val", "test"):
        pd.DataFrame(rows).to_csv(processed / f"{split}.csv", index=False)

    try:
        AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    except Exception:
        pytest.skip("Could not load tokenizer (no network?)")

    args = argparse.Namespace(
        processed_dir=str(processed),
        train_file="train.csv",
        val_file="val.csv",
        test_file="test.csv",
        text_col="text_combined",
        target_col="label",
        model_name=TOKENIZER_NAME,
        output_dir=str(output),
        max_length=32,
        batch_size=4,
        epochs=1,
        learning_rate=2e-5,
        weight_decay=0.01,
        seed=42,
    )

    metrics = run_training(args)

    assert "val" in metrics and "test" in metrics
    assert (output / "config.json").exists()
    assert (output / "metrics.json").exists()
