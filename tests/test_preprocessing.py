import pandas as pd
import io
import pytest
from src import preprocessing

# these tests require scikit-learn for stratified splitting; skip if absent
pytest.importorskip("sklearn")


def _make_df():
    # build a medium-size DataFrame with a mix of issues
    # (duplicates, whitespace noise, and one invalid row)
    data = {
        "source": [
            "kaggle", "kaggle", "kaggle", "kaggle",
            "enron", "enron", "enron", "enron",
            "nazario", "nazario", "nazario", "nazario",
            "kaggle", "enron", "nazario",
        ],
        "from_address": [
            "support@paypal.com", "support@paypal.com", "security@chase.com", "alerts@apple.com",
            "john@company.com", "it@corp.com", "noreply@microsoft.com", "billing@adobe.com",
            "alerts@bank-secure.com", "support@amazon.com", "admin@dropbox.com", "notice@coinbase.com",
            "support@paypal.com", "unknown@example.com", "noreply@service.com",
        ],
        "subject": [
            "Verify your account", "Verify   your   account", "Security Alert", "Update payment method",
            "Meeting tomorrow", "Project timeline", "Password reset", "Invoice available",
            "URGENT: Account Locked", "Order confirmation", "Shared document", "Account verification",
            "Verify your account", "Lunch plans", None,
        ],
        "body": [
            "Click here to verify your account immediately.",
            "Click   here   to   verify   your   account   immediately.",
            "Your account is at risk. Confirm now.",
            "Please update your billing details to avoid suspension.",
            "Let's meet at 3pm.",
            "Attached is the latest timeline for review.",
            "Use this link to reset your password.",
            "Your monthly invoice is ready.",
            "Please login at http://secure-bank.com to unlock.",
            "Your order has shipped and will arrive tomorrow.",
            "A document has been shared with you.",
            "Confirm your identity to keep access.",
            "Click here to verify your account immediately.",
            "Are we still on for lunch today?",
            None,
        ],
        "text_combined": [
            "support@paypal.com Verify your account Click here to verify your account immediately.",
            "support@paypal.com Verify   your   account Click   here   to   verify   your   account   immediately.",
            "security@chase.com Security Alert Your account is at risk. Confirm now.",
            "alerts@apple.com Update payment method Please update your billing details to avoid suspension.",
            "john@company.com Meeting tomorrow Let's meet at 3pm.",
            "it@corp.com Project timeline Attached is the latest timeline for review.",
            "noreply@microsoft.com Password reset Use this link to reset your password.",
            "billing@adobe.com Invoice available Your monthly invoice is ready.",
            "alerts@bank-secure.com URGENT: Account Locked Please login at http://secure-bank.com to unlock.",
            "support@amazon.com Order confirmation Your order has shipped and will arrive tomorrow.",
            "admin@dropbox.com Shared document A document has been shared with you.",
            "notice@coinbase.com Account verification Confirm your identity to keep access.",
            "support@paypal.com Verify your account Click here to verify your account immediately.",
            "unknown@example.com Lunch plans Are we still on for lunch today?",
            None,
        ],
        "label": [
            1, 1, 1, 1,
            0, 0, 0, 0,
            1, 0, 0, 1,
            1, 0, None,
        ],
        "has_url": [
            0, 0, 0, 0,
            0, 0, 1, 0,
            1, 0, 0, 0,
            0, 0, 0,
        ],
    }
    return pd.DataFrame(data)

def test_sanity_check(capsys):
    df = pd.DataFrame({"text_combined": ["x", "y"], "label": [0, 1]})
    stats = preprocessing.sanity_check(df, before_count=5)
    assert stats["total"] == 2
    assert stats["label_counts"][0] == 1
    assert stats["removed"] == 3
    captured = capsys.readouterr().out
    assert "Total rows" in captured
    assert "Rows removed" in captured


def test_stratified_split():
    df = pd.DataFrame({
        "text_combined": [f"m{i}" for i in range(100)],
        "label": [0] * 80 + [1] * 20,
    })
    train, val, test = preprocessing.stratified_split(df, val_size=0.1, test_size=0.1, random_state=1)
    # proportions roughly
    assert abs(len(val) - 10) <= 1
    assert abs(len(test) - 10) <= 1
    # label ratios preserved
    assert train["label"].mean() == pytest.approx(df["label"].mean(), rel=0.1)


def test_stratified_split_indices_and_proportions():
    df = pd.DataFrame({
        "text_combined": [f"m{i}" for i in range(100)],
        "label": [0] * 80 + [1] * 20,
    })
    train, val, test = preprocessing.stratified_split(df, val_size=0.1, test_size=0.1, random_state=7)
    # indices reset
    assert train.index[0] == 0 and val.index[0] == 0 and test.index[0] == 0
    # sizes approximate
    total = len(df)
    assert abs(len(val) - int(total * 0.1)) <= 2
    assert abs(len(test) - int(total * 0.1)) <= 2
    # label proportions preserved within tolerance
    orig_ratio = df["label"].mean()
    assert abs(train["label"].mean() - orig_ratio) < 0.05


def test_invalid_sizes_raise():
    df = pd.DataFrame({"text_combined": ["a", "b"], "label": [0, 1]})
    with pytest.raises(ValueError):
        preprocessing.stratified_split(df, val_size=0.0, test_size=0.1)
    with pytest.raises(ValueError):
        preprocessing.stratified_split(df, val_size=0.1, test_size=0.0)
    with pytest.raises(ValueError):
        preprocessing.stratified_split(df, val_size=0.6, test_size=0.5)


def test_small_class_failure():
    # class 1 has only one sample; stratification should fail with clear message
    df = pd.DataFrame({
        "text_combined": [f"m{i}" for i in range(11)],
        "label": [0] * 10 + [1],
    })
    with pytest.raises(ValueError) as exc:
        preprocessing.stratified_split(df, val_size=0.1, test_size=0.1)
    assert "Stratified split failed" in str(exc.value)


def test_preprocess_and_split(tmp_path):
    df = _make_df()
    train, val, test = preprocessing.preprocess_and_split(
        df, val_size=0.2, test_size=0.2, random_state=42,
        save_paths={
            "train": str(tmp_path / "train.csv"),
            "val": str(tmp_path / "val.csv"),
            "test": str(tmp_path / "test.csv"),
        },
    )
    # no duplicate text_combined values across the full dataset
    all_texts = pd.concat([train, val, test])["text_combined"]
    assert all_texts.duplicated().sum() == 0
    # saved files exist
    assert (tmp_path / "train.csv").exists()
    assert (tmp_path / "val.csv").exists()
    assert (tmp_path / "test.csv").exists()

