import pandas as pd

from src.features import build_features


def _make_base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text_combined": [
                "FROM: support@bank.com SUBJECT: URGENT BODY: CLICK NOW! Visit https://secure-bank.com/login",
                "FROM: hr@company.com SUBJECT: Meeting BODY: See you tomorrow",
            ],
            "body": [
                "URGENT ACTION REQUIRED! CLICK NOW!",
                "See you tomorrow",
            ],
            "subject": ["URGENT", "Meeting"],
            "from_address": ["support@bank.com", "hr@company.com"],
            "source": ["set_a", "set_b"],
            "label": [1, 0],
        }
    )


def test_build_features_adds_expected_columns():
    df = _make_base_df()
    out = build_features(df)

    expected_cols = {
        "text_len",
        "body_len",
        "subject_len",
        "num_urls_in_text",
        "has_url",
        "has_urgent_terms",
        "has_action_terms",
        "has_credential_terms",
        "sender_is_free_email",
        "sender_url_domain_mismatch",
        "num_currency_symbols_text",
        "has_crypto_terms",
    }
    assert expected_cols.issubset(set(out.columns))


def test_build_features_infers_has_url_when_missing():
    df = _make_base_df().drop(columns=["label"])
    out = build_features(df)

    assert out.loc[0, "has_url"] == 1
    assert out.loc[1, "has_url"] == 0
    assert out.loc[0, "has_url_was_missing"] == 1


def test_build_features_uses_provided_has_url_when_valid():
    df = _make_base_df()
    df["has_url"] = [0, 1]
    out = build_features(df)

    assert out.loc[0, "has_url"] == 0
    assert out.loc[1, "has_url"] == 1
    assert out["has_url_was_missing"].sum() == 0


def test_uppercase_feature_is_preserved_signal():
    df = _make_base_df()
    out = build_features(df)

    assert out.loc[0, "num_uppercase_chars_body"] > out.loc[1, "num_uppercase_chars_body"]
    assert out.loc[0, "uppercase_ratio_body"] > out.loc[1, "uppercase_ratio_body"]


def test_new_url_and_html_features():
    df = pd.DataFrame(
        {
            "text_combined": [
                "FROM: x SUBJECT: y BODY: Visit bit.ly/abc and http://evil.zip now",
                "FROM: x SUBJECT: y BODY: See docs at secure-bank.com",
            ],
            "body": [
                "<a href='http://evil.zip'>Click</a> now!",
                "Visit secure-bank.com for details",
            ],
            "subject": ["Alert", "Info"],
            "from_address": ["support@bank.com", "hr@company.com"],
        }
    )

    out = build_features(df)

    assert out.loc[0, "num_short_urls"] >= 1
    assert out.loc[0, "tld_risk_flag"] == 1
    assert out.loc[0, "num_html_tags_body"] >= 1
    assert out.loc[1, "has_bare_domain_links_body"] == 1
    assert out.loc[0, "char_entropy_text"] > 0


def test_typosquat_features_with_custom_brand_list():
    df = pd.DataFrame(
        {
            "text_combined": [
                "FROM: support@paypa1.com SUBJECT: Alert BODY: Visit http://paypa1.com/login",
                "FROM: help@paypal.com SUBJECT: Alert BODY: Visit http://paypal.com/login",
            ],
            "body": ["Click now", "Click now"],
            "subject": ["Alert", "Alert"],
            "from_address": ["support@paypa1.com", "help@paypal.com"],
        }
    )

    out = build_features(df, brand_domains=["paypal.com"])

    assert out.loc[0, "sender_typosquat_flag"] == 1
    assert out.loc[0, "url_typosquat_flag"] == 1
    assert out.loc[1, "sender_typosquat_flag"] == 0
    assert out.loc[1, "url_typosquat_flag"] == 0
