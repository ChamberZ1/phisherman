import math
import re
from collections import Counter
from urllib.parse import urlparse

import pandas as pd

from src.typosquat_risk import best_brand_match, typosquat_flag
from src.email_detection_constants import (
    URL_REGEX, BARE_DOMAIN_REGEX, HTML_TAG_REGEX,
    URGENT_REGEX, ACTION_REGEX, CREDENTIAL_REGEX,
    CURRENCY_REGEX, CRYPTO_REGEX, IP_DOMAIN_REGEX,
    SHORTENER_REGEX, FREE_EMAIL_DOMAINS, KNOWN_BRAND_DOMAINS,
    RISKY_TLDS, PUBLIC_SUFFIX_2LDS,
)


def _to_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str)


def _extract_first_url(text: str) -> str:
    match = URL_REGEX.search(text)
    return match.group(0) if match else ""



def _url_netloc(url: str) -> str:
    if not url:
        return ""
    try:
        netloc = urlparse(url).netloc.lower()
    except ValueError:
        return ""
    return netloc.split(":")[0]


def _root_domain(domain: str) -> str:
    d = (domain or "").strip().lower()
    if not d:
        return ""
    parts = [p for p in d.split(".") if p]
    if len(parts) < 2:
        return d

    last_two = ".".join(parts[-2:])
    if len(parts) >= 3 and last_two in PUBLIC_SUFFIX_2LDS:
        return ".".join(parts[-3:])
    return last_two


def _char_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def _count_short_urls(text: str) -> int:
    # Count shortener links with or without explicit scheme.
    return len(SHORTENER_REGEX.findall(text))


def _count_bare_domains(text: str) -> int:
    no_urls = URL_REGEX.sub(" ", text)
    return len(BARE_DOMAIN_REGEX.findall(no_urls))

def _has_risky_tld(text: str) -> int:
    for m in URL_REGEX.finditer(text):
        domain = _url_netloc(m.group(0))
        if not domain:
            continue
        parts = [p for p in domain.split(".") if p]
        if parts and parts[-1].lower() in RISKY_TLDS:
            return 1
    return 0

# currently just guardrail that doesn't do anything because the data structure is normalized in data_loader.py
def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col not in df.columns:
            df[col] = ""


def _add_text_features(out: pd.DataFrame, text: pd.Series, body: pd.Series, subject: pd.Series) -> None:
    out["text_len"] = text.str.len()
    out["body_len"] = body.str.len()
    out["subject_len"] = subject.str.len()
    out["subject_missing"] = (subject.str.strip() == "").astype(int)
    out["num_words_text"] = text.str.count(r"\S+")
    out["num_digits_text"] = text.str.count(r"\d")
    out["num_currency_symbols_text"] = text.str.count(CURRENCY_REGEX)
    out["has_crypto_terms"] = text.str.contains(CRYPTO_REGEX, na=False).astype(int)
    out["num_crypto_term_hits"] = text.str.count(CRYPTO_REGEX)
    unique_text = text.drop_duplicates()
    entropy_map = {t: _char_entropy(t) for t in unique_text}
    out["char_entropy_text"] = text.map(entropy_map)
    out["num_exclamations_body"] = body.str.count("!")
    out["num_question_marks_body"] = body.str.count(r"\?")
    out["num_uppercase_chars_body"] = body.str.count(r"[A-Z]")
    out["uppercase_ratio_body"] = (
        out["num_uppercase_chars_body"] / out["body_len"].replace(0, 1)
    )

    out["body_to_subject_len_ratio"] = out["body_len"] / out["subject_len"].replace(0, 1)

    out["num_html_tags_body"] = body.str.count(HTML_TAG_REGEX)
    out["has_html_tags_body"] = (out["num_html_tags_body"] > 0).astype(int)

    out["num_bare_domain_links_body"] = body.apply(_count_bare_domains)
    out["has_bare_domain_links_body"] = (out["num_bare_domain_links_body"] > 0).astype(int)

    out["has_urgent_terms"] = body.str.contains(URGENT_REGEX, na=False).astype(int)
    out["has_action_terms"] = body.str.contains(ACTION_REGEX, na=False).astype(int)
    out["has_credential_terms"] = text.str.contains(CREDENTIAL_REGEX, na=False).astype(int)
    out["num_suspicious_keyword_hits"] = (
        body.str.count(URGENT_REGEX)
        + body.str.count(ACTION_REGEX)
        + text.str.count(CREDENTIAL_REGEX)
        + text.str.count(CRYPTO_REGEX)
    )


def _add_url_features(out: pd.DataFrame, text: pd.Series, has_url_col: str) -> None:
    out["num_urls_in_text"] = text.str.count(URL_REGEX)
    out["num_short_urls"] = text.apply(_count_short_urls)
    out["first_url"] = text.apply(_extract_first_url)
    out["first_url_has_https"] = out["first_url"].str.startswith("https://").astype(int)
    out["first_url_len"] = out["first_url"].str.len()

    netloc = out["first_url"].apply(_url_netloc)
    out["first_url_domain_len"] = netloc.str.len()
    out["first_url_domain_num_dots"] = netloc.str.count(r"\.")
    out["first_url_domain_num_hyphens"] = netloc.str.count("-")
    out["first_url_domain_is_ip"] = netloc.str.match(IP_DOMAIN_REGEX, na=False).astype(int)

    inferred_has_url = (out["num_urls_in_text"] > 0).astype(int)
    if has_url_col in out.columns:
        provided = pd.to_numeric(out[has_url_col], errors="coerce")
        provided = provided.where(provided.isin([0, 1]), pd.NA)
        out["has_url"] = provided.where(provided.notna(), inferred_has_url).astype(int)
        out["has_url_was_missing"] = provided.isna().astype(int)
    else:
        out["has_url"] = inferred_has_url
        out["has_url_was_missing"] = 1

    out["first_url_domain"] = netloc
    out["tld_risk_flag"] = text.apply(_has_risky_tld).astype(int)


def _add_sender_features(out: pd.DataFrame, from_addr: pd.Series) -> None:
    sender_domain = from_addr.str.extract(r"@([^\s@]+)$", expand=False).fillna("")
    sender_local = from_addr.str.extract(r"^([^@\s]+)@", expand=False).fillna("")

    out["sender_missing"] = (from_addr == "").astype(int)
    out["sender_domain_len"] = sender_domain.str.len()
    out["sender_local_len"] = sender_local.str.len()
    out["sender_has_digits"] = from_addr.str.contains(r"\d", regex=True, na=False).astype(int)
    out["sender_is_free_email"] = sender_domain.isin(FREE_EMAIL_DOMAINS).astype(int)

    url_root = out["first_url_domain"].apply(_root_domain)
    sender_root = sender_domain.apply(_root_domain)
    out["sender_url_domain_mismatch"] = (
        (out["has_url"] == 1)
        & (sender_root != "")
        & (url_root != "")
        & (sender_root != url_root)
    ).astype(int)


def _add_typosquat_features(
    out: pd.DataFrame,
    from_addr: pd.Series,
    brand_domains: list[str],
    similarity_threshold: float,
    max_edit_distance: int,
) -> None:
    sender_domain = from_addr.str.extract(r"@([^\s@]+)$", expand=False).fillna("")
    url_domain = _to_text(out.get("first_url_domain", pd.Series([""] * len(out), index=out.index)))

    unique_sender = sender_domain.drop_duplicates()
    sender_map = {d: best_brand_match(d, brand_domains) for d in unique_sender}
    sender_metrics = sender_domain.map(sender_map)
    out["sender_brand_similarity"] = sender_metrics.map(lambda m: m["similarity"])
    out["sender_brand_edit_distance"] = sender_metrics.map(lambda m: m["edit_distance"])
    out["sender_domain_punycode"] = sender_metrics.map(lambda m: m["is_punycode"])
    out["sender_domain_non_ascii"] = sender_metrics.map(lambda m: m["has_non_ascii"])
    out["sender_typosquat_flag"] = sender_metrics.map(
        lambda m: typosquat_flag(m, similarity_threshold, max_edit_distance)
    )

    unique_url = url_domain.drop_duplicates()
    url_map = {d: best_brand_match(d, brand_domains) for d in unique_url}
    url_metrics = url_domain.map(url_map)
    out["url_brand_similarity"] = url_metrics.map(lambda m: m["similarity"])
    out["url_brand_edit_distance"] = url_metrics.map(lambda m: m["edit_distance"])
    out["url_domain_punycode"] = url_metrics.map(lambda m: m["is_punycode"])
    out["url_domain_non_ascii"] = url_metrics.map(lambda m: m["has_non_ascii"])
    out["url_typosquat_flag"] = (
        (out["has_url"] == 1)
        & url_metrics.map(lambda m: typosquat_flag(m, similarity_threshold, max_edit_distance) == 1)
    ).astype(int)


def _add_metadata_features(out: pd.DataFrame) -> None:
    if "source" in out.columns:
        source = _to_text(out["source"]).str.strip().str.lower()
        out["source_missing"] = (source == "").astype(int)
    else:
        out["source_missing"] = 1


def build_features(
    df: pd.DataFrame,
    text_col: str = "text_combined",
    body_col: str = "body",
    subject_col: str = "subject",
    from_col: str = "from_address",
    has_url_col: str = "has_url",
    brand_domains: list[str] | None = None,
    typosquat_similarity_threshold: float = 0.8,
    typosquat_max_edit_distance: int = 2,
    copy: bool = True,
) -> pd.DataFrame:
    """Build phishing-oriented baseline features from normalized email dataframe."""

    out = df.copy() if copy else df
    _ensure_columns(out, [text_col, body_col, subject_col, from_col])  # not necessary currently but is here in case upstream data structure changes.

    text = _to_text(out[text_col])
    body = _to_text(out[body_col])
    subject = _to_text(out[subject_col])
    from_addr = _to_text(out[from_col]).str.strip().str.lower()

    _add_text_features(out, text=text, body=body, subject=subject)
    _add_url_features(out, text=text, has_url_col=has_url_col)
    _add_sender_features(out, from_addr=from_addr)
    out["from_subject_missing"] = ((out["sender_missing"] == 1) & (out["subject_missing"] == 1)).astype(int)
    brands = brand_domains if brand_domains is not None else KNOWN_BRAND_DOMAINS
    _add_typosquat_features(
        out,
        from_addr=from_addr,
        brand_domains=brands,
        similarity_threshold=typosquat_similarity_threshold,
        max_edit_distance=typosquat_max_edit_distance,
    )
    _add_metadata_features(out)

    out = out.drop(columns=["first_url", "first_url_domain"], errors="ignore")
    return out


__all__ = ["build_features"]
