from dataclasses import dataclass
from typing import Callable, Iterable
from urllib.parse import urlparse

from src.email_detection_constants import (
    URL_REGEX, SHORTENER_REGEX, URGENT_REGEX, ACTION_REGEX,
    CREDENTIAL_REGEX, CRYPTO_REGEX, IP_DOMAIN_REGEX,
    FREE_EMAIL_DOMAINS, KNOWN_BRAND_DOMAINS, RISKY_TLDS, PUBLIC_SUFFIX_2LDS,
    HARD_BLOCK_EXTENSIONS, SUSPICIOUS_EXTENSIONS, ATTACHMENT_LURE_REGEX,
)


def _to_text(value: object) -> str:
    return "" if value is None else str(value)


def _extract_urls(text: str) -> list[str]:
    return [m.group(0) for m in URL_REGEX.finditer(text or "")]


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


def _has_risky_tld(domains: Iterable[str]) -> bool:
    for domain in domains:
        parts = [p for p in (domain or "").split(".") if p]
        if parts and parts[-1].lower() in RISKY_TLDS:
            return True
    return False


def _has_punycode_or_non_ascii(domain: str) -> bool:
    d = domain or ""
    return ("xn--" in d) or any(ord(ch) > 127 for ch in d)


def _brand_terms() -> set[str]:
    terms = set()
    for d in KNOWN_BRAND_DOMAINS:
        root = _root_domain(d)
        if root:
            terms.add(root.split(".")[0])
    return terms


BRAND_TERMS = _brand_terms()


@dataclass(frozen=True)
class Rule:
    rule_id: str
    description: str
    weight: int
    predicate: Callable[[dict], bool]


def _rule_free_sender_brand_text(record: dict) -> bool:
    from_addr = _to_text(record.get("from_address", "")).lower()
    sender_domain = from_addr.split("@")[-1] if "@" in from_addr else ""
    if sender_domain not in FREE_EMAIL_DOMAINS:
        return False
    text = _to_text(record.get("text_combined", "")).lower()
    return any(term in text for term in BRAND_TERMS)


def _rule_url_domain_mismatch(record: dict) -> bool:
    from_addr = _to_text(record.get("from_address", "")).lower()
    sender_domain = from_addr.split("@")[-1] if "@" in from_addr else ""
    sender_root = _root_domain(sender_domain)
    urls = _extract_urls(_to_text(record.get("text_combined", "")))
    if not urls or not sender_root:
        return False
    first_domain = _url_netloc(urls[0])
    url_root = _root_domain(first_domain)
    return url_root and sender_root != url_root


def _rule_ip_url(record: dict) -> bool:
    urls = _extract_urls(_to_text(record.get("text_combined", "")))
    return any(IP_DOMAIN_REGEX.match(_url_netloc(u) or "") for u in urls)


def _rule_risky_tld(record: dict) -> bool:
    urls = _extract_urls(_to_text(record.get("text_combined", "")))
    domains = [_url_netloc(u) for u in urls]
    return _has_risky_tld(domains)


def _rule_shortener(record: dict) -> bool:
    text = _to_text(record.get("text_combined", ""))
    return bool(SHORTENER_REGEX.search(text))


def _rule_urgent_action(record: dict) -> bool:
    text = _to_text(record.get("text_combined", ""))
    return bool(URGENT_REGEX.search(text) or ACTION_REGEX.search(text))


def _rule_credentials_and_url(record: dict) -> bool:
    text = _to_text(record.get("text_combined", ""))
    urls = _extract_urls(text)
    return bool(urls) and bool(CREDENTIAL_REGEX.search(text))


def _rule_crypto_and_url(record: dict) -> bool:
    text = _to_text(record.get("text_combined", ""))
    urls = _extract_urls(text)
    return bool(urls) and bool(CRYPTO_REGEX.search(text))


def _rule_punycode_domain(record: dict) -> bool:
    text = _to_text(record.get("text_combined", ""))
    urls = _extract_urls(text)
    return any(_has_punycode_or_non_ascii(_url_netloc(u)) for u in urls)


def _rule_malicious_attachment_ext(record: dict) -> bool:
    exts = record.get("attachment_extensions") or []
    return bool(set(exts) & HARD_BLOCK_EXTENSIONS)


def _rule_suspicious_attachment_lure(record: dict) -> bool:
    """Suspicious (non-hard-block) attachment extension combined with lure language."""
    exts = set(record.get("attachment_extensions") or [])
    if not exts:
        return False
    has_suspicious = bool(exts & (SUSPICIOUS_EXTENSIONS | HARD_BLOCK_EXTENSIONS))
    if not has_suspicious:
        return False
    text = _to_text(record.get("text_combined", ""))
    return bool(ATTACHMENT_LURE_REGEX.search(text))


def _rule_subject_missing_with_url(record: dict) -> bool:
    subject = _to_text(record.get("subject", "")).strip()
    text = _to_text(record.get("text_combined", ""))
    urls = _extract_urls(text)
    return (subject == "") and bool(urls)


def _rule_many_urls(record: dict) -> bool:
    text = _to_text(record.get("text_combined", ""))
    urls = set(_extract_urls(text))
    return len(urls) >= 3


BASELINE_RULES: list[Rule] = [
    Rule(
        rule_id="malicious_attachment_ext",
        description="Attachment uses a directly-executable or credential-harvesting extension",
        weight=3,
        predicate=_rule_malicious_attachment_ext,
    ),
    Rule(
        rule_id="suspicious_attachment_lure",
        description="Suspicious attachment extension paired with invoice/payment lure language",
        weight=2,
        predicate=_rule_suspicious_attachment_lure,
    ),
    Rule(
        rule_id="free_sender_brand_text",
        description="Free email sender with brand terms in message",
        weight=2,
        predicate=_rule_free_sender_brand_text,
    ),
    Rule(
        rule_id="sender_url_domain_mismatch",
        description="Sender domain does not match first URL domain",
        weight=2,
        predicate=_rule_url_domain_mismatch,
    ),
    Rule(
        rule_id="ip_url",
        description="URL uses a raw IP address",
        weight=3,
        predicate=_rule_ip_url,
    ),
    Rule(
        rule_id="risky_tld",
        description="URL uses a risky TLD",
        weight=2,
        predicate=_rule_risky_tld,
    ),
    Rule(
        rule_id="shortener",
        description="Message contains a URL shortener",
        weight=2,
        predicate=_rule_shortener,
    ),
    Rule(
        rule_id="urgent_or_action",
        description="Urgent or action language present",
        weight=1,
        predicate=_rule_urgent_action,
    ),
    Rule(
        rule_id="credentials_with_url",
        description="Credential terms with URL present",
        weight=2,
        predicate=_rule_credentials_and_url,
    ),
    Rule(
        rule_id="crypto_with_url",
        description="Crypto terms with URL present",
        weight=2,
        predicate=_rule_crypto_and_url,
    ),
    Rule(
        rule_id="punycode_domain",
        description="URL domain uses punycode or non-ASCII characters",
        weight=3,
        predicate=_rule_punycode_domain,
    ),
    Rule(
        rule_id="subject_missing_with_url",
        description="Missing subject with URL present",
        weight=1,
        predicate=_rule_subject_missing_with_url,
    ),
    Rule(
        rule_id="many_urls",
        description="Message contains multiple URLs",
        weight=1,
        predicate=_rule_many_urls,
    ),
]


def evaluate_baseline_rules(record: dict) -> dict:
    """Return matched baseline rules and total score for a record."""
    matches: list[dict] = []
    score = 0
    for rule in BASELINE_RULES:
        if rule.predicate(record):
            matches.append({
                "rule_id": rule.rule_id,
                "description": rule.description,
                "weight": rule.weight,
            })
            score += rule.weight
    return {"score": score, "matches": matches}



__all__ = [
    "BASELINE_RULES",
    "evaluate_baseline_rules",
]
