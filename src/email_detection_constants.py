"""Shared regex patterns and domain constants used across feature engineering
and rule-based detection layers."""

import re

URL_REGEX = re.compile(r"https?://[^\s<>'\"]+", re.IGNORECASE)
BARE_DOMAIN_REGEX = re.compile(r"(?<!@)\b(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)+\b", re.IGNORECASE)
HTML_TAG_REGEX = re.compile(r"<[^>]+>", re.IGNORECASE)
URGENT_REGEX = re.compile(r"\b(?:urgent|immediate|action required|verify|suspend|locked|alert)\b", re.IGNORECASE)
ACTION_REGEX = re.compile(r"\b(?:click|login|confirm|update|reset|submit|open)\b", re.IGNORECASE)
CREDENTIAL_REGEX = re.compile(r"\b(?:password|credential|ssn|account|bank|billing|payment|wallet)\b", re.IGNORECASE)
CURRENCY_REGEX = re.compile(r"[$\u20AC\u00A3\u00A5\u20B9\u20A9\u20BD]")
CRYPTO_REGEX = re.compile(r"\b(?:seed phrase|btc|bitcoin|eth|ethereum|usdt|crypto)\b", re.IGNORECASE)
IP_DOMAIN_REGEX = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
SHORTENER_REGEX = re.compile(
    r"\b(?:https?://)?(?:www\.)?(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl|is\.gd|ow\.ly|buff\.ly|rebrand\.ly|cutt\.ly|tiny\.cc)(?:/[^\s<>\'\"]*)?",
    re.IGNORECASE,
)

FREE_EMAIL_DOMAINS = {
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "hotmail.com",
    "aol.com",
    "icloud.com",
    "proton.me",
    "protonmail.com",
}

KNOWN_BRAND_DOMAINS = [
    "paypal.com",
    "microsoft.com",
    "apple.com",
    "amazon.com",
    "google.com",
    "bankofamerica.com",
    "chase.com",
    "wellsfargo.com",
    "citibank.com",
    "coinbase.com",
]

RISKY_TLDS = {
    "zip",
    "review",
    "work",
    "click",
    "gq",
    "tk",
    "ml",
    "cf",
    "ga",
}

PUBLIC_SUFFIX_2LDS = {
    "co.uk",
    "org.uk",
    "gov.uk",
    "ac.uk",
    "com.au",
    "net.au",
    "org.au",
    "co.nz",
}
