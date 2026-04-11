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
    r"\b(?:https?://)?(?:www\.)?(?:bit\.ly|tinyurl\.com|t\.co|t\.ly|goo\.gl|is\.gd|ow\.ly|buff\.ly|rebrand\.ly|cutt\.ly|tiny\.cc|rb\.gy|clck\.ru|u\.to)(?:/[^\s<>\'\"]*)?",
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

# Known legitimate sending domains for major services, including common ESP
# relay domains they use (e.g. amazonses.com for Amazon SES). Used alongside
# DKIM verification — a DKIM pass from one of these domains suppresses the
# weaker verdict paths (rule_assisted, transformer_rule_corroborated) to
# reduce false positives on transactional and notification email.
TRUSTED_SENDING_DOMAINS: frozenset[str] = frozenset({
    # Amazon
    "amazon.com", "amazon.co.uk", "amazon.ca",
    "amazon.de", "amazon.fr", "amazon.co.jp",
    # PayPal (mail.paypal.com excluded — abused via PayPal account notifications)
    "paypal.com", "paypal-communication.com",
    "googlemail.com",
    # Apple
    "apple.com",
    # Microsoft (outlook.com excluded — free email provider, same as gmail.com)
    "microsoft.com", "microsoftemail.com",
    # GitHub excluded — notification abuse: phisher posts malicious links in
    # issues/comments, triggering GitHub notification emails with valid DKIM
    # Stripe
    "stripe.com",
    # Shopify
    "shopify.com", "shopifyemail.com",
    # LinkedIn
    "linkedin.com",
    # Meta
    "facebook.com", "facebookmail.com", "meta.com",
    # Netflix
    "netflix.com",
    # Spotify
    "spotify.com",
    # Dropbox
    "dropbox.com",
    # Slack
    "slack.com",
    # Zoom
    "zoom.us",
    # Intuit / QuickBooks / TurboTax
    "intuit.com", "turbotax.com", "quickbooks.com",
    # Banks & financial
    "chase.com", "bankofamerica.com", "wellsfargo.com",
    "citibank.com", "americanexpress.com", "capitalone.com",
    "discover.com", "usbank.com", "ally.com",
    # Crypto exchanges
    "coinbase.com", "kraken.com",
    # Adobe
    "adobe.com",
    # Notion
    "notion.so",
    # Zendesk excluded — shared platform anyone can use, same problem as amazonses.com
    # E-commerce & retail
    "ebay.com", "etsy.com", "walmart.com", "target.com",
    # Shipping & delivery
    "ups.com", "fedex.com", "usps.com", "dhl.com",
    # Travel
    "airbnb.com", "booking.com", "expedia.com", "united.com",
    "delta.com", "southwest.com",
    # Productivity & services
    "trello.com", "atlassian.com", "salesforce.com",
    # AI / developer platforms
    "huggingface.co",
})

# Maps a DKIM signing root domain → set of trusted From address domains.
# Used for platforms like Google where the broad domain is abusable (Google
# Forms, Workspace) but specific subdomains are not — only Google's own
# account/payment systems can send from accounts.google.com or payments.google.com.
# An attacker cannot make accounts.google.com send arbitrary content.
TRUSTED_FROM_SUBDOMAINS: dict[str, frozenset] = {
    "google.com": frozenset({
        "accounts.google.com",   # Account security alerts
        "payments.google.com",   # Google Pay receipts
    }),
}

# File extensions that are direct execution vectors — hard-blocked immediately,
# no ML agreement needed. HTML/HTM attachments are the most common phishing lure
# (credential-harvest forms masquerading as email attachments).
HARD_BLOCK_EXTENSIONS: frozenset[str] = frozenset({
    ".html", ".htm",  # Credential-harvesting form disguised as attachment
    ".js",            # JavaScript
    ".vbs",           # VBScript
    ".lnk",           # Windows shortcut (common dropper)
    ".iso", ".img",   # Disk image container for malware
    ".exe", ".bat",   # Direct executables
    ".ps1",           # PowerShell script
    ".scr",           # Screen saver / disguised executable
    ".hta",           # HTML Application
})

# Extensions that are suspicious in context (with lure language) but not
# hard-block worthy on their own.
SUSPICIOUS_EXTENSIONS: frozenset[str] = frozenset({
    ".zip", ".rar", ".7z",  # Archives used to bypass extension filters
    ".xlsm", ".docm",       # Macro-enabled Office documents
    ".pdf",                 # PDFs with embedded JS or phishing links
    ".jar",                 # Java archive
})

# Lure phrases that, alongside any attachment, strongly suggest a phishing email.
ATTACHMENT_LURE_REGEX = re.compile(
    r"\b(?:see\s+attached|find\s+attached|invoice|receipt|payment\s+(?:confirmation|receipt|details)"
    r"|statement|open\s+the\s+attached|review\s+the\s+attached|attached\s+(?:file|document))\b",
    re.IGNORECASE,
)
