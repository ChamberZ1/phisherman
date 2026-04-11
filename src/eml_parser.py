"""Parse a raw .eml file into fields the cascade can consume.

Extracts from_address, subject, and body text. The body is the plain-text
part of the email (HTML-stripped and whitespace-normalised) with all real
href URLs appended. Tracking pixels and fragment-only anchors are filtered out.
"""
from __future__ import annotations

import codecs
import email
import email.policy
import re
from email.header import decode_header
from html.parser import HTMLParser
from urllib.parse import parse_qs, urlparse, unquote


# URL path segments that reliably indicate tracking pixels / open-beacon URLs.
_TRACKING_PATTERNS = re.compile(
    r"/(open|track|pixel|beacon|click\.php|wf/open|trk|e/|o\.aspx)",
    re.IGNORECASE,
)

# Tags whose text content should be discarded entirely (not fed to models).
_SKIP_TAGS = {"style", "script", "head"}


def _decode_safelink(url: str) -> str:
    """Unwrap a Microsoft SafeLinks URL to its real destination."""
    try:
        parsed = urlparse(url)
        if "safelinks.protection.outlook.com" in parsed.netloc:
            params = parse_qs(parsed.query)
            if "url" in params:
                return unquote(params["url"][0])
    except Exception:
        pass
    return url


class _HtmlStripper(HTMLParser):
    """Converts HTML to plain text, skipping style/script/head content."""

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth: int = 0

    def handle_starttag(self, tag: str, _: list) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._chunks.append(data)

    def get_text(self) -> str:
        return " ".join(self._chunks)


class _HrefExtractor(HTMLParser):
    """Collects href attribute values from anchor tags."""

    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            for attr, val in attrs:
                if attr == "href" and val:
                    self.hrefs.append(val)


def _safe_charset(charset: str | None) -> str:
    """Return a Python-recognised charset, falling back to latin-1 for unknowns."""
    if not charset:
        return "utf-8"
    try:
        codecs.lookup(charset)
        return charset
    except LookupError:
        return "latin-1"


def _safe_decode(raw: bytes, charset: str | None) -> str:
    """Decode bytes to str, falling back through latin-1 if the charset fails."""
    for enc in (_safe_charset(charset), "utf-8", "latin-1"):
        try:
            return raw.decode(enc, errors="replace")
        except (LookupError, UnicodeDecodeError):
            continue
    return raw.decode("latin-1", errors="replace")


def _decode_header_value(value: str | None) -> str:
    if not value:
        return ""
    parts = decode_header(value)
    result = []
    for part, charset in parts:
        if isinstance(part, bytes):
            result.append(part.decode(charset or "utf-8", errors="replace"))
        else:
            result.append(part)
    return "".join(result).strip()


def _extract_address(from_header: str) -> str:
    """Pull the bare email address out of a From header like 'Name <addr>'."""
    match = re.search(r"<([^>]+)>", from_header)
    if match:
        return match.group(1).strip()
    return from_header.strip()


def _strip_html(html: str) -> str:
    """Strip all HTML tags and return normalised plain text."""
    stripper = _HtmlStripper()
    try:
        stripper.feed(html)
    except Exception:
        pass
    text = stripper.get_text()
    # Collapse runs of whitespace / blank lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = "\n".join(line for line in text.splitlines() if line.strip())
    return text.strip()


def _normalise_body(text: str) -> str:
    """Normalise whitespace in plain-text body."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = "\n".join(line for line in text.splitlines() if line.strip())
    return text.strip()


def _parse_auth_results(msg: email.message.Message) -> dict:
    """Parse Authentication-Results and ARC-Authentication-Results headers for
    DKIM pass status.

    Emails forwarded through services like Outlook preserve DKIM results in
    ARC-Authentication-Results (the ARC chain) rather than Authentication-Results,
    so we check both. Domain is extracted from header.i or header.d depending on
    which format the receiving server used.

    Returns dkim_pass (bool) and dkim_domain (str | None). Returns False/None on
    any parse failure or if no header is present (e.g. small/private mail servers).
    """
    try:
        headers = (msg.get_all("Authentication-Results") or []) + \
                  (msg.get_all("ARC-Authentication-Results") or [])
        for header in headers:
            try:
                if re.search(r"\bdkim\s*=\s*pass\b", header, re.IGNORECASE):
                    # Try header.i=@domain first, then header.d=domain
                    match = re.search(
                        r"header\.[id]\s*=\s*@?([\w.\-]+)", header, re.IGNORECASE
                    )
                    dkim_domain = match.group(1).lower() if match else None
                    return {"dkim_pass": True, "dkim_domain": dkim_domain}
            except Exception:
                continue
    except Exception:
        pass
    return {"dkim_pass": False, "dkim_domain": None}


def _extract_attachments(msg: email.message.Message) -> dict:
    """Return attachment metadata extracted from a MIME message.

    Returns:
        Dict with has_attachment, attachment_count, attachment_extensions,
        and attachment_filenames.
    """
    filenames: list[str] = []
    extensions: list[str] = []

    for part in msg.walk():
        filename = part.get_filename()
        if not filename:
            continue
        filename = _decode_header_value(filename)
        if not filename:
            continue
        filenames.append(filename)
        dot = filename.rfind(".")
        if dot != -1:
            extensions.append(filename[dot:].lower())

    return {
        "has_attachment": bool(filenames),
        "attachment_count": len(filenames),
        "attachment_extensions": extensions,
        "attachment_filenames": filenames,
    }


def _get_body_parts(msg: email.message.Message) -> tuple[str, str]:
    """Return (plain_text, html_text) from a potentially multipart message."""
    plain = ""
    html = ""

    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/plain" and not plain:
                raw = part.get_payload(decode=True)
                if raw:
                    plain = _safe_decode(raw, part.get_content_charset())
            elif ct == "text/html" and not html:
                raw = part.get_payload(decode=True)
                if raw:
                    html = _safe_decode(raw, part.get_content_charset())
    else:
        raw = msg.get_payload(decode=True)
        if raw:
            text = _safe_decode(raw, msg.get_content_charset())
            if msg.get_content_type() == "text/html":
                html = text
            else:
                plain = text

    return plain, html


def _extract_hrefs(html: str) -> list[str]:
    """Return deduplicated, cleaned http/https hrefs from an HTML string.

    Filters out:
    - Non-http(s) schemes (mailto:, tel:, #fragments)
    - Tracking pixel / open-beacon URLs
    """
    extractor = _HrefExtractor()
    try:
        extractor.feed(html)
    except Exception:
        pass

    seen: set[str] = set()
    result = []
    for href in extractor.hrefs:
        if not href.startswith("http"):
            continue
        href = _decode_safelink(href)
        if _TRACKING_PATTERNS.search(href):
            continue
        if href in seen:
            continue
        seen.add(href)
        result.append(href)
    return result


def parse_eml(content: bytes) -> dict:
    """Parse raw .eml bytes and return extracted fields optimised for the cascade.

    Returns:
        Dict with keys:
          - from_address: sender email address
          - subject: decoded subject line
          - body: normalised plain text with real href URLs appended
          - urls_found: number of unique URLs extracted from HTML
    """
    msg = email.message_from_bytes(content, policy=email.policy.compat32)

    from_address = _extract_address(_decode_header_value(msg.get("From", "")))
    subject = _decode_header_value(msg.get("Subject", ""))

    plain, html = _get_body_parts(msg)
    hrefs = _extract_hrefs(html) if html else []
    attachments = _extract_attachments(msg)
    auth = _parse_auth_results(msg)

    # Prefer plain text; fall back to HTML-stripped text if no plain part
    if plain:
        body = _normalise_body(plain)
    elif html:
        body = _strip_html(html)
    else:
        body = ""

    if hrefs:
        body = body + "\n\n" + "\n".join(hrefs) if body else "\n".join(hrefs)

    return {
        "from_address": from_address,
        "subject": subject,
        "body": body,
        "urls_found": len(hrefs),
        **attachments,
        **auth,
    }


__all__ = ["parse_eml"]
