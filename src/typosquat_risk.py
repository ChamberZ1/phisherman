import re
from difflib import SequenceMatcher
from typing import Iterable


DOMAIN_RE = re.compile(r"^(?:https?://)?(?:www\.)?", re.IGNORECASE)


def normalize_domain(value: str) -> str:
    """Normalize domain-like strings for comparison."""
    text = (value or "").strip().lower()
    text = DOMAIN_RE.sub("", text)
    text = text.split("/")[0].split(":")[0]
    return text


def registrable_domain(domain: str) -> str:
    """Naive registrable domain (last two labels)."""
    d = normalize_domain(domain)
    if not d or "." not in d:
        return d
    parts = [p for p in d.split(".") if p]
    if len(parts) < 2:
        return d
    return ".".join(parts[-2:])


def domain_label(domain: str) -> str:
    """Extract the second-level label from a domain (naive)."""
    rd = registrable_domain(domain)
    if not rd:
        return ""
    return rd.split(".")[0]


def edit_distance(a: str, b: str) -> int:
    """Simple Levenshtein distance implementation."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + cost,
            ))
        prev = cur
    return prev[-1]


def _sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def best_brand_match(domain: str, brand_domains: Iterable[str]) -> dict:
    """Return best similarity/edit-distance metrics against known brand domains."""
    d = normalize_domain(domain)
    if not d:
        return {
            "best_brand": "",
            "similarity": 0.0,
            "edit_distance": 999,
            "is_exact_match": 0,
            "is_punycode": 0,
            "has_non_ascii": 0,
        }

    brands = [normalize_domain(b) for b in brand_domains if normalize_domain(b)]
    if not brands:
        return {
            "best_brand": "",
            "similarity": 0.0,
            "edit_distance": 999,
            "is_exact_match": 0,
            "is_punycode": int("xn--" in d),
            "has_non_ascii": int(any(ord(ch) > 127 for ch in d)),
        }

    d_label = domain_label(d)
    best_brand = ""
    best_sim = 0.0
    best_dist = 999
    exact = 0

    for b in brands:
        b_label = domain_label(b)
        sim = max(_sim(d, b), _sim(d_label, b_label))
        dist = edit_distance(d_label, b_label)
        if d == b:
            exact = 1
        if sim > best_sim or (sim == best_sim and dist < best_dist):
            best_sim = sim
            best_dist = dist
            best_brand = b

    return {
        "best_brand": best_brand,
        "similarity": float(best_sim),
        "edit_distance": int(best_dist),
        "is_exact_match": int(exact),
        "is_punycode": int("xn--" in d),
        "has_non_ascii": int(any(ord(ch) > 127 for ch in d)),
    }


def typosquat_flag(metrics: dict, similarity_threshold: float = 0.8, max_edit_distance: int = 2) -> int:
    """Binary heuristic: likely typo-squat lookalike of a known brand domain."""
    if not metrics or metrics.get("is_exact_match", 0) == 1:
        return 0
    return int(
        metrics.get("similarity", 0.0) >= similarity_threshold
        and metrics.get("edit_distance", 999) <= max_edit_distance
    )


__all__ = [
    "normalize_domain",
    "registrable_domain",
    "domain_label",
    "edit_distance",
    "best_brand_match",
    "typosquat_flag",
]
