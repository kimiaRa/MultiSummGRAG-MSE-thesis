# src/phase_0/fsi_detector.py

import re
import requests
from urllib.parse import urlparse

HEADERS = {"User-Agent": "fsi-thesis-bot/1.0"}
TIMEOUT = 10

# ── URL-pattern rules — no scraping needed ────────────────────────────────────

NON_FSI_DOMAINS = {
    "opendocs.ids.ac.uk", "researchgate.net", "academia.edu",
    "scholar.google.com", "jstor.org",                    # academic
    "meetup.com", "simpletix.com", "eventbrite.com",      # event platforms
    "blogspot.com", "wordpress.com", "medium.com",        # generic blogs
    "restaurantsbrighton.co.uk", "tripadvisor.com",       # restaurant aggregators
    "emmaus-international.org", "emmaus-europe.org",      # international umbrellas
    "realfarming.org", "seedsovereignty.info",            # national orgs
}

NON_FSI_PATH_PATTERNS = [
    r"/covid[-_]?19",
    r"/coronavirus",
    r"/glossary",
    r"/dictionary",
    r"/definitions",
    r"/search\?",
    r"/Search\?",
    r"/jobs/blog/",
    r"\d{4}/\d{2}/\d{2}/",   # date-stamped blog posts
    r"/articles/report/",
    r"/gathering-the-seed-community",   # specific known non-FSI
]

NON_FSI_TITLE_SIGNALS = [
    "page not found", "404", "not found", "error",
    "search results", "jobs", "blog post",
    "glossary", "dictionary",
]

NON_FSI_TEXT_SIGNALS = [
    "this page cannot be found",
    "404 error",
    "page not found",
    "domain for sale",
    "buy this domain",
    "this site has not been published",
    "site offline",
    "coming soon",
    "under construction",
    "account suspended",
]

FSI_TEXT_SIGNALS = [
    "food", "sharing", "community", "garden", "bank", "kitchen",
    "donate", "volunteer", "meals", "hunger", "waste", "grow",
    "harvest", "distribution", "pantry", "fridge", "surplus",
    "cooperative", "charity", "initiative", "project",
]


def _url_pattern_check(url: str) -> tuple[bool, str]:
    """
    Fast check — no network requests.
    Returns (is_non_fsi, reason).
    """
    parsed = urlparse(url)
    domain = parsed.netloc.replace("www.", "").lower()
    path   = parsed.path.lower()

    # domain check
    if domain in NON_FSI_DOMAINS:
        return True, f"non-FSI domain: {domain}"

    # path pattern check
    for pat in NON_FSI_PATH_PATTERNS:
        if re.search(pat, url, re.IGNORECASE):
            return True, f"non-FSI path pattern: {pat}"

    return False, ""


def _text_check(url: str) -> tuple[bool, str, int]:
    """
    Lightweight fetch — checks page text for FSI signals.
    Returns (is_non_fsi, reason, fsi_score).
    """
    try:
        resp = requests.get(url, timeout=TIMEOUT, headers=HEADERS,
                            allow_redirects=True)
        text_lower = resp.text.lower()

        # check for dead/non-FSI signals
        for signal in NON_FSI_TEXT_SIGNALS:
            if signal in text_lower:
                return True, f"page signal: '{signal}'", 0

        # check title
        title_match = re.search(
            r"<title[^>]*>(.*?)</title>", resp.text, re.IGNORECASE | re.DOTALL
        )
        if title_match:
            title_lower = title_match.group(1).lower()
            for signal in NON_FSI_TITLE_SIGNALS:
                if signal in title_lower:
                    return True, f"title signal: '{signal}'", 0

        # score FSI relevance
        fsi_score = sum(
            1 for sig in FSI_TEXT_SIGNALS if sig in text_lower
        )
        return False, "", fsi_score

    except Exception as e:
        return False, "", 0


def detect_non_fsi(rows: list[dict]) -> list[dict]:
    """
    Flags rows that are likely not FSIs.
    Adds: is_non_fsi, non_fsi_reason, fsi_score, non_fsi_confidence
    """
    print(f"  Checking {len(rows)} URLs for FSI relevance...")

    for i, row in enumerate(rows):
        url = row["URL"]

        # layer 1 — URL pattern (instant)
        is_non_fsi, reason = _url_pattern_check(url)
        if is_non_fsi:
            row["is_non_fsi"]         = True
            row["non_fsi_reason"]     = reason
            row["fsi_score"]          = 0
            row["non_fsi_confidence"] = "high"
            print(f"    [{i+1}/{len(rows)}] ✗ HIGH  {url[:55]}")
            print(f"           reason: {reason}")
            continue

        # layer 2 — text check (requires fetch)
        is_non_fsi, reason, fsi_score = _text_check(url)
        if is_non_fsi:
            row["is_non_fsi"]         = True
            row["non_fsi_reason"]     = reason
            row["fsi_score"]          = 0
            row["non_fsi_confidence"] = "high"
            print(f"    [{i+1}/{len(rows)}] ✗ HIGH  {url[:55]}")
            print(f"           reason: {reason}")
        elif fsi_score == 0:
            row["is_non_fsi"]         = False
            row["non_fsi_reason"]     = ""
            row["fsi_score"]          = 0
            row["non_fsi_confidence"] = "low"
            print(f"    [{i+1}/{len(rows)}] ? LOW   {url[:55]}")
            print(f"           no FSI signals found — review manually")
        else:
            row["is_non_fsi"]         = False
            row["non_fsi_reason"]     = ""
            row["fsi_score"]          = fsi_score
            row["non_fsi_confidence"] = "ok"
            print(f"    [{i+1}/{len(rows)}] ✓ OK    {url[:55]}")
            print(f"           FSI score: {fsi_score}")

    return rows