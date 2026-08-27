import requests
from langdetect import detect, LangDetectException

TIMEOUT = 10
HEADERS = {"User-Agent": "fsi-thesis-bot/1.0"}


def _fetch_text(url: str) -> str | None:
    try:
        resp = requests.get(url, timeout=TIMEOUT, headers=HEADERS)
        # rough text extraction — just strip tags
        import re
        text = re.sub(r"<[^>]+>", " ", resp.text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:2000]   # first 2000 chars is enough for detection
    except Exception:
        return None


def detect_language(rows: list[dict]) -> list[dict]:
    """
    Detects language for each reachable URL.
    Skips dead or unreachable pages.
    """
    for row in rows:
        if not row.get("reachable", False):
            row["language"]      = "unknown"
            row["language_conf"] = None
            continue
        text = _fetch_text(row["URL"])
        if not text:
            row["language"]      = "unknown"
            row["language_conf"] = None
            continue
        try:
            lang = detect(text)
            row["language"]      = lang
            row["language_conf"] = "detected"
        except LangDetectException:
            row["language"]      = "unknown"
            row["language_conf"] = None

    return rows