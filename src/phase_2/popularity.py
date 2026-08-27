import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import POPULARITY


def _match_keyword(text_lower: str, kw: str) -> bool:
    if " " in kw:
        return kw in text_lower
    return bool(re.search(r'\b' + re.escape(kw) + r'\b', text_lower))


def _count_keywords(text: str, keywords: list[str]) -> int:
    text_lower = text.lower()
    return sum(1 for kw in keywords if _match_keyword(text_lower, kw))


def _extract_signals(text: str) -> dict:
    numbers = re.findall(r'\b(\d{2,6})\s*(?:people|members|volunteers|families|meals|kg|tonnes)',
                         text, re.IGNORECASE)
    quantities = [int(n) for n in numbers]

    return {
        "max_quantity":    max(quantities) if quantities else 0,
        "high_keywords":   _count_keywords(text, POPULARITY["high"]),
        "medium_keywords": _count_keywords(text, POPULARITY["medium"]),
        "low_keywords":    _count_keywords(text, POPULARITY["low"]),
        "text_length":     len(text),
        "has_numbers":     len(quantities) > 0,
    }


def _raw_score(signals: dict) -> int:
    return (
        signals["high_keywords"]   * 3 +
        signals["medium_keywords"] * 1 +
        signals["low_keywords"]    * -1 +
        (2 if signals["max_quantity"] > 100  else 0) +
        (1 if signals["max_quantity"] > 10   else 0) +
        (1 if signals["text_length"]  > 1000 else 0)
    )


def score_all(records: list[dict]) -> list[dict]:
    """
    Score all records and assign popularity levels by percentile rank within
    the city dataset. Top 25% → high, middle 50% → medium, bottom 25% → low.
    This avoids penalising cities where most FSIs have minimal web presence.
    """
    # first pass: compute raw scores and signals for every record
    scored = []
    for r in records:
        text    = r.get("text") or ""
        signals = _extract_signals(text)
        score   = _raw_score(signals)
        scored.append((r, score, signals))

    # derive percentile thresholds from this city's distribution
    all_scores = sorted(s for _, s, _ in scored)
    n = len(all_scores)
    hi_thresh = all_scores[max(0, int(0.75 * n))]
    lo_thresh = all_scores[max(0, int(0.25 * n))]

    # second pass: assign levels
    for r, score, signals in scored:
        if score >= hi_thresh:
            level = "high"
        elif score > lo_thresh:
            level = "medium"
        else:
            level = "low"
        r["popularity"] = {
            "popularity_level":   level,
            "popularity_score":   score,
            "popularity_signals": signals,
        }

    return records
