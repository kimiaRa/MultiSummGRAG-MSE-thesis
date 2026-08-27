import re
from collections import Counter


# ── keyword sets for each dimension ──────────────────────────────────────────

KEYWORDS = {
    "urban_indicators": [
        "city centre", "urban", "inner city", "city council",
        "metropolitan", "town centre",
    ],
    "peri_urban_indicators": [
        "suburban", "peri-urban", "rural", "outskirts",
        "village", "countryside", "farm",
    ],
    "crowdsourcing": [
        "crowdfund", "crowdsource", "donate", "donation", "fundrais",
        "gofundme", "patreon", "sponsor",
    ],
    "volunteer_model": [
        "volunteer", "volunteering", "voluntary", "unpaid",
        "community-led", "grassroots",
    ],
    "cooperative_model": [
        "cooperative", "co-op", "coop", "collective", "shared ownership",
        "member", "membership",
    ],
    "ngo_charity": [
        "charity", "charitable", "ngo", "non-profit", "nonprofit",
        "registered charity", "foundation",
    ],
    "informal": [
        "informal", "community group", "neighbours", "neighbors",
        "local group", "grassroots",
    ],
    "social_media": [
        "facebook", "instagram", "twitter", "tiktok", "youtube",
        "social media", "follow us",
    ],
    "online_coordination": [
        "online", "virtual", "app", "website", "platform",
        "digital", "whatsapp", "telegram",
    ],
    "physical_location": [
        "location", "address", "visit us", "open", "drop in",
        "centre", "center", "hub", "shop", "store",
    ],
    "low_income_target": [
        "low income", "poverty", "food poverty", "food insecurity",
        "vulnerable", "disadvantaged", "homeless", "refugees",
    ],
    "student_target": [
        "student", "university", "college", "school", "youth",
    ],
    "elderly_target": [
        "elderly", "older people", "senior", "age friendly",
    ],
    "general_public": [
        "everyone", "all welcome", "community", "public",
        "open to all", "anyone",
    ],
    "sustainability": [
        "sustainable", "sustainability", "organic", "compost",
        "zero waste", "eco", "environment", "green", "carbon",
        "biodiversity", "recycl",
    ],
    "farm_collaboration": [
        "farm", "farmer", "allotment", "grow", "harvest",
        "growers", "garden",
    ],
    "restaurant_collaboration": [
        "restaurant", "cafe", "coffee shop", "supermarket",
        "grocery", "food business", "catering",
    ],
    "education": [
        "workshop", "course", "class", "learn", "education",
        "training", "skill", "teach", "seminar",
    ],
    "events": [
        "event", "festival", "gathering", "meetup", "meeting",
        "swap", "fair", "market", "open day",
    ],
    "food_network": [
        "food not bombs", "transition town", "slow food",
        "real food", "food bank network", "fareshare",
        "olio", "too good to go",
    ],
}


def _count_keyword(text: str, keywords: list[str]) -> int:
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def _has_keyword(text: str, keywords: list[str]) -> bool:
    return _count_keyword(text, keywords) > 0


def _extract_language_hints(text: str) -> str:
    """Very lightweight language detection based on common function words."""
    samples = {
        "en": ["the", "and", "is", "are", "we", "our", "for", "with"],
        "fr": ["le", "la", "les", "et", "est", "nous", "pour", "avec"],
        "de": ["der", "die", "das", "und", "ist", "wir", "für", "mit"],
        "es": ["el", "la", "los", "y", "es", "nosotros", "para", "con"],
        "ga": ["an", "na", "agus", "le", "do", "ar", "faoi"],   # Irish
    }
    text_lower = text.lower()
    words      = re.findall(r'\b\w+\b', text_lower)
    word_set   = set(words)
    scores     = {lang: sum(1 for w in kws if w in word_set)
                  for lang, kws in samples.items()}
    return max(scores, key=scores.get)


def analyse_records(records: list[dict]) -> dict:
    """
    Runs lightweight text analytics over all enriched FSI records.
    Returns a flat dict of counts and percentages for each dimension.
    """
    n = len(records)
    if n == 0:
        return {}

    counts = {key: 0 for key in KEYWORDS}
    counts["total"] = n
    language_counts = Counter()

    for r in records:
        text = (r.get("text") or "") + " " + (r.get("description") or "")
        text = text.lower()

        for key, kws in KEYWORDS.items():
            if _has_keyword(text, kws):
                counts[key] += 1

        lang = _extract_language_hints(text)
        language_counts[lang] += 1

    # compute percentages
    pct = {k: round(v / n * 100, 1) for k, v in counts.items() if k != "total"}

    return {
        "total_records": n,
        "counts":        counts,
        "percentages":   pct,
        "languages":     dict(language_counts.most_common()),

        # derived answers to specific questions
        "q2_urban_vs_periurban": {
            "urban":      counts["urban_indicators"],
            "peri_urban": counts["peri_urban_indicators"],
            "unclassified": n - counts["urban_indicators"] - counts["peri_urban_indicators"],
        },
        "q10_crowdsourcing":       counts["crowdsourcing"],
        "q12_volunteer_model":     counts["volunteer_model"],
        "q12_cooperative_model":   counts["cooperative_model"],
        "q13_ngo_or_charity":      counts["ngo_charity"],
        "q14_informal":            counts["informal"],
        "q15_social_media":        counts["social_media"],
        "q15_online_coordination": counts["online_coordination"],
        "q16_food_networks":       counts["food_network"],
        "q20_languages":           dict(language_counts.most_common()),
        "q24_physical_location":   counts["physical_location"],
        "q25_low_income_target":   counts["low_income_target"],
        "q25_student_target":      counts["student_target"],
        "q25_elderly_target":      counts["elderly_target"],
        "q25_general_public":      counts["general_public"],
        "q26_sustainability":      counts["sustainability"],
        "q27_farm_collab":         counts["farm_collaboration"],
        "q27_restaurant_collab":   counts["restaurant_collaboration"],
        "q28_education":           counts["education"],
        "q29_events":              counts["events"],
    }