import re
from collections import Counter, defaultdict


SUSTAINABILITY_KW = ["organic", "compost", "zero waste", "sustainable",
                     "biodiversity", "eco", "recycle", "upcycled"]
EDUCATION_KW      = ["workshop", "course", "training", "learn", "education",
                     "skill", "teach", "class", "seminar"]
EVENT_KW          = ["event", "festival", "market", "fair", "gathering",
                     "meetup", "social", "celebration"]
COLLAB_KW         = ["farm", "restaurant", "supermarket", "store", "shop",
                     "partner", "collaborate", "supplier", "donated by"]
VOLUNTEER_KW      = ["volunteer", "volunteering", "voluntary", "unpaid"]
COOPERATIVE_KW    = ["cooperative", "co-op", "coop", "collective", "mutual"]
NETWORK_KW        = ["food not bombs", "transition towns", "feastly",
                     "olio", "too good to go", "real junk food", "hubbub",
                     "fareshare", "fare share", "trussell trust",
                     "felix project", "community fridge network",
                     "pantry network", "food cloud", "foodcloud"]
ONLINE_KW         = ["app", "digital platform", "online booking",
                     "register online", "sign up online", "book online",
                     "virtual", "social media", "facebook", "instagram",
                     "twitter", "whatsapp", "telegram", "online order",
                     "online form", "digital coordination"]
PHYSICAL_KW       = ["shop", "centre", "center", "location", "address",
                     "visit us", "drop in", "collection point", "drop-in",
                     "our premises", "come in", "walk in"]
AUDIENCE_KW       = {
    "low-income": ["low income", "poverty", "food poverty", "disadvantaged",
                   "vulnerable", "hardship", "struggling"],
    "students":   ["student", "university", "college", "school", "campus"],
    "general":    ["community", "everyone", "public", "all welcome",
                   "anyone", "neighbours"],
    "elderly":    ["elderly", "older people", "seniors", "age"],
    "children":   ["children", "kids", "families", "youth", "young people"],
}
HOURS_KW          = ["opening hours", "open hours", "opening times",
                     "drop-in", "drop in session",
                     "monday to friday", "monday -", "tuesday to",
                     "closed on", "open daily", "open weekdays",
                     "open weekends", "9am", "10am", "11am", "12pm",
                     "1pm", "2pm", "9:00", "10:00", "11:00"]
CROWDFUND_KW      = ["crowdfund", "crowdsource", "gofundme", "kickstarter",
                     "donate", "donation", "fundrais"]
NGO_KW            = ["charity", "ngo", "non-profit", "nonprofit",
                     "registered charity", "charitable"]
INFORMAL_KW       = ["community group", "informal", "grassroots",
                     "neighbourhood", "neighbors", "neighbours", "local group"]


def _count_contains(records: list[dict], keywords: list[str]) -> int:
    return sum(
        1 for r in records
        if any(kw in r.get("text", "").lower() for kw in keywords)
    )


def extract_facts(records: list[dict]) -> dict:
    total = len(records)

    # ── geographic ────────────────────────────────────────────────────────────
    district_counts = Counter(
        r.get("location", {}).get("district_name", "")
        for r in records
        if r.get("location", {}).get("match_type") != "nearest"
        and r.get("location", {}).get("district_name", "")
        not in ("", "Unknown", "Not Specified")
    )
    district_counts = dict(
        sorted(district_counts.items(), key=lambda x: -x[1])
    )
    top_districts  = sorted(district_counts.items(),
                            key=lambda x: -x[1])[:3]
    sparse         = [d for d, n in district_counts.items() if n == 1][:3]
    approx_count   = sum(
        1 for r in records
        if r.get("location", {}).get("coords_approx", False)
    )

    # ── FSI types ─────────────────────────────────────────────────────────────
    type_counts = Counter(r.get("fsi_type", "unknown") for r in records)
    type_counts = dict(sorted(type_counts.items(), key=lambda x: -x[1]))
    type_pct    = {t: round(n / total * 100, 1) for t, n in type_counts.items()}

    # ── operational ───────────────────────────────────────────────────────────
    level_counts = Counter(
        r.get("operational_level", "unknown") for r in records
    )

    # ── activity ──────────────────────────────────────────────────────────────
    pop_counts = Counter(
        r.get("popularity", {}).get("popularity_level", "unknown")
        for r in records
    )
    quantities = []
    for r in records:
        nums = re.findall(
            r'\b(\d{2,6})\s*'
            r'(?:people|members|meals|kg|tonnes|families|volunteers)',
            r.get("text", ""), re.IGNORECASE
        )
        quantities.extend(int(n) for n in nums)

    return {
        "total":            total,
        "approx_coord_count": approx_count,
        "approx_coord_pct":   round(approx_count / total * 100, 1) if total else 0,
        "district_count":   len(district_counts),
        "district_counts":  district_counts,
        "top_districts":    top_districts,
        "sparse_districts": sparse,
        "type_counts":      type_counts,
        "type_pct":         type_pct,
        "level_counts":     dict(level_counts.most_common()),
        "n_govt":           level_counts.get("government_funded", 0),
        "n_council":        level_counts.get("council_supported", 0),
        "n_ngo":            level_counts.get("ngo_led", 0),
        "n_community":      level_counts.get("community_led", 0),
        "n_commercial":     level_counts.get("commercial", 0),
        "n_volunteer":      _count_contains(records, VOLUNTEER_KW),
        "n_coop":           _count_contains(records, COOPERATIVE_KW),
        "n_crowdfund":      _count_contains(records, CROWDFUND_KW),
        "n_ngo_formal":     _count_contains(records, NGO_KW),
        "n_informal":       _count_contains(records, INFORMAL_KW),
        "n_network":        _count_contains(records, NETWORK_KW),
        "pop_counts":       dict(pop_counts),
        "total_volume":     sum(quantities),
        "max_volume":       max(quantities) if quantities else 0,
        "n_online":         _count_contains(records, ONLINE_KW),
        "n_physical":       _count_contains(records, PHYSICAL_KW),
        "n_hours_listed":   _count_contains(records, HOURS_KW),
        "audience_counts":  {
            label: _count_contains(records, kws)
            for label, kws in AUDIENCE_KW.items()
        },
        "n_sustainability": _count_contains(records, SUSTAINABILITY_KW),
        "n_collab":         _count_contains(records, COLLAB_KW),
        "n_education":      _count_contains(records, EDUCATION_KW),
        "n_events":         _count_contains(records, EVENT_KW),
    }