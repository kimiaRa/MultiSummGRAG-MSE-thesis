from collections import Counter


def generate_stats(rows: list[dict],
                   city: str,
                   country: str) -> dict:
    total        = len(rows)
    duplicates   = sum(1 for r in rows if r.get("is_duplicate"))
    non_fsi_high = sum(1 for r in rows if r.get("is_non_fsi")
                       and r.get("non_fsi_confidence") == "high")
    low_conf     = sum(1 for r in rows if not r.get("is_non_fsi")
                       and r.get("non_fsi_confidence") == "low")
    with_coords  = sum(
        1 for r in rows
        if r.get("Lat") and str(r.get("Lat", "")).strip() != ""
    )
    missing_coords = total - with_coords
    lang_counts  = Counter(r.get("language", "unknown") for r in rows)

    return {
        "city":                   city,
        "country":                country,
        "total_input":            total,
        "duplicates_detected":    duplicates,
        "likely_non_fsi":         non_fsi_high,
        "low_confidence":         low_conf,
        "with_coords":            with_coords,
        "missing_coords":         missing_coords,
        "language_distribution":  dict(lang_counts.most_common()),
        "note": "Phase 0 is analysis only — no URLs removed from pipeline"
    }


def print_stats(stats: dict):
    print(f"\n{'='*50}")
    print(f"Phase 0 — Dataset Analysis Report")
    print(f"  City    : {stats['city']}, {stats['country']}")
    print(f"{'='*50}")
    print(f"  Total URLs              : {stats['total_input']}")
    print(f"  Duplicates detected     : {stats['duplicates_detected']}")
    print(f"  Likely non-FSI          : {stats['likely_non_fsi']}")
    print(f"  Low confidence          : {stats['low_confidence']}")
    print(f"  With coordinates        : {stats['with_coords']}")
    print(f"  Missing coordinates     : {stats['missing_coords']}")
    print(f"\n  Language distribution:")
    for lang, n in stats["language_distribution"].items():
        print(f"    {lang:10s} {n}")
    print(f"\n  NOTE: All URLs pass through to Phase 1 unchanged.")
    print(f"        Flags are for thesis documentation only.")
    print(f"{'='*50}\n")