import json
from pathlib import Path


def generate_review_report(rows: list[dict],
                            city: str,
                            out_path: Path) -> dict:
    """
    Generates a report of all records needing manual review.
    Saves as JSON and prints a readable summary.
    """
    needs_review = [r for r in rows if r.get("needs_review")]
    resolved_s1  = [r for r in rows if r.get("coord_strategy") == 1]
    resolved_s2  = [r for r in rows if r.get("coord_strategy") == 2]

    report = {
        "city":              city,
        "total_missing":     len([r for r in rows
                                  if not r.get("Lat") or
                                  str(r.get("Lat","")).strip() == ""]
                                 ) + len(resolved_s1) + len(resolved_s2)
                                   + len(needs_review),
        "resolved_strategy1": len(resolved_s1),
        "resolved_strategy2": len(resolved_s2),
        "unresolved":         len(needs_review),
        "needs_manual_review": [
            {
                "url":    r["URL"],
                "reason": "coordinates could not be found automatically",
                "action": "look up on Google Maps and add Lat/Lng to raw_urls.csv",
            }
            for r in needs_review
        ],
        "resolved": [
            {
                "url":      r["URL"],
                "lat":      r.get("Lat"),
                "lng":      r.get("Lng"),
                "strategy": r.get("coord_strategy"),
                "query":    r.get("coord_query"),
            }
            for r in resolved_s1 + resolved_s2
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # print summary
    print(f"\n  {'='*46}")
    print(f"  Coordinate Resolution Report — {city}")
    print(f"  {'='*46}")
    print(f"  Resolved via page scrape    : {len(resolved_s1)}")
    print(f"  Resolved via domain search  : {len(resolved_s2)}")
    print(f"  Unresolved (needs manual)   : {len(needs_review)}")
    if needs_review:
        print(f"\n  URLs needing manual coordinates:")
        for r in needs_review:
            print(f"    → {r['URL']}")
    print(f"\n  Full report → {out_path}")
    print(f"  {'='*46}\n")

    return report