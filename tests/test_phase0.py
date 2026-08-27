import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
from config_loader import load_city_config

city_name = sys.argv[1] if len(sys.argv) > 1 else "dublin"
cfg       = load_city_config(city_name)

annotated_path = cfg["out_dir"] / "phase0_annotated.json"
review_path    = cfg["out_dir"] / "phase0_review.json"

if not annotated_path.exists():
    print(f"✗ Phase 0 output not found at {annotated_path}")
    sys.exit(1)

data  = json.loads(annotated_path.read_text(encoding="utf-8"))
stats = data["stats"]
rows  = data["rows"]

print(f"=== Phase 0 — {stats['city']}, {stats['country']} ===\n")
print(f"Input URLs              : {stats['total_input']}")
print(f"Duplicates detected     : {stats['duplicates_detected']}")
print(f"Likely non-FSI          : {stats['likely_non_fsi']}")
print(f"Low confidence          : {stats['low_confidence']}")
print(f"With coordinates        : {stats['with_coords']}")
print(f"Missing coordinates     : {stats['missing_coords']}")
print(f"Note                    : {stats['note']}")

print(f"\nLanguage distribution:")
for lang, n in stats["language_distribution"].items():
    print(f"  {lang:10s} {n}")

print(f"\nLikely non-FSI URLs (flagged, not removed):")
for r in rows:
    if r.get("is_non_fsi"):
        print(f"  ✗ {r['URL'][:65]}")
        print(f"    reason: {r.get('non_fsi_reason')}")

print(f"\nLow confidence — review manually:")
for r in rows:
    if not r.get("is_non_fsi") and r.get("non_fsi_confidence") == "low":
        print(f"  ? {r['URL'][:65]}")

print(f"\nMissing coordinates (will be Not Specified in pipeline):")
for r in rows:
    if not r.get("Lat") or str(r.get("Lat", "")).strip() == "":
        print(f"  → {r['URL'][:65]}")

if review_path.exists():
    review = json.loads(review_path.read_text(encoding="utf-8"))
    print(f"\nCoordinate resolution:")
    print(f"  Unresolved: {review['unresolved']}")

print(f"\nurls.csv → {cfg['urls_file']}")