import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
from collections import Counter
from config_loader import load_city_config

city_name = sys.argv[1] if len(sys.argv) > 1 else "dublin"
cfg       = load_city_config(city_name)

records = [
    json.loads(l)
    for l in cfg["fsi_enriched"].read_text(encoding="utf-8").splitlines()
    if l.strip()
]

print(f"=== Phase 2 — {cfg['city']}, {cfg['country']} ===\n")
print(f"Total enriched records : {len(records)}")

print(f"\n--- Location ---")
print(f"Exact district match   : {sum(1 for r in records if r['location']['match_type'] == 'exact')}")
print(f"Nearest fallback       : {sum(1 for r in records if r['location']['match_type'] == 'nearest')}")
print(f"No coords              : {sum(1 for r in records if r['location']['match_type'] == 'no_coords')}")

print(f"\n--- FSI Types ---")
for t, n in Counter(r["fsi_type"] for r in records).most_common():
    print(f"  {t:30s} {n}")

print(f"\n--- Operational Levels ---")
for t, n in Counter(r["operational_level"] for r in records).most_common():
    print(f"  {t:30s} {n}")

print(f"\n--- Sentiment ---")
for t, n in Counter(r["sentiment"] for r in records).most_common():
    print(f"  {t:30s} {n}")

print(f"\n--- Popularity ---")
for t, n in Counter(r["popularity"]["popularity_level"] for r in records).most_common():
    print(f"  {t:30s} {n}")

print(f"\n--- Districts ---")
for t, n in Counter(r["location"]["district_name"] for r in records).most_common():
    print(f"  {t:30s} {n}")

print(f"\n--- Sample record ---")
print(json.dumps(records[0], indent=2, ensure_ascii=False))