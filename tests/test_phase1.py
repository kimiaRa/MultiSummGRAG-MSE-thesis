import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
from config_loader import load_city_config

city_name = sys.argv[1] if len(sys.argv) > 1 else "dublin"
cfg       = load_city_config(city_name)

records = [
    json.loads(l)
    for l in cfg["fsi_records"].read_text(encoding="utf-8").splitlines()
    if l.strip()
]

print(f"=== Phase 1 — {cfg['city']}, {cfg['country']} ===\n")
print(f"Total records : {len(records)}")
print(f"With text     : {sum(1 for r in records if len(r.get('text','')) > 100)}")
print(f"With geo      : {sum(1 for r in records if r.get('geo'))}")
print(f"With images   : {sum(1 for r in records if r.get('images'))}")
print(f"Errors        : {sum(1 for r in records if r.get('error'))}")

for r in records:
    if r.get("text"):
        print(f"\nSample — {r['title']}")
        print(f"Geo    — {r.get('geo')}")
        print(f"Text   — {r['text'][:300]}")
        break