import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
from config_loader import load_city_config

city_name = sys.argv[1] if len(sys.argv) > 1 else "dublin"
cfg       = load_city_config(city_name)

results = json.loads(
    cfg["phase3_answers"].read_text(encoding="utf-8")
)

print(f"=== Phase 3 — {cfg['city']}, {cfg['country']} ===\n")
print(f"Total query fields answered: {len(results)}\n")

for field, data in results.items():
    answer = data.get("answer", "")
    error  = data.get("error", "")
    status = "✓" if answer and not error else "✗"
    print(f"{status} {field}")
    print(f"  mode   : {data['mode']}")
    if error:
        print(f"  error  : {error}")
    else:
        print(f"  length : {len(answer)} chars")
        print(f"  preview: {answer[:200]}...")
    print()