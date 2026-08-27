import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
from config_loader import load_city_config

city_name = sys.argv[1] if len(sys.argv) > 1 else "dublin"
cfg       = load_city_config(city_name)

schema = json.loads(
    cfg["phase4_schema"].read_text(encoding="utf-8")
)

overview  = schema["city_overview"]
districts = schema["districts"]
stats     = overview["stats"]

print(f"=== Phase 4 — {cfg['city']}, {cfg['country']} ===\n")
print(f"Country  : {schema['meta']['country']}")
print(f"City     : {schema['meta']['city']}")
print(f"FSIs     : {stats['total_fsi_count']}")
print(f"Districts: {stats['district_count']}")

print(f"\n=== FSI type breakdown ===")
for t, n in stats["fsi_type_breakdown"].items():
    print(f"  {t:30s} {n}")

print(f"\n=== Sentiment breakdown ===")
for t, n in stats["sentiment_breakdown"].items():
    print(f"  {t:30s} {n}")

print(f"\n=== District breakdown ===")
for name, d in districts.items():
    imgs = len(d["representative_images"])
    print(f"  {name:15s} {d['fsi_count']:2d} FSIs  "
          f"{imgs} imgs  "
          f"types: {list(d['fsi_types'].keys())[:2]}")

print(f"\n=== City representative images ===")
for img in overview["representative_images"]:
    print(f"  [{img['width']}x{img['height']}] {img.get('district','')} "
          f"— {Path(img['local_path']).name}")

print(f"\n=== Sample district — top initiative ===")
first_district = list(districts.values())[0]
top = first_district["initiatives"][0]
print(f"District   : {first_district['district_name']}")
print(f"Initiative : {top['title']}")
print(f"Type       : {top['fsi_type']}")
print(f"Level      : {top['operational_level']}")
print(f"Sentiment  : {top['sentiment']} — {top['sentiment_reason']}")
print(f"Popularity : {top['popularity_level']} (score {top['popularity_score']})")