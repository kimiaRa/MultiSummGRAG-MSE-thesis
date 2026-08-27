import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "phase_2"))

import json
import csv as _csv
from config import CITIES
from geofencer import load_districts, geofence_all
from classifier import classify_all
from popularity import score_all
from merger import merge_all


def _apply_csv_coords(records: list[dict], urls_file: str) -> int:
    """Override geo coords in records with the curated values from urls_cleaned.csv."""
    coord_map: dict[str, tuple[float, float]] = {}
    with open(urls_file, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            try:
                coord_map[row["URL"]] = (float(row["Lat"]), float(row["Lng"]))
            except (KeyError, ValueError):
                pass

    updated = 0
    for r in records:
        url = r.get("url", "")
        if url in coord_map:
            lat, lng = coord_map[url]
            r["geo"] = {"lat": lat, "lng": lng, "source": "csv"}
            updated += 1
    return updated


def run(city_name: str):
    from config_loader import load_city_config, build_districts_geojson
    cfg = load_city_config(city_name)

    print(f"[{city_name}] Loading records...")
    records = [
        json.loads(l)
        for l in Path(cfg["in_file"]).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    print(f"  {len(records)} records loaded")

    # Use curated CSV coordinates as the authoritative source for geofencing
    urls_file = cfg.get("urls_file")
    if urls_file and Path(str(urls_file)).exists():
        n = _apply_csv_coords(records, str(urls_file))
        print(f"  Applied CSV coordinates to {n} records")

    print(f"[{city_name}] Step 2a — geofencing...")
    districts = load_districts(cfg["districts_file"])
    records = geofence_all(records, districts)

    print(f"[{city_name}] Step 2b — LLM classification...")
    records = classify_all(records)

    print(f"[{city_name}] Step 2c — popularity scoring...")
    records = score_all(records)

    print(f"[{city_name}] Step 2d — merging into enriched records...")
    enriched = merge_all(records, cfg)

    out_path = Path(cfg["out_file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in enriched:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # summary stats
    districts_seen = set(r["location"]["district_name"] for r in enriched)
    types_seen     = set(r["fsi_type"] for r in enriched)
    exact_geo      = sum(1 for r in enriched if r["location"]["match_type"] == "exact")

    print(f"\n✓ {len(enriched)} enriched records written to {out_path}")
    print(f"✓ {exact_geo}/{len(enriched)} exact district matches")
    print(f"✓ {len(districts_seen)} districts represented: {sorted(districts_seen)}")
    print(f"✓ FSI types found: {sorted(types_seen)}")


if __name__ == "__main__":
    import sys as _sys
    city = _sys.argv[1] if len(_sys.argv) > 1 else "dublin"
    run(city)