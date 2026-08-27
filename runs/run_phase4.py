import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "phase_4"))

import json
from config import CITIES
from schema_builder import build_schema


def run(city_name: str):
    from config_loader import load_city_config, build_districts_geojson     
    cfg = load_city_config(city_name)

    print(f"[{city_name}] Loading enriched records...")
    records = [
        json.loads(l)
        for l in Path(cfg["enriched_file"]).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    print(f"  {len(records)} records loaded")

    print(f"[{city_name}] Loading Phase 3 graph answers...")
    graph_answers = json.loads(
        Path(cfg["answers_file"]).read_text(encoding="utf-8")
    )
    print(f"  {len(graph_answers)} query answers loaded")

    print(f"[{city_name}] Building summary schema...")
    schema = build_schema(records, graph_answers, cfg)

    out_path = Path(cfg["out_file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(schema, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # summary
    districts = schema["districts"]
    total_imgs = sum(
        len(d["representative_images"]) for d in districts.values()
    )
    print(f"\n✓ Schema written to {out_path}")
    print(f"  Districts        : {len(districts)}")
    print(f"  FSIs total       : {schema['city_overview']['stats']['total_fsi_count']}")
    print(f"  District images  : {total_imgs}")
    print(f"  City images      : {len(schema['city_overview']['representative_images'])}")
    print(f"\n  Top districts by FSI count:")
    for name, d in list(districts.items())[:5]:
        print(f"    {name:15s} {d['fsi_count']} FSIs  "
              f"| types: {list(d['fsi_types'].keys())[:3]}")


if __name__ == "__main__":
    import sys as _sys
    city = _sys.argv[1] if len(_sys.argv) > 1 else "dublin"
    run(city)