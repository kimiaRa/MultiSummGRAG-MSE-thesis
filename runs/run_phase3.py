import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "phase_3"))

from config import CITIES
from prepare_corpus import prepare_corpus
from graph_builder import build_index
from querier import query_all


def run(city_name: str):
    from config_loader import load_city_config, build_districts_geojson     
    cfg = load_city_config(city_name)

    print(f"[{city_name}] Step 3a — preparing corpus...")
    documents = prepare_corpus(
        in_file=Path(cfg["in_file"]),
        graph_dir=Path(cfg["graph_dir"]),
    )

    print(f"[{city_name}] Step 3b — building graph index...")
    build_index(documents, str(cfg["graph_dir"]))

    print(f"[{city_name}] Step 3c — querying graph...")
    results = query_all(
        working_dir=str(cfg["graph_dir"]),
        out_file=Path(cfg["out_file"]),
    )

    print(f"\n✓ Phase 3 complete")
    for field, result in results.items():
        answer = result.get("answer", "")
        print(f"\n{'─'*60}")
        print(f"Field : {field}")
        print(f"Answer: {answer[:300]}{'...' if len(answer) > 300 else ''}")


if __name__ == "__main__":
    import sys as _sys
    city = _sys.argv[1] if len(_sys.argv) > 1 else "dublin"
    run(city)