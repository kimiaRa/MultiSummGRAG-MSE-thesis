import sys, argparse
from pathlib import Path

src = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src))

from config_loader import load_city_config, build_districts_geojson, OLLAMA

def parse_args():
    parser = argparse.ArgumentParser(
        description="FSI MultiSumm Pipeline"
    )
    parser.add_argument("--city",   required=True,
                        help="City name matching a config/CITY.yaml file")
    parser.add_argument("--phases", default="1,2,3,4,5",
                        help="Comma-separated phases to run (default: all). "
                             "The pipeline ends at Phase 5 (report generation); "
                             "the former Phase 6 evaluation stage is retired "
                             "and no longer part of this repository.")
    parser.add_argument("--skip",   default="",
                        help="Comma-separated phases to skip")
    return parser.parse_args()


def _use_phase(phase_name: str):
    """Context manager that adds a phase dir to sys.path then removes it."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        path = str(src / phase_name)
        sys.path.insert(0, path)
        try:
            yield
        finally:
            # remove all copies of this path to avoid pollution
            while path in sys.path:
                sys.path.remove(path)
            # also remove any cached modules from this phase
            to_del = [k for k in sys.modules
                      if hasattr(sys.modules[k], '__file__')
                      and sys.modules[k].__file__
                      and path in str(sys.modules[k].__file__)]
            for k in to_del:
                del sys.modules[k]
    return _ctx()

def run_phase0(cfg):
    with _use_phase("phase_0"):
        import json
        from url_cleaner import clean_urls
        from cleaner import load_raw_csv
        from duplicate_detector import detect_duplicates
        from missing_coord_resolver import resolve_missing_coords
        from language_detector import detect_language
        from fsi_detector import detect_non_fsi
        from stats_reporter import generate_stats, print_stats
        from review_reporter import generate_review_report

        print(f"\n{'='*50}\nPhase 0 — URL Cleaning & Data Analysis\n{'='*50}")

        raw_path = cfg["raw_urls_file"]
        if not Path(raw_path).exists():
            raise FileNotFoundError(
                f"Raw URL list not found: {raw_path}\n"
                f"Place your collected URLs at {raw_path}"
            )

        # ── Step 0a: clean URLs ───────────────────────────────────────────────
        print(f"\n  Step 0a — cleaning URLs...")
        clean_urls(cfg["city"], Path(raw_path), cfg["urls_file"])

        # ── Steps 0b-f: analyse cleaned URLs ─────────────────────────────────
        print(f"\n  Step 0b — loading cleaned URLs for analysis...")
        rows = load_raw_csv(cfg["urls_file"])
        print(f"  {len(rows)} URLs loaded")

        print(f"\n  Step 0c — detecting duplicates...")
        rows = detect_duplicates(rows)
        dupes = sum(1 for r in rows if r["is_duplicate"])
        print(f"  {dupes} duplicates detected")

        print(f"\n  Step 0d — detecting non-FSI URLs...")
        rows = detect_non_fsi(rows)
        non_fsi  = sum(1 for r in rows if r.get("is_non_fsi"))
        low_conf = sum(1 for r in rows if not r.get("is_non_fsi")
                      and r.get("non_fsi_confidence") == "low")
        print(f"  {non_fsi} likely non-FSI URLs detected")
        print(f"  {low_conf} low-confidence URLs flagged for review")

        print(f"\n  Step 0e — flagging missing coordinates...")
        rows = resolve_missing_coords(rows, cfg["city"], cfg["country"])

        print(f"\n  Step 0f — detecting languages...")
        rows = detect_language(rows)

        print(f"\n  Step 0g — generating statistics...")
        stats = generate_stats(rows, cfg["city"], cfg["country"])
        print_stats(stats)

        review_path = cfg["out_dir"] / "phase0_review.json"
        generate_review_report(rows, cfg["city"], review_path)

        annotated_path = cfg["out_dir"] / "phase0_annotated.json"
        annotated_path.parent.mkdir(parents=True, exist_ok=True)
        annotated_path.write_text(
            json.dumps({"stats": stats, "rows": rows},
                       indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"  ✓ Annotated data → {annotated_path}")

def run_phase1(cfg):
    with _use_phase("phase_1"):
        from scraper import scrape_city, load_urls
        from merger import build_fsi_record
        import asyncio, json
        from tqdm import tqdm

        print(f"\n{'='*50}\nPhase 1 — Data Acquisition\n{'='*50}")
        rows    = load_urls(cfg["urls_file"])
        results = asyncio.run(scrape_city(cfg, rows))

        seen, records = set(), []
        for scraped in tqdm(results, desc="Building records"):
            records.append(build_fsi_record(
                scraped,
                cfg["city"],
                cfg["images_dir"],
                seen,
                country=cfg["country"],   # pass country for geocoding
            ))

        cfg["fsi_records"].parent.mkdir(parents=True, exist_ok=True)
        with open(cfg["fsi_records"], "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        ok  = sum(1 for r in records if not r["error"])
        geo = sum(1 for r in records if r["geo"])
        geo_provided  = sum(1 for r in records
                            if r["geo"] and
                            r["geo"].get("source") == "provided")
        geo_extracted = sum(1 for r in records
                            if r["geo"] and
                            r["geo"].get("source") == "text_extracted")
        print(f"✓ {ok}/{len(records)} scraped")
        print(f"✓ {geo} geocoded "
              f"({geo_provided} provided, {geo_extracted} text-extracted)")

def run_phase2(cfg):
    with _use_phase("phase_2"):
        import json
        from geofencer import load_districts, geofence_all
        from classifier import classify_all
        from popularity import score_all
        from merger import merge_all

        print(f"\n{'='*50}\nPhase 2 — Enrichment\n{'='*50}")
        import csv as _csv
        build_districts_geojson(cfg)
        records = [json.loads(l) for l in
                   cfg["fsi_records"].read_text().splitlines() if l.strip()]

        # Apply curated CSV coordinates before geofencing
        urls_file = cfg.get("urls_file")
        if urls_file and Path(str(urls_file)).exists():
            coord_map = {}
            with open(str(urls_file), newline="", encoding="utf-8") as f:
                for row in _csv.DictReader(f):
                    try:
                        coord_map[row["URL"]] = (float(row["Lat"]), float(row["Lng"]))
                    except (KeyError, ValueError):
                        pass
            for r in records:
                if r.get("url") in coord_map:
                    lat, lng = coord_map[r["url"]]
                    r["geo"] = {"lat": lat, "lng": lng, "source": "csv"}
            print(f"  CSV coordinates applied to {len(coord_map)} records")

        districts = load_districts(cfg["districts_file"])
        records   = geofence_all(records, districts)
        records   = classify_all(records)
        records   = score_all(records)
        enriched  = merge_all(records, cfg)

        with open(cfg["fsi_enriched"], "w", encoding="utf-8") as f:
            for r in enriched:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"✓ {len(enriched)} enriched records written")


def run_phase3(cfg):
    with _use_phase("phase_3"):
        import json
        from prepare_corpus import prepare_corpus
        from graph_builder import build_index
        from querier import query_all

        print(f"\n{'='*50}\nPhase 3 — GraphRAG\n{'='*50}")
        docs = prepare_corpus(cfg["fsi_enriched"], cfg["graph_dir"])
        build_index(docs, str(cfg["graph_dir"]))
        query_all(str(cfg["graph_dir"]), cfg["phase3_answers"])


def run_phase4(cfg):
    with _use_phase("phase_4"):
        import json
        from schema_builder import build_schema

        print(f"\n{'='*50}\nPhase 4 — Schema Assembly\n{'='*50}")
        records = [json.loads(l) for l in
                   cfg["fsi_enriched"].read_text().splitlines() if l.strip()]
        answers = json.loads(cfg["phase3_answers"].read_text())
        schema  = build_schema(records, answers, cfg)

        cfg["phase4_schema"].write_text(
            json.dumps(schema, indent=2, ensure_ascii=False)
        )
        print(f"✓ Schema written — {len(schema['districts'])} districts")


def run_phase5(cfg):
    with _use_phase("phase_5"):
        import json
        from data_extractor import extract_facts
        from text_synthesizer import synthesize_all
        from image_selector import select_representative_images
        from report_renderer import render_html, save_html
        from pdf_exporter import export_pdf

        print(f"\n{'='*50}\nPhase 5 — Report Generation\n{'='*50}")
        schema  = json.loads(cfg["phase4_schema"].read_text())
        records = [json.loads(l) for l in
                   cfg["fsi_enriched"].read_text().splitlines() if l.strip()]
        answers = json.loads(cfg["phase3_answers"].read_text())

        facts  = extract_facts(records)
        images = select_representative_images(records, max_images=4)
        prose  = synthesize_all(facts, answers, OLLAMA,
                                cfg["city"], cfg["country"],
                                language=cfg.get("language", "en"))
        html   = render_html(schema, facts, prose, images, records=records, cfg=cfg)
        save_html(html, cfg["report_html"])
        export_pdf(cfg["report_html"], cfg["report_pdf"])
        print(f"✓ Report → {cfg['report_html']}")


PHASE_RUNNERS = {
    "0": run_phase0,
    "1": run_phase1,
    "2": run_phase2,
    "3": run_phase3,
    "4": run_phase4,
    "5": run_phase5,
}


if __name__ == "__main__":
    args   = parse_args()
    cfg    = load_city_config(args.city)
    phases = [p.strip() for p in args.phases.split(",")]
    skip   = [p.strip() for p in args.skip.split(",") if args.skip]

    print(f"City    : {cfg['city']}, {cfg['country']}")
    print(f"Phases  : {', '.join(phases)}")
    if skip:
        print(f"Skipping: {', '.join(skip)}")

    for phase in phases:
        if phase in skip:
            print(f"\nSkipping phase {phase}")
            continue
        if phase not in PHASE_RUNNERS:
            print(f"Unknown phase: {phase}")
            continue
        PHASE_RUNNERS[phase](cfg)

    print(f"\n✓ Pipeline complete for {cfg['city']}")