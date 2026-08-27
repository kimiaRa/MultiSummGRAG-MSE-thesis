import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "phase_5"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
from config_loader import OLLAMA
from data_extractor import extract_facts
from text_synthesizer import synthesize_all
from image_selector import select_representative_images
from report_renderer import render_html, save_html
from pdf_exporter import export_pdf


def run(city_name: str):
    from config_loader import load_city_config, build_districts_geojson     
    cfg = load_city_config(city_name)

    print(f"[{city_name}] Loading data...")
    schema = json.loads(
        Path(cfg["phase4_schema"]).read_text(encoding="utf-8")
    )
    records = [
        json.loads(l)
        for l in Path(cfg["fsi_enriched"]).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    graph_answers = json.loads(
        Path(cfg["phase3_answers"]).read_text(encoding="utf-8")
    )
    print(f"  {len(records)} FSI records loaded")

    print(f"[{city_name}] Step 5a — extracting facts from data...")
    facts = extract_facts(records)
    print(f"  ✓ {len(facts)} fact fields extracted")

    print(f"[{city_name}] Step 5b — selecting representative images...")
    images = select_representative_images(records, max_images=4)
    print(f"  ✓ {len(images)} images selected")

    print(f"[{city_name}] Step 5c — synthesizing report prose...")
    prose = synthesize_all(facts, graph_answers, OLLAMA,
                           schema["meta"]["city"], schema["meta"]["country"],
                           language=cfg.get("language", "en"))

    print(f"[{city_name}] Step 5d — rendering HTML...")
    html = render_html(schema, facts, prose, images, records=records, cfg=cfg)
    save_html(html, cfg["report_html"])

    print(f"[{city_name}] Step 5e — exporting PDF...")
    export_pdf(cfg["report_html"], cfg["report_pdf"])

    print(f"\n✓ Phase 5 complete")
    print(f"  HTML → {cfg['report_html']}")
    print(f"  PDF  → {cfg['report_pdf']}")


if __name__ == "__main__":
    import sys as _sys
    city = _sys.argv[1] if len(_sys.argv) > 1 else "dublin"
    run(city)