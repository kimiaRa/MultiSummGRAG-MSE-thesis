"""
B1 ablation — {instruction} (the GraphRAG qualitative context block) set to
an empty string for every section, everything else identical to the real
Phase 5 run: same prompt template (text_synthesizer.write, untouched), same
{data_points} built from the existing fsi_enriched.jsonl via extract_facts,
same model/temperature (qwen3:14b, temperature=0, num_ctx=16384, from
config_loader.OLLAMA), same six sections.

Reuses cached artefacts only. Does not call anything from phases 0-4: reads
fsi_enriched.jsonl, phase3_answers.json (loaded but never passed to
synthesize_all — see below), phase4_schema.json and districts.geojson
exactly as run_phase5.py does, and does not re-scrape, rebuild the graph, or
regenerate district polygons.

How {instruction} is forced empty without touching text_synthesizer.py:
synthesize_all() builds instruction per section as
graph_answers.get(<field>, {}).get("answer", ""). Passing graph_answers={}
makes every .get(...) miss and fall back to "" for every section, which is
exactly the ablation condition, using the *exact* prompt-building code the
real pipeline uses (no reimplementation, no risk of prompt drift). The one
section that already hardcodes instruction="" in the unablated pipeline
("digital") is therefore identical in both the pipeline and B1 reports.

Writes only:
  data/<city>/output/report_<city>_b1.html
  data/<city>/output/report_<city>_b1.pdf
  <scratchpad>/b1_ablation/<city>_prose.json   (raw section text, for diffing)
Never touches report_<city>.html/.pdf (the existing pipeline report) or any
phase 0-4 artefact.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "phase_5"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config_loader import OLLAMA, load_city_config
from data_extractor import extract_facts
from text_synthesizer import synthesize_all
from image_selector import select_representative_images
from report_renderer import render_html, save_html
from pdf_exporter import export_pdf

SCRATCH = Path(
    "/tmp/claude-1000/-home-ubuntu-MT-ZHAW/27768811-bdf6-40a9-88f3-b79bdc54fd6f"
    "/scratchpad/b1_ablation"
)


def run(city_name: str):
    cfg = load_city_config(city_name)
    slug = city_name.lower().replace(" ", "_").replace("&", "and")

    print(f"[{city_name}] B1 ablation — loading cached artefacts (no phases 0-4)...")
    schema = json.loads(Path(cfg["phase4_schema"]).read_text(encoding="utf-8"))
    records = [
        json.loads(l)
        for l in Path(cfg["fsi_enriched"]).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    print(f"  {len(records)} FSI records loaded from cached fsi_enriched.jsonl")

    facts = extract_facts(records)
    images = select_representative_images(records, max_images=4)

    print(f"[{city_name}] Synthesizing with {{instruction}}='' for all 6 sections...")
    prose = synthesize_all(
        facts,
        {},  # graph_answers={} -> every section's instruction resolves to ""
        OLLAMA,
        schema["meta"]["city"], schema["meta"]["country"],
        language=cfg.get("language", "en"),
    )

    html = render_html(schema, facts, prose, images, records=records, cfg=cfg)

    b1_html = cfg["out_dir"] / f"report_{slug}_b1.html"
    b1_pdf = cfg["out_dir"] / f"report_{slug}_b1.pdf"
    assert b1_html != cfg["report_html"] and b1_pdf != cfg["report_pdf"], \
        "B1 output path collided with the existing pipeline report path — aborting"

    save_html(html, b1_html)
    export_pdf(b1_html, b1_pdf)

    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / f"{slug}_prose.json").write_text(
        json.dumps(prose, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"✓ [{city_name}] B1 done -> {b1_html}, {b1_pdf}")


if __name__ == "__main__":
    cities = sys.argv[1:] or ["barcelona", "brighton", "dublin", "london", "milan"]
    for c in cities:
        run(c)
