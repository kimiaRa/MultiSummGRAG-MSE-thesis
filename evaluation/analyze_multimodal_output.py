"""
Multimodal output-capability audit — Full MultiSummGRAG vs. commercial
baseline_v2, existing rendered HTML reports only.

Strictly read-only against the 10 inspected report files:
  data/<city>/output/report_<city>.html
  data/<city>/output/report_<city>_baseline_v2.html
for barcelona, brighton, dublin, london, milan. Every one is opened only in
text-read mode. No pipeline phase, Ollama, or GraphRAG call is reachable
from this script -- it only imports bs4 (HTML parsing) and stdlib.

This is a PRACTICAL OUTPUT-CAPABILITY audit: what each condition's rendered
artefact actually contains and states about itself. It is NOT a test of
factual accuracy, and NOT evidence for GraphRAG's causal contribution to
report quality -- see analysis_summary.md's explicit statement to that
effect.

Full condition: the map/chart/district-grouping determination is drawn
directly from source-code tracing (src/phase_5/map_renderer.py,
chart_renderer.py, data_extractor.py, report_renderer.py), since the
renderer is identical, deterministic code for every city -- not inferred
from any single HTML file's appearance.

baseline_v2 condition: every determination is drawn ONLY from that city's
own stored HTML (alt text, figcaption text, presence/absence of <a href>),
since baseline_v2 has no fixed renderer -- it is independently
LLM-generated per city and its class names are NOT consistent across
cities (verified: "photo"/"figure" for Dublin, "two" for Barcelona,
"hero-card"/"wide-figure" for Brighton, "hero-card"/"figure" for London,
"photo-card"/"chart" for Milan). Chart/map figures are identified instead
by their alt-text prefix ("Coordinate map...", "Bar chart...", "Donut
chart..."), which was found consistent across all five baseline_v2 reports.

Writes only:
  evaluation/results/multimodal_output_audit/report_level.csv
  evaluation/results/multimodal_output_audit/analysis_summary.md
Refuses to overwrite either if it already exists, unless --overwrite is
passed. Hashes all 10 inspected HTML files before and after the run and
aborts without writing if any content differs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = (ROOT / "data").resolve()
OUTPUT_ROOT = (ROOT / "evaluation" / "results" / "multimodal_output_audit").resolve()

CITIES = ["barcelona", "brighton", "dublin", "london", "milan"]
ALLOWED_OUTPUT_NAMES = {"report_level.csv", "analysis_summary.md"}

CHART_ALT_RE = re.compile(r"^\s*(coordinate map|bar chart|donut chart)", re.IGNORECASE)
MAP_ALT_RE = re.compile(r"^\s*coordinate map", re.IGNORECASE)
BARCHART_ALT_RE = re.compile(r"^\s*bar chart", re.IGNORECASE)

assert DATA_ROOT not in OUTPUT_ROOT.parents and OUTPUT_ROOT != DATA_ROOT, \
    "OUTPUT_ROOT must not resolve inside data/"


def _assert_safe_output_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == DATA_ROOT or DATA_ROOT in resolved.parents:
        raise RuntimeError(f"SAFETY ABORT: refusing to write under data/: {resolved}")
    if resolved.parent != OUTPUT_ROOT:
        raise RuntimeError(f"SAFETY ABORT: output path must be a direct child of {OUTPUT_ROOT}: {resolved}")
    if resolved.name not in ALLOWED_OUTPUT_NAMES:
        raise RuntimeError(f"SAFETY ABORT: filename not in the allowed output list: {resolved.name}")
    return resolved


def _report_path(city: str, condition: str) -> Path:
    if condition == "full":
        return DATA_ROOT / city / "output" / f"report_{city}.html"
    return DATA_ROOT / city / "output" / f"report_{city}_baseline_v2.html"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_inspected_reports() -> dict[str, str]:
    hashes = {}
    for city in CITIES:
        for condition in ("full", "baseline_v2"):
            p = _report_path(city, condition)
            hashes[str(p.relative_to(ROOT))] = _sha256_file(p)
    return hashes


def diff_hashes(before: dict[str, str], after: dict[str, str]) -> list[str]:
    return [k for k in before if before.get(k) != after.get(k)]


# ═══════════════════════════════════════════════════════════════════════
# Full condition — traced from source code, not inferred per-file
# ═══════════════════════════════════════════════════════════════════════

def analyse_full_report(city: str) -> dict:
    path = _report_path(city, "full")
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")

    map_box = soup.select_one("div.map-box")
    map_img = map_box.find("img") if map_box else None
    map_caption_el = map_box.find("div", class_="map-caption") if map_box else None

    chart_boxes = soup.select("div.charts-row div.chart-box")
    district_chart = chart_boxes[0] if chart_boxes else None  # template order: district box first
    district_title_el = district_chart.find("div", class_="chart-title") if district_chart else None

    figures = soup.select("div.image-strip figure")
    n_images = len(figures)
    n_with_link = 0
    n_with_literal_url = 0
    for fig in figures:
        cap = fig.find("figcaption")
        a = cap.find("a") if cap else None
        if a and a.get("href"):
            n_with_link += 1
            href = a.get("href").strip()
            link_text = a.get_text().strip()
            if href and href in link_text:
                n_with_literal_url += 1

    return {
        "city": city, "condition": "full",
        "representative_images": n_images,
        "map_present": bool(map_img),
        "map_alt_text": map_img.get("alt", "") if map_img else "",
        "map_caption": map_caption_el.get_text(strip=True) if map_caption_el else "",
        "coordinate_map": False,  # code-traced: map_renderer.py plots district
                                  # polygon choropleth only, never individual
                                  # FSI lat/lng points -- see module docstring
        "administrative_choropleth": bool(map_img),  # code-traced: gpd.read_file(districts_file)
                                                       # polygons coloured by density category
        "district_chart_present": bool(district_chart),
        "district_chart_caption": district_title_el.get_text(strip=True) if district_title_el else "",
        "district_basis": "administrative_boundary_geometry",
        "district_basis_evidence": (
            "src/phase_5/map_renderer.py:render_map() loads cfg['districts_file'] via "
            "geopandas and plots those polygons; src/phase_5/data_extractor.py's "
            "district_counts and the bar chart both key off record['location']"
            "['district_name'], which src/phase_2/geofencer.py assigns via point-in-"
            "polygon against the same districts.geojson -- map and district chart share "
            "one administrative-boundary-derived grouping, not page-stated or coordinate-"
            "only grouping."
        ),
        "images_with_direct_source_link": n_with_link,
        "images_with_literal_url_displayed": n_with_literal_url,
        "notes": (
            "Map alt/caption describe 'FSI locations' but the rendered map is a district-"
            "density choropleth (no individual FSI point markers are plotted) -- see "
            "district_basis_evidence."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════
# baseline_v2 condition — determined only from each report's own HTML
# ═══════════════════════════════════════════════════════════════════════

def analyse_baseline_v2_report(city: str) -> dict:
    path = _report_path(city, "baseline_v2")
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")

    all_figs = soup.find_all("figure")
    chart_figs = []
    photo_figs = []
    for fig in all_figs:
        img = fig.find("img")
        alt = (img.get("alt") or "") if img else ""
        if CHART_ALT_RE.match(alt):
            chart_figs.append((fig, alt))
        else:
            photo_figs.append((fig, alt))

    map_fig = next((fig for fig, alt in chart_figs if MAP_ALT_RE.match(alt)), None)
    district_fig = next((fig for fig, alt in chart_figs if BARCHART_ALT_RE.match(alt)), None)

    def fig_alt(fig):
        img = fig.find("img") if fig else None
        return img.get("alt", "") if img else ""

    def fig_caption(fig):
        cap = fig.find("figcaption") if fig else None
        return cap.get_text(strip=True) if cap else ""

    n_images = len(photo_figs)
    n_with_link = 0
    n_with_literal_url = 0
    for fig, alt in photo_figs:
        cap = fig.find("figcaption")
        a = cap.find("a") if cap else None
        if a and a.get("href"):
            n_with_link += 1
            href = a.get("href").strip()
            link_text = a.get_text().strip()
            if href and href in link_text:
                n_with_literal_url += 1

    district_caption = fig_caption(district_fig)

    return {
        "city": city, "condition": "baseline_v2",
        "representative_images": n_images,
        "map_present": map_fig is not None,
        "map_alt_text": fig_alt(map_fig),
        "map_caption": fig_caption(map_fig),
        "coordinate_map": map_fig is not None,  # per this report's own alt text: "Coordinate map..."
        "administrative_choropleth": False,  # no baseline_v2 report's own text claims
                                              # administrative-boundary polygons; every
                                              # district caption explicitly says otherwise
        "district_chart_present": district_fig is not None,
        "district_chart_caption": district_caption,
        "district_basis": "page_stated_or_coordinate_derived",
        "district_basis_evidence": (
            f"Per this report's own district-chart caption (verbatim): {district_caption!r}"
        ),
        "images_with_direct_source_link": n_with_link,
        "images_with_literal_url_displayed": n_with_literal_url,
        "notes": (
            f"{n_images} representative image figure(s) found (identified by alt text NOT "
            f"matching the coordinate-map/bar-chart/donut-chart pattern; this report's "
            f"figure CSS classes are not comparable across cities -- see module docstring)."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════
# Writing
# ═══════════════════════════════════════════════════════════════════════

CSV_FIELDS = [
    "city", "condition", "representative_images", "map_present", "map_alt_text",
    "map_caption", "coordinate_map", "administrative_choropleth",
    "district_chart_present", "district_chart_caption", "district_basis",
    "district_basis_evidence", "images_with_direct_source_link",
    "images_with_literal_url_displayed", "notes",
]


def write_csv(rows: list[dict], overwrite: bool) -> Path:
    path = _assert_safe_output_path(OUTPUT_ROOT / "report_level.csv")
    if path.exists() and not overwrite:
        raise RuntimeError(f"REFUSING TO OVERWRITE existing output (pass --overwrite to replace): {path}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def write_summary(text: str, overwrite: bool) -> Path:
    path = _assert_safe_output_path(OUTPUT_ROOT / "analysis_summary.md")
    if path.exists() and not overwrite:
        raise RuntimeError(f"REFUSING TO OVERWRITE existing output (pass --overwrite to replace): {path}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def build_summary(rows: list[dict]) -> str:
    md = []
    md.append("# Multimodal Output-Capability Audit — Full vs. baseline_v2\n")
    md.append(
        "Practical output-capability comparison of the existing, already-rendered HTML "
        "reports for both conditions, across all five cities. **This is not a test of "
        "factual accuracy and is not evidence for GraphRAG's causal contribution to report "
        "quality** -- it records only what each stored artefact contains and states about "
        "itself.\n"
    )

    md.append("## City-by-city evidence\n")
    for city in CITIES:
        full = next(r for r in rows if r["city"] == city and r["condition"] == "full")
        base = next(r for r in rows if r["city"] == city and r["condition"] == "baseline_v2")
        md.append(f"### {city}\n")
        md.append(
            f"- **Full**: {full['representative_images']} representative image(s); map "
            f"present={full['map_present']} (alt: {full['map_alt_text']!r}); coordinate "
            f"map={full['coordinate_map']}; administrative choropleth="
            f"{full['administrative_choropleth']}; district chart present="
            f"{full['district_chart_present']}; district basis={full['district_basis']}; "
            f"images with direct source link={full['images_with_direct_source_link']}/"
            f"{full['representative_images']}; images with literal URL displayed="
            f"{full['images_with_literal_url_displayed']}/{full['representative_images']}.\n"
        )
        md.append(
            f"- **baseline_v2**: {base['representative_images']} representative image(s); "
            f"map present={base['map_present']} (alt: {base['map_alt_text']!r}); coordinate "
            f"map={base['coordinate_map']}; administrative choropleth="
            f"{base['administrative_choropleth']}; district chart present="
            f"{base['district_chart_present']}; district basis={base['district_basis']} "
            f"(caption: {base['district_chart_caption']!r}); images with direct source "
            f"link={base['images_with_direct_source_link']}/{base['representative_images']}; "
            f"images with literal URL displayed={base['images_with_literal_url_displayed']}/"
            f"{base['representative_images']}.\n"
        )

    md.append("## Pooled counts\n")
    full_rows = [r for r in rows if r["condition"] == "full"]
    base_rows = [r for r in rows if r["condition"] == "baseline_v2"]

    def pooled(rows_, key):
        return sum(1 for r in rows_ if r[key])

    md.append(
        f"- Map present: Full {pooled(full_rows, 'map_present')}/5, "
        f"baseline_v2 {pooled(base_rows, 'map_present')}/5.\n"
        f"- Administrative-boundary choropleth present: Full "
        f"{pooled(full_rows, 'administrative_choropleth')}/5, baseline_v2 "
        f"{pooled(base_rows, 'administrative_choropleth')}/5.\n"
        f"- Coordinate/point map present: Full {pooled(full_rows, 'coordinate_map')}/5, "
        f"baseline_v2 {pooled(base_rows, 'coordinate_map')}/5.\n"
        f"- District chart present: Full {pooled(full_rows, 'district_chart_present')}/5, "
        f"baseline_v2 {pooled(base_rows, 'district_chart_present')}/5.\n"
        f"- Total representative images: Full "
        f"{sum(r['representative_images'] for r in full_rows)}, baseline_v2 "
        f"{sum(r['representative_images'] for r in base_rows)} (baseline_v2: 0 in "
        f"barcelona).\n"
        f"- Images with a direct source hyperlink: Full "
        f"{sum(r['images_with_direct_source_link'] for r in full_rows)}/"
        f"{sum(r['representative_images'] for r in full_rows)}, baseline_v2 "
        f"{sum(r['images_with_direct_source_link'] for r in base_rows)}/"
        f"{sum(r['representative_images'] for r in base_rows)}.\n"
        f"- Images with the literal URL displayed as visible text (not hidden behind "
        f"generic anchor text): Full "
        f"{sum(r['images_with_literal_url_displayed'] for r in full_rows)}/"
        f"{sum(r['representative_images'] for r in full_rows)}, baseline_v2 "
        f"{sum(r['images_with_literal_url_displayed'] for r in base_rows)}/"
        f"{sum(r['representative_images'] for r in base_rows)}.\n"
    )

    md.append("## Distinctions this audit deliberately preserves\n")
    md.append(
        "- **\"Map exists\" vs. \"administrative choropleth exists\"**: every Full report "
        "has a map, and every one of those maps is an administrative-boundary choropleth "
        "(district polygons coloured by density) -- but the map's own alt text/caption say "
        "'FSI locations', which reads as a point map despite the underlying rendering never "
        "plotting individual coordinates (code-traced, not inferred). Every baseline_v2 "
        "report also has a map, and every one is a genuine coordinate/point map (per its own "
        "stated alt text and caption) -- baseline_v2 has zero administrative-choropleth maps, "
        "by its own account, since it has no district boundary data available to it.\n"
        "- **\"Images exist\" vs. \"image provenance is explicitly traceable\"**: representative "
        "images exist in 4/5 baseline_v2 reports (0 in barcelona) and all 5 Full reports. "
        "Direct source hyperlinks exist in Full (100% of its images) and in "
        "brighton/dublin's baseline_v2 images, but NOT in london's (no `<a>` at all) or "
        "milan's (plain-text attribution, no link) baseline_v2 images.\n"
        "- **Clickable generic links vs. literal URLs displayed to the reader**: Full's "
        "template prints the raw URL itself as the visible link text for every image that "
        "has one. Every baseline_v2 image link found (brighton, dublin) uses generic anchor "
        "text ('Source page'/'source page') with the real URL hidden in the href attribute, "
        "never shown to the reader as text.\n"
    )

    md.append("## What this does NOT establish\n")
    md.append(
        "- This audit does not test or claim that either condition's reported figures, "
        "district assignments, or captions are factually correct.\n"
        "- Neither condition is described as more accurate here -- accuracy was not "
        "independently tested in this audit.\n"
        "- This is a practical output-capability comparison only, and is NOT evidence for "
        "the causal contribution of GraphRAG to report quality -- Full's use of "
        "administrative-boundary geometry and baseline_v2's use of raw coordinates reflects "
        "a difference in available input data (a districts.geojson file vs. none), not "
        "necessarily a difference attributable to GraphRAG itself.\n"
    )

    return "\n".join(md)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Multimodal output-capability audit.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow overwriting existing analysis outputs. Off by default.")
    args = parser.parse_args()

    for city in CITIES:
        for condition in ("full", "baseline_v2"):
            p = _report_path(city, condition)
            if not p.is_file():
                print(f"VALIDATION FAILED — missing report: {p}")
                sys.exit(1)

    hashes_before = hash_inspected_reports()

    rows = []
    for city in CITIES:
        rows.append(analyse_full_report(city))
        rows.append(analyse_baseline_v2_report(city))

    hashes_after = hash_inspected_reports()
    changed = diff_hashes(hashes_before, hashes_after)
    if changed:
        print("INTEGRITY CHECK FAILED — inspected reports changed during this run:")
        for c in changed:
            print(f"  - {c}")
        print("Aborting WITHOUT writing any output.")
        sys.exit(1)

    csv_path = write_csv(rows, args.overwrite)
    summary_text = build_summary(rows)
    md_path = write_summary(summary_text, args.overwrite)

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"\nAll 10 inspected reports: hash unchanged before/after (content-only SHA-256).\n")

    return rows, summary_text, csv_path


if __name__ == "__main__":
    main()
