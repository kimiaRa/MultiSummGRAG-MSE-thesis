#!/usr/bin/env python3
"""
Corpus characterisation analysis — thesis chapter.

Offline, reproducible descriptive analysis of the seed URL corpus and its
transformation through the pipeline, for all cities present in data/ on this
branch (barcelona, brighton, dublin, london, milan). Reads only files already
on disk. Makes no network calls and runs no pipeline stage.

Run from the repo root with the project venv active:
    source venv/bin/activate
    python3 analysis/corpus_analysis.py

Requires: pandas, numpy, matplotlib, geopandas, shapely, pymupdf (fitz),
tldextract, tabulate (all present in venv/ at the time this was written).

Outputs (all regenerated from scratch on every run):
    analysis/tables/*.csv
    analysis/figures/*.png + *.pdf   (300 dpi)
    analysis/corpus_stats.json
    analysis/CORPUS_ANALYSIS.md
"""
from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import fitz  # PyMuPDF
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tldextract
from shapely.geometry import Point

# ─────────────────────────────────────────────────────────────────────────
# Paths and constants
# ─────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ANALYSIS_DIR = ROOT / "analysis"
TABLES_DIR = ANALYSIS_DIR / "tables"
FIGURES_DIR = ANALYSIS_DIR / "figures"
STATS_PATH = ANALYSIS_DIR / "corpus_stats.json"
MD_PATH = ANALYSIS_DIR / "CORPUS_ANALYSIS.md"
APPENDIX_REPORTS = ROOT / "appendix_reports"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CITIES = ["barcelona", "brighton", "dublin", "london", "milan"]

# Per-city raw-seed file. Barcelona has no raw_urls.csv on this branch (the
# `multilingual` branch); its urls.csv was verified byte-identical to the
# `main` branch's data/barcelona/raw_urls.csv via `git show main:...` before
# this script was written (see Section 1 notes / CORPUS_ANALYSIS.md). Per
# explicit user confirmation, urls.csv is used as Barcelona's raw-seed proxy.
RAW_FILE = {
    "barcelona": DATA / "barcelona" / "urls.csv",
    "brighton": DATA / "brighton" / "raw_urls.csv",
    "dublin": DATA / "dublin" / "raw_urls.csv",
    "london": DATA / "london" / "raw_urls.csv",
    "milan": DATA / "milan" / "raw_urls.csv",
}
RAW_IS_PROXY = {c: (c == "barcelona") for c in CITIES}

CLEANED_FILE = {c: DATA / c / "urls_cleaned.csv" for c in CITIES}

# urls.csv is produced by a function in src/phase_0/cleaner.py that is
# entirely commented out (dead code) — not written by the current pipeline.
# Where present it is a stale/superseded intermediate, not a ledger stage.
# (For Barcelona it is being reused above as the raw-seed proxy instead.)
SUPERSEDED_URLS_CSV = {
    "barcelona": None,
    "brighton": DATA / "brighton" / "urls.csv",
    "dublin": DATA / "dublin" / "urls.csv",
    "london": None,
    "milan": None,
}

RAW_DIR = {c: DATA / c / "raw" for c in CITIES}
GRAPHRAG_INPUT_DIR = {c: DATA / c / "graphrag" / "input" for c in CITIES}
FSI_ENRICHED = {c: DATA / c / "output" / "fsi_enriched.jsonl" for c in CITIES}
FSI_RECORDS = {c: DATA / c / "output" / "fsi_records.jsonl" for c in CITIES}
REPORT_HTML = {c: DATA / c / "output" / f"report_{c}.html" for c in CITIES}
DISTRICTS_GEOJSON = {c: DATA / c / "districts.geojson" for c in CITIES}
BASELINE_PDF = {c: APPENDIX_REPORTS / f"report_{c}_baseline.pdf" for c in CITIES}

EXCLUDED_DIRS_NOTE = (
    "NOTUSE/ and NOTUSED_dataset_comparison/ are excluded from this analysis: "
    "both are named by the repository owner as not-in-use, and contain an "
    "alternate GPT-sourced dataset for a separate comparison arm, not the "
    "corpus this chapter characterises."
)

SOCIAL_DOMAINS = {
    "facebook.com", "instagram.com", "twitter.com", "x.com",
    "linkedin.com", "youtube.com", "youtu.be", "tiktok.com",
}
NON_HTML_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".csv", ".zip", ".jpg", ".jpeg", ".png",
}

EARTH_RADIUS_M = 6_371_008.8  # mean Earth radius, WGS84 authalic sphere
METERS_PER_DEGREE_LAT = 111_320.0  # ~constant across latitudes

_TLDX = tldextract.TLDExtract(suffix_list_urls=())  # offline: bundled snapshot only

# ─────────────────────────────────────────────────────────────────────────
# Accumulators: markdown report, scalar stats, anomalies
# ─────────────────────────────────────────────────────────────────────────

MD: list[str] = []
STATS: dict = {}
ANOMALIES: list[dict] = []


def md(text: str = "") -> None:
    MD.append(text)


def add_anomaly(section: str, summary: str, evidence: str) -> None:
    ANOMALIES.append({"section": section, "summary": summary, "evidence": evidence})


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def save_table(df: pd.DataFrame, name: str, title: str) -> pd.DataFrame:
    """Write CSV to analysis/tables/, append a markdown table to the report."""
    csv_path = TABLES_DIR / f"{name}.csv"
    df.to_csv(csv_path, index=False)
    md(f"\n**Table: {title}** (`tables/{name}.csv`)\n")
    md(df.to_markdown(index=False))
    md("")
    print(f"\n=== {title} ===")
    print(df.to_markdown(index=False))
    return df


def save_fig(fig, name: str, caption: str) -> None:
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    md(f"\n![{caption}](figures/{name}.png)\n")
    md(f"*{caption}*\n")
    print(f"Saved figure: {name}.png / .pdf")


# ─────────────────────────────────────────────────────────────────────────
# Generic helpers
# ─────────────────────────────────────────────────────────────────────────

def read_url_csv(path: Path) -> pd.DataFrame:
    """Read a raw/cleaned URL CSV as strings (no float coercion), returning
    columns: url, lat_raw, lng_raw, lat, lng, coords_approx.
    Handles the Lat/Lng vs Lat/Lon header variant (london's raw_urls.csv)."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        lng_col = "Lng" if "Lng" in fieldnames else ("Lon" if "Lon" in fieldnames else None)
        rows = []
        for row in reader:
            url = (row.get("URL") or "").strip()
            if not url:
                continue
            lat_raw = (row.get("Lat") or "").strip()
            lng_raw = (row.get(lng_col) or "").strip() if lng_col else ""
            approx = row.get("coords_approx")
            rows.append({
                "url": url,
                "lat_raw": lat_raw if lat_raw != "" else None,
                "lng_raw": lng_raw if lng_raw != "" else None,
                "coords_approx": approx.strip() if isinstance(approx, str) else None,
            })
    df = pd.DataFrame(rows, columns=["url", "lat_raw", "lng_raw", "coords_approx"])
    df["lat"] = pd.to_numeric(df["lat_raw"], errors="coerce")
    df["lng"] = pd.to_numeric(df["lng_raw"], errors="coerce")
    return df


def normalize_url(u: str) -> str:
    """scheme-insensitive, www-stripped, single-trailing-slash-stripped,
    lowercased host+path(+query). Documented, deterministic, not a guess."""
    u = u.strip()
    try:
        p = urlparse(u)
    except ValueError:
        return u.lower()
    netloc = p.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = p.path
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    out = netloc + path
    if p.query:
        out += "?" + p.query
    return out.lower()


def registrable_domain(u: str) -> str:
    ext = _TLDX(u)
    if ext.suffix:
        return f"{ext.domain}.{ext.suffix}".lower()
    return (ext.domain or "").lower()


def path_depth(u: str) -> int:
    path = urlparse(u).path
    segments = [s for s in path.split("/") if s]
    return len(segments)


def url_extension(u: str) -> str:
    path = urlparse(u).path
    return Path(path).suffix.lower()


def decimal_places(raw: str | None) -> int | None:
    if raw is None:
        return None
    raw = raw.strip()
    if raw == "":
        return None
    if "." not in raw:
        return 0
    return len(raw.split(".", 1)[1])


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def herfindahl(counts: pd.Series) -> float:
    """Sum of squared market shares, 0-1 scale (multiply by 10000 for the
    classic 0-10000 HHI scale)."""
    total = counts.sum()
    if total == 0:
        return float("nan")
    shares = counts / total
    return float((shares ** 2).sum())


def city_bbox(city: str, margin_frac: float = 0.25) -> tuple[float, float, float, float]:
    """Generous bounding box derived from districts.geojson total_bounds,
    buffered by margin_frac of width/height on each side (file-derived, not
    a hand-typed constant)."""
    gdf = gpd.read_file(DISTRICTS_GEOJSON[city])
    minx, miny, maxx, maxy = gdf.total_bounds
    dx = (maxx - minx) * margin_frac
    dy = (maxy - miny) * margin_frac
    return (minx - dx, miny - dy, maxx + dx, maxy + dy)


def in_bbox(lat, lon, bbox) -> bool:
    minx, miny, maxx, maxy = bbox
    return (minx <= lon <= maxx) and (miny <= lat <= maxy)


def coarse_geo_reasoning(lat: float, lon: float) -> str:
    """Offline, coarse hemisphere/region reasoning from coordinate magnitude
    alone -- general geographic knowledge (equator/meridian/latitude bands),
    not a geocoding lookup of a specific address."""
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return "coordinate value(s) out of valid range for lat/lon"
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    if abs(lat) < 1 and abs(lon) < 1:
        return "~(0,0): Gulf of Guinea, off the coast of West Africa -- classic null-island artifact"
    band = (
        "polar" if abs(lat) > 66.5 else
        "high-latitude" if abs(lat) > 50 else
        "temperate" if abs(lat) > 35 else
        "sub-tropical" if abs(lat) > 23.5 else
        "tropical"
    )
    region = "Europe/North Africa/Middle East longitudes" if -25 <= lon <= 60 else (
        "Americas longitudes" if lon < -25 else "Asia-Pacific longitudes"
    )
    return f"{abs(lat):.2f} deg {ns}, {abs(lon):.2f} deg {ew} -- {band} latitude band, {region}"


print("=" * 70)
print("CORPUS CHARACTERISATION ANALYSIS")
print("=" * 70)

md("# Corpus Characterisation Analysis")
md("")
md(
    "Descriptive analysis of the seed URL corpus and its transformation "
    "through the pipeline, computed entirely from files on disk (no pipeline "
    "stage was run, no network call was made to produce these numbers). "
    "Generated by `analysis/corpus_analysis.py`; rerun that script to "
    "regenerate every table, figure and number in this document from "
    "scratch."
)
md("")
md(f"Cities analysed: {', '.join(CITIES)}. {EXCLUDED_DIRS_NOTE}")

# ═══════════════════════════════════════════════════════════════════════
# SECTION 1 — Inventory of the CSV lineage
# ═══════════════════════════════════════════════════════════════════════
md("\n## 1. Data inventory\n")
md(
    "Every CSV found under `data/<city>/` for the five active cities, its "
    "row count, columns, dtypes and modification time, and which pipeline "
    "stage it corresponds to. Classification is evidence-based: it draws on "
    "`config/<city>.yaml` (`raw_urls_file` / `urls_file` keys), on "
    "`src/phase_0/cleaner.py` (the function that would write `urls.csv` is "
    "entirely commented out -- dead code, not run by the current pipeline), "
    "and on a byte-level `diff` against `raw_urls.csv` where both files "
    "exist for a city."
)

inventory_rows = []


def _inventory_row(city: str, path: Path, stage: str) -> None:
    if not path.exists():
        inventory_rows.append({
            "city": city, "path": str(path.relative_to(ROOT)), "stage": stage,
            "row_count": None, "n_columns": None, "columns": None, "dtypes": None,
            "mtime": None, "exists": False,
        })
        return
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    # second pass, type-inferring, purely to report dtypes (not used for values)
    df_typed = pd.read_csv(path)
    mtime = pd.Timestamp(path.stat().st_mtime, unit="s").isoformat()
    inventory_rows.append({
        "city": city,
        "path": str(path.relative_to(ROOT)),
        "stage": stage,
        "row_count": len(df),
        "n_columns": len(df.columns),
        "columns": ";".join(df.columns),
        "dtypes": ";".join(f"{c}:{t}" for c, t in df_typed.dtypes.astype(str).items()),
        "mtime": mtime,
        "exists": True,
    })


for city in CITIES:
    raw_path = RAW_FILE[city]
    stage = "RAW_SEED (proxy: no raw_urls.csv on this branch; verified byte-identical " \
            "to main branch's raw_urls.csv via git history)" if RAW_IS_PROXY[city] else "RAW_SEED"
    _inventory_row(city, raw_path, stage)

    superseded = SUPERSEDED_URLS_CSV[city]
    if superseded is not None:
        _inventory_row(
            city, superseded,
            "SUPERSEDED_INTERMEDIATE (urls.csv writer is dead code in "
            "src/phase_0/cleaner.py; not consumed by current pipeline; "
            "excluded from the attrition ledger)"
        )

    _inventory_row(
        city, CLEANED_FILE[city],
        "CLEANED (post dedup + dead-link/social-wall removal, combined -- "
        "no intermediate file on disk separates these two sub-stages)"
    )

inventory_df = pd.DataFrame(inventory_rows)
save_table(inventory_df, "01_inventory", "CSV lineage inventory")

for city in CITIES:
    if RAW_IS_PROXY[city]:
        add_anomaly(
            "1. Inventory",
            f"{city}: no raw_urls.csv on disk on this branch (multilingual)",
            "config/barcelona.yaml declares raw_urls_file: data/barcelona/raw_urls.csv, "
            "but that file does not exist in the working tree on this branch. "
            "`git show main:data/barcelona/raw_urls.csv` (commit d80cd64) is 0-diff "
            "byte-identical to the current data/barcelona/urls.csv (262 lines incl. "
            "header). urls.csv is used here as the raw-seed proxy for Barcelona only, "
            "per user confirmation."
        )

STATS["inventory"] = inventory_df.to_dict(orient="records")

# ═══════════════════════════════════════════════════════════════════════
# SECTION 2 — Attrition ledger
# ═══════════════════════════════════════════════════════════════════════
md("\n## 2. Attrition ledger\n")
md(
    "Per-city record survival across pipeline stages. Two caveats, "
    "established from the files themselves rather than assumed:\n\n"
    "1. **Dedup and dead-link/social-wall removal are reported as one "
    "combined stage.** Only one cleaned file exists per city "
    "(`urls_cleaned.csv`); no file on disk separates 'post-dedup' from "
    "'post-dead-link-removal'. `data/<city>/output/phase0_annotated.json` "
    "reports `duplicates_detected: 0` for every city, but its own "
    "`total_input` field matches the *raw* count for Brighton/Dublin and "
    "the *cleaned* count for Barcelona/London/Milan -- i.e. it was computed "
    "at a different pipeline point per city, so `duplicates_detected` is "
    "not a reliable per-city dedup count either (see Data quality issues).\n"
    "2. **'Successfully scraped' (`data/<city>/raw/` file count) is not on "
    "the same unit as the URL-row stages above it.** Filenames in that "
    "directory (e.g. multiple `abarkacoop.com_*.html` variants) show it "
    "holds a multi-page crawl per seed domain, not one file per seed URL, "
    "so its retention percentage against the URL-row stages is not a "
    "meaningful comparison; it is reported anyway per the anomaly-reporting "
    "rule, flagged rather than omitted.\n"
    "3. **Merged GraphRAG input docs is a parallel branch, not a strict "
    "funnel input to `fsi_enriched.jsonl`.** `fsi_records.jsonl` (pre-"
    "enrichment) equals `fsi_enriched.jsonl` exactly in every city, and "
    "both are consistently *larger* than the graphrag input doc count -- "
    "so FSI records are not simply 1:1 derived from the graphrag-merged "
    "document set."
)

FSI_TOTAL_RE = re.compile(r"([\d,]+)\s+initiatives\s+across\s+(\d+)\s+districts")


def _count_lines(path: Path) -> int | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _count_files(path: Path, pattern: str = "*") -> int | None:
    if not path.exists():
        return None
    return len(list(path.glob(pattern)))


def extract_reported_total(city: str) -> tuple[int | None, int | None]:
    path = REPORT_HTML[city]
    if not path.exists():
        return None, None
    text = path.read_text(encoding="utf-8", errors="replace")
    m = FSI_TOTAL_RE.search(text)
    if not m:
        return None, None
    return int(m.group(1).replace(",", "")), int(m.group(2))


ledger_rows = []
reported_totals = {}
reported_district_counts = {}

for city in CITIES:
    raw_n = len(read_url_csv(RAW_FILE[city])) if RAW_FILE[city].exists() else None
    cleaned_df = read_url_csv(CLEANED_FILE[city]) if CLEANED_FILE[city].exists() else None
    cleaned_n = len(cleaned_df) if cleaned_df is not None else None
    coords_n = int(((cleaned_df["lat"].notna()) & (cleaned_df["lng"].notna())).sum()) \
        if cleaned_df is not None else None
    scraped_n = _count_files(RAW_DIR[city])
    graphrag_n = _count_files(GRAPHRAG_INPUT_DIR[city], "*.txt")
    enriched_n = _count_lines(FSI_ENRICHED[city])
    reported_n, district_n = extract_reported_total(city)
    reported_totals[city] = reported_n
    reported_district_counts[city] = district_n

    def pct(num, den):
        if num is None or den in (None, 0):
            return None
        return round(100 * num / den, 1)

    row = {
        "city": city,
        "1_raw_seed": raw_n,
        "2_cleaned": cleaned_n,
        "3_usable_coords": coords_n,
        "4_scraped_files": scraped_n,
        "5_graphrag_docs": graphrag_n,
        "6_fsi_enriched": enriched_n,
        "7_reported_fsi_total": reported_n,
        "pct_raw_to_cleaned": pct(cleaned_n, raw_n),
        "pct_cleaned_to_coords": pct(coords_n, cleaned_n),
        "pct_cleaned_to_scraped": pct(scraped_n, cleaned_n),
        "pct_cleaned_to_graphrag": pct(graphrag_n, cleaned_n),
        "pct_cleaned_to_enriched": pct(enriched_n, cleaned_n),
        "pct_end_to_end_raw_to_enriched": pct(enriched_n, raw_n),
        "pct_end_to_end_raw_to_reported": pct(reported_n, raw_n),
    }
    ledger_rows.append(row)

ledger_df = pd.DataFrame(ledger_rows)
save_table(ledger_df, "02_attrition_ledger", "Attrition ledger: records surviving each stage")

# Flags: any stage-to-stage increase; any reported != enriched mismatch
for row in ledger_rows:
    city = row["city"]
    stage_pairs = [
        ("2_cleaned", "1_raw_seed"), ("3_usable_coords", "2_cleaned"),
        ("4_scraped_files", "2_cleaned"), ("5_graphrag_docs", "2_cleaned"),
        ("6_fsi_enriched", "5_graphrag_docs"), ("6_fsi_enriched", "2_cleaned"),
        ("7_reported_fsi_total", "6_fsi_enriched"),
    ]
    for later, earlier in stage_pairs:
        a, b = row[later], row[earlier]
        if a is not None and b is not None and a > b:
            add_anomaly(
                "2. Attrition ledger",
                f"{city}: count INCREASES from {earlier} ({b}) to {later} ({a})",
                f"analysis/tables/02_attrition_ledger.csv, row city={city}"
            )
    if row["4_scraped_files"] == 0:
        add_anomaly(
            "2. Attrition ledger",
            f"{city}: data/{city}/raw/ contains 0 files -- 'successfully scraped' stage is empty",
            f"data/{city}/raw/ (directory empty or missing) vs {row['2_cleaned']} cleaned URLs"
        )
    if row["7_reported_fsi_total"] is not None and row["6_fsi_enriched"] is not None \
            and row["7_reported_fsi_total"] != row["6_fsi_enriched"]:
        add_anomaly(
            "2. Attrition ledger",
            f"{city}: reported FSI total ({row['7_reported_fsi_total']}) != "
            f"fsi_enriched.jsonl line count ({row['6_fsi_enriched']})",
            f"data/{city}/output/report_{city}.html vs data/{city}/output/fsi_enriched.jsonl"
        )

STATS["attrition_ledger"] = ledger_rows
STATS["reported_totals"] = reported_totals
STATS["reported_district_counts"] = reported_district_counts

# ═══════════════════════════════════════════════════════════════════════
# SECTION 3 — Per-city descriptive statistics (on the raw-seed file)
# ═══════════════════════════════════════════════════════════════════════
md("\n## 3. Per-city descriptive statistics\n")
md(
    "Computed on each city's raw-seed file (see Section 1 for which file "
    "that is per city). `normalize_url()` lowercases scheme/host, strips a "
    "leading `www.`, strips one trailing slash from the path, and keeps the "
    "query string -- see the function docstring in the script."
)

RAW_DF = {c: read_url_csv(RAW_FILE[c]) for c in CITIES if RAW_FILE[c].exists()}
for c, df in RAW_DF.items():
    df["url_norm"] = df["url"].map(normalize_url)
    df["domain"] = df["url"].map(registrable_domain)
    df["depth"] = df["url"].map(path_depth)
    df["ext"] = df["url"].map(url_extension)
    df["is_social"] = df["domain"].isin(SOCIAL_DOMAINS)
    df["is_nonhtml"] = df["ext"].isin(NON_HTML_EXTENSIONS)
    df["tld"] = df["url"].map(lambda u: _TLDX(u).suffix.lower())

desc_rows = []
domain_concentration_curves = {}  # city -> sorted cumulative share array, for fig 5

for c, df in RAW_DF.items():
    n = len(df)
    n_unique_raw = df["url"].nunique()
    n_unique_norm = df["url_norm"].nunique()
    dup_raw = n - n_unique_raw
    dup_norm = n - n_unique_norm

    domain_counts = df["domain"].value_counts()
    hhi = herfindahl(domain_counts)
    total_dom = domain_counts.sum()
    share_top1 = domain_counts.iloc[0] / total_dom if len(domain_counts) else float("nan")
    share_top3 = domain_counts.iloc[:3].sum() / total_dom if len(domain_counts) else float("nan")
    share_top10 = domain_counts.iloc[:10].sum() / total_dom if len(domain_counts) else float("nan")
    n_singleton_domains = int((domain_counts == 1).sum())

    domain_concentration_curves[c] = (domain_counts.values / total_dom).cumsum()

    social_share = df["is_social"].mean()
    nonhtml_share = df["is_nonhtml"].mean()
    homepage_share = (df["depth"] == 0).mean()

    desc_rows.append({
        "city": c,
        "n_rows": n,
        "n_unique_urls_raw": n_unique_raw,
        "n_unique_urls_normalised": n_unique_norm,
        "duplicate_urls_raw": dup_raw,
        "duplicate_urls_normalised": dup_norm,
        "n_unique_domains": df["domain"].nunique(),
        "herfindahl_index_0_1": round(hhi, 4),
        "herfindahl_index_0_10000": round(hhi * 10000, 1),
        "domain_share_top1_pct": round(share_top1 * 100, 1),
        "domain_share_top3_pct": round(share_top3 * 100, 1),
        "domain_share_top10_pct": round(share_top10 * 100, 1),
        "n_singleton_domains": n_singleton_domains,
        "social_platform_share_pct": round(social_share * 100, 1),
        "pdf_or_nonhtml_share_pct": round(nonhtml_share * 100, 1),
        "homepage_share_pct": round(homepage_share * 100, 1),
        "median_path_depth": float(df["depth"].median()),
    })

# combined row across all cities pooled together
all_raw = pd.concat([df.assign(city=c) for c, df in RAW_DF.items()], ignore_index=True)
domain_counts_all = all_raw["domain"].value_counts()
hhi_all = herfindahl(domain_counts_all)
desc_rows.append({
    "city": "ALL_CITIES_POOLED",
    "n_rows": len(all_raw),
    "n_unique_urls_raw": all_raw["url"].nunique(),
    "n_unique_urls_normalised": all_raw["url_norm"].nunique(),
    "duplicate_urls_raw": len(all_raw) - all_raw["url"].nunique(),
    "duplicate_urls_normalised": len(all_raw) - all_raw["url_norm"].nunique(),
    "n_unique_domains": all_raw["domain"].nunique(),
    "herfindahl_index_0_1": round(hhi_all, 4),
    "herfindahl_index_0_10000": round(hhi_all * 10000, 1),
    "domain_share_top1_pct": round(domain_counts_all.iloc[0] / domain_counts_all.sum() * 100, 1),
    "domain_share_top3_pct": round(domain_counts_all.iloc[:3].sum() / domain_counts_all.sum() * 100, 1),
    "domain_share_top10_pct": round(domain_counts_all.iloc[:10].sum() / domain_counts_all.sum() * 100, 1),
    "n_singleton_domains": int((domain_counts_all == 1).sum()),
    "social_platform_share_pct": round(all_raw["is_social"].mean() * 100, 1),
    "pdf_or_nonhtml_share_pct": round(all_raw["is_nonhtml"].mean() * 100, 1),
    "homepage_share_pct": round((all_raw["depth"] == 0).mean() * 100, 1),
    "median_path_depth": float(all_raw["depth"].median()),
})

desc_df = pd.DataFrame(desc_rows)
save_table(desc_df, "03_per_city_descriptive_stats", "Per-city descriptive statistics (raw-seed file)")
md(
    "*`pdf_or_nonhtml_share_pct` is a URL-extension proxy only (`.pdf`, "
    "`.docx`, image extensions, etc. in the path) -- actual Content-Type "
    "cannot be checked without a network request, which this script does "
    "not make; this is a lower bound.*"
)

# TLD distribution and a language-of-source proxy
LOCAL_TLD_BY_CITY = {
    "barcelona": {"es", "cat", "eu"},
    "brighton": {"uk", "co.uk", "org.uk", "gov.uk", "sch.uk"},
    "dublin": {"ie"},
    "london": {"uk", "co.uk", "org.uk", "gov.uk", "sch.uk"},
    "milan": {"it"},
}
tld_rows = []
for c, df in RAW_DF.items():
    tld_counts = df["tld"].value_counts()
    local = LOCAL_TLD_BY_CITY[c]
    n = len(df)
    n_local = int(df["tld"].isin(local).sum())
    n_generic = int(df["tld"].isin({"com", "org", "net", "info", "coop"}).sum())
    n_other_country = n - n_local - n_generic
    for tld, cnt in tld_counts.items():
        tld_rows.append({"city": c, "tld": tld, "count": int(cnt), "share_pct": round(100 * cnt / n, 1)})
    STATS.setdefault("language_proxy", {})[c] = {
        "n": n, "n_local_tld": n_local, "n_generic_tld": n_generic,
        "n_other_country_tld": n_other_country,
        "local_tld_share_pct": round(100 * n_local / n, 1),
        "generic_tld_share_pct": round(100 * n_generic / n, 1),
        "other_country_tld_share_pct": round(100 * n_other_country / n, 1),
        "note": f"local TLD set for {c}: {sorted(local)}",
    }

tld_df = pd.DataFrame(tld_rows).sort_values(["city", "count"], ascending=[True, False])
save_table(tld_df, "03b_tld_distribution", "TLD distribution per city")
lang_proxy_df = pd.DataFrame(STATS["language_proxy"]).T.reset_index().rename(columns={"index": "city"})
save_table(lang_proxy_df, "03c_language_of_source_proxy",
           "Language-of-source proxy: local vs generic vs other-country TLD share")
md(
    "*Language-of-source proxy is TLD-based, not a language-detection result: "
    "`local` = the city/country's own ccTLD family (defined in "
    "`LOCAL_TLD_BY_CITY` in the script), `generic` = .com/.org/.net/.info/.coop, "
    "`other_country` = any other ccTLD. It is a coarse proxy, not a measured fact "
    "about page language.*"
)

# path depth distribution table
depth_rows = []
for c, df in RAW_DF.items():
    vc = df["depth"].value_counts().sort_index()
    for depth, cnt in vc.items():
        depth_rows.append({"city": c, "path_depth": int(depth), "count": int(cnt),
                            "share_pct": round(100 * cnt / len(df), 1)})
depth_df = pd.DataFrame(depth_rows)
save_table(depth_df, "03d_path_depth_distribution", "URL path depth distribution per city")

STATS["per_city_descriptive"] = desc_rows

# ═══════════════════════════════════════════════════════════════════════
# SECTION 4 — Coordinate quality
# ═══════════════════════════════════════════════════════════════════════
md("\n## 4. Coordinate quality\n")

CLEANED_DF = {c: read_url_csv(CLEANED_FILE[c]) for c in CITIES if CLEANED_FILE[c].exists()}

# --- 4a. Coverage --------------------------------------------------------
md("\n### 4a. Coordinate coverage\n")
coverage_rows = []
for c, df in CLEANED_DF.items():
    n = len(df)
    n_complete = int(((df["lat"].notna()) & (df["lng"].notna())).sum())
    coverage_rows.append({
        "city": c, "n_rows": n, "n_complete_pairs": n_complete,
        "coverage_pct": round(100 * n_complete / n, 1) if n else None,
    })
coverage_df = pd.DataFrame(coverage_rows)
save_table(coverage_df, "04a_coordinate_coverage", "Coordinate coverage per city (cleaned corpus)")

# --- 4b. Precision ---------------------------------------------------------
md("\n### 4b. Coordinate decimal precision\n")
precision_rows = []
precision_all_dp = []  # for the histogram figure
for c, df in CLEANED_DF.items():
    have = df[(df["lat_raw"].notna()) & (df["lng_raw"].notna())].copy()
    have["lat_dp"] = have["lat_raw"].map(decimal_places)
    have["lng_dp"] = have["lng_raw"].map(decimal_places)
    dp_counts = pd.concat([have["lat_dp"], have["lng_dp"]]).value_counts()
    modal_dp = int(dp_counts.idxmax()) if len(dp_counts) else None
    mean_lat = have["lat"].mean()
    if modal_dp is not None:
        lat_res_m = (10 ** -modal_dp) * METERS_PER_DEGREE_LAT
        lon_res_m = (10 ** -modal_dp) * METERS_PER_DEGREE_LAT * math.cos(math.radians(mean_lat))
    else:
        lat_res_m = lon_res_m = None
    precision_rows.append({
        "city": c,
        "n_with_coords": len(have),
        "modal_decimal_places": modal_dp,
        "modal_dp_share_pct": round(100 * dp_counts.max() / dp_counts.sum(), 1) if len(dp_counts) else None,
        "mean_latitude_used_for_conversion": round(mean_lat, 4) if not math.isnan(mean_lat) else None,
        "resolution_m_lat_direction": round(lat_res_m, 1) if lat_res_m is not None else None,
        "resolution_m_lon_direction": round(lon_res_m, 1) if lon_res_m is not None else None,
    })
    precision_all_dp.append((c, dp_counts))
precision_df = pd.DataFrame(precision_rows)
save_table(precision_df, "04b_coordinate_precision", "Modal coordinate decimal precision and implied ground resolution")
md(
    "*Resolution = 10^-modal_dp degrees, converted to metres using "
    "111,320 m/degree latitude (~constant) and 111,320 * cos(mean latitude) "
    "m/degree longitude. This is the size of the smallest distinguishable "
    "ground offset at the modal precision -- it bounds what district-level "
    "assignment can resolve.*"
)

# --- 4c. Granularity collapse ---------------------------------------------
md("\n### 4c. Granularity collapse\n")
granularity_rows = []
cluster_size_dist = {}  # city -> Series of cluster sizes, for fig 4
for c, df in CLEANED_DF.items():
    have = df[(df["lat"].notna()) & (df["lng"].notna())].copy()
    have["coord_key"] = have["lat_raw"] + "," + have["lng_raw"]
    cluster_sizes = have["coord_key"].value_counts()
    cluster_size_dist[c] = cluster_sizes
    n_rows_with_coords = len(have)
    n_distinct_coords = have["coord_key"].nunique()
    largest_cluster = int(cluster_sizes.max()) if len(cluster_sizes) else None
    n_on_shared_coord = int((cluster_sizes[cluster_sizes > 1]).sum())
    granularity_rows.append({
        "city": c,
        "n_rows_with_coords": n_rows_with_coords,
        "n_distinct_coordinate_pairs": n_distinct_coords,
        "largest_cluster_size": largest_cluster,
        "n_rows_on_a_shared_coordinate": n_on_shared_coord,
        "pct_rows_on_a_shared_coordinate": round(100 * n_on_shared_coord / n_rows_with_coords, 1)
        if n_rows_with_coords else None,
    })
granularity_df = pd.DataFrame(granularity_rows)
save_table(granularity_df, "04c_granularity_collapse", "Coordinate granularity collapse per city")

for row in granularity_rows:
    if row["largest_cluster_size"] and row["largest_cluster_size"] >= 5:
        add_anomaly(
            "4c. Granularity collapse",
            f"{row['city']}: {row['largest_cluster_size']} rows share one identical coordinate pair",
            f"analysis/tables/04c_granularity_collapse.csv, row city={row['city']}"
        )

# --- 4d. Plausibility (bounding box) ---------------------------------------
md("\n### 4d. Plausibility: bounding-box outliers\n")
bboxes = {c: city_bbox(c) for c in CITIES if DISTRICTS_GEOJSON[c].exists()}
bbox_table_rows = [{"city": c, "min_lon": b[0], "min_lat": b[1], "max_lon": b[2], "max_lat": b[3]}
                    for c, b in bboxes.items()]
save_table(pd.DataFrame(bbox_table_rows), "04d_city_bounding_boxes",
           "Generous per-city bounding boxes (districts.geojson total_bounds + 25% margin)")

outlier_rows = []
for c, df in CLEANED_DF.items():
    if c not in bboxes:
        continue
    have = df[(df["lat"].notna()) & (df["lng"].notna())].copy()
    for _, r in have.iterrows():
        if not in_bbox(r["lat"], r["lng"], bboxes[c]):
            other_city_match = None
            for c2, b2 in bboxes.items():
                if c2 != c and in_bbox(r["lat"], r["lng"], b2):
                    other_city_match = c2
                    break
            outlier_rows.append({
                "city": c, "url": r["url"], "lat": r["lat"], "lng": r["lng"],
                "coords_approx": r["coords_approx"],
                "falls_inside_bbox_of": other_city_match,
                "coarse_geo_reasoning": coarse_geo_reasoning(r["lat"], r["lng"]),
            })
outlier_df = pd.DataFrame(outlier_rows)
save_table(outlier_df, "04d_bounding_box_outliers", "Rows outside their city's generous bounding box")
for _, r in outlier_df.iterrows():
    other_city = r["falls_inside_bbox_of"]
    other_city = "none of the other 4 listed cities' bboxes" if pd.isna(other_city) else other_city
    add_anomaly(
        "4d. Plausibility",
        f"{r['city']}: out-of-bbox point at ({r['lat']}, {r['lng']}) -- url={r['url']}",
        f"falls_inside_bbox_of={other_city}; {r['coarse_geo_reasoning']}"
    )

# --- 4e. Nearest-neighbour distances ----------------------------------------
md("\n### 4e. Nearest-neighbour distances (haversine, distinct points)\n")
nn_rows = []
for c, df in CLEANED_DF.items():
    have = df[(df["lat"].notna()) & (df["lng"].notna())].copy()
    pts = have.drop_duplicates(subset=["lat_raw", "lng_raw"])[["lat", "lng"]].to_numpy()
    n = len(pts)
    if n < 2:
        nn_rows.append({"city": c, "n_distinct_points": n, "median_nn_m": None, "iqr_low_m": None, "iqr_high_m": None})
        continue
    min_dists = []
    for i in range(n):
        best = None
        for j in range(n):
            if i == j:
                continue
            d = haversine_m(pts[i][0], pts[i][1], pts[j][0], pts[j][1])
            if best is None or d < best:
                best = d
        min_dists.append(best)
    min_dists = np.array(min_dists)
    nn_rows.append({
        "city": c, "n_distinct_points": n,
        "median_nn_m": round(float(np.median(min_dists)), 1),
        "iqr_low_m": round(float(np.percentile(min_dists, 25)), 1),
        "iqr_high_m": round(float(np.percentile(min_dists, 75)), 1),
    })
nn_df = pd.DataFrame(nn_rows)
save_table(nn_df, "04e_nearest_neighbour_distances", "Nearest-neighbour distance (median, IQR) between distinct points")

# --- 4f. District polygon membership ----------------------------------------
md("\n### 4f. District polygon membership\n")
district_membership_rows = []
district_counts_recomputed = {}
for c, df in CLEANED_DF.items():
    if not DISTRICTS_GEOJSON[c].exists():
        continue
    districts_gdf = gpd.read_file(DISTRICTS_GEOJSON[c])
    have = df[(df["lat"].notna()) & (df["lng"].notna())].copy()
    n_inside_by_district = Counter()
    n_outside_all = 0
    for _, r in have.iterrows():
        pt = Point(r["lng"], r["lat"])
        matched = None
        for _, drow in districts_gdf.iterrows():
            if drow.geometry.contains(pt) or drow.geometry.touches(pt):
                matched = drow.get("name", drow.get("district_code"))
                break
        if matched is None:
            n_outside_all += 1
        else:
            n_inside_by_district[matched] += 1
    district_counts_recomputed[c] = dict(n_inside_by_district)
    n_distinct_districts_recomputed = len(n_inside_by_district)
    reported_n_districts = reported_district_counts.get(c)
    district_membership_rows.append({
        "city": c,
        "n_geocoded_points": len(have),
        "n_districts_in_geojson": len(districts_gdf),
        "n_distinct_districts_hit_recomputed": n_distinct_districts_recomputed,
        "n_points_outside_all_districts": n_outside_all,
        "n_districts_in_final_report": reported_n_districts,
        "matches_report": (reported_n_districts == n_distinct_districts_recomputed)
        if reported_n_districts is not None else None,
    })
    if reported_n_districts is not None and reported_n_districts != n_distinct_districts_recomputed:
        add_anomaly(
            "4f. District polygon membership",
            f"{c}: recomputed distinct districts hit ({n_distinct_districts_recomputed}) "
            f"!= district count in final report ({reported_n_districts})",
            f"data/{c}/districts.geojson point-in-polygon recompute vs report_{c}.html"
        )
    if n_outside_all > 0:
        add_anomaly(
            "4f. District polygon membership",
            f"{c}: {n_outside_all} geocoded point(s) fall outside every polygon in districts.geojson",
            f"data/{c}/districts.geojson"
        )

district_membership_df = pd.DataFrame(district_membership_rows)
save_table(district_membership_df, "04f_district_membership",
           "Geocoded points inside districts.geojson polygons vs report's stated district count")

STATS["coordinate_quality"] = {
    "coverage": coverage_rows, "precision": precision_rows,
    "granularity": granularity_rows, "bboxes": bbox_table_rows,
    "outliers": outlier_df.to_dict(orient="records") if len(outlier_df) else [],
    "nearest_neighbour": nn_rows, "district_membership": district_membership_rows,
}

# ═══════════════════════════════════════════════════════════════════════
# SECTION 5 — Cross-city checks
# ═══════════════════════════════════════════════════════════════════════
md("\n## 5. Cross-city checks\n")

# 5a. URLs appearing in more than one city's raw file
url_to_cities_exact = {}
url_to_cities_norm = {}
for c, df in RAW_DF.items():
    for u in df["url"]:
        url_to_cities_exact.setdefault(u, set()).add(c)
    for u in df["url_norm"]:
        url_to_cities_norm.setdefault(u, set()).add(c)

cross_url_exact = [{"url": u, "cities": ";".join(sorted(cs))} for u, cs in url_to_cities_exact.items() if len(cs) > 1]
cross_url_norm = [{"url_normalised": u, "cities": ";".join(sorted(cs))} for u, cs in url_to_cities_norm.items() if len(cs) > 1]
cross_url_exact_df = pd.DataFrame(cross_url_exact)
cross_url_norm_df = pd.DataFrame(cross_url_norm)
save_table(cross_url_exact_df if len(cross_url_exact_df) else pd.DataFrame(columns=["url", "cities"]),
           "05a_cross_city_urls_exact", "URLs appearing in more than one city's raw file (exact string match)")
save_table(cross_url_norm_df if len(cross_url_norm_df) else pd.DataFrame(columns=["url_normalised", "cities"]),
           "05b_cross_city_urls_normalised", "URLs appearing in more than one city's raw file (normalised match)")

# 5b. Domains appearing in more than one city
domain_to_cities = {}
for c, df in RAW_DF.items():
    for d in df["domain"].dropna().unique():
        domain_to_cities.setdefault(d, set()).add(c)
cross_domain = [{"domain": d, "n_cities": len(cs), "cities": ";".join(sorted(cs))}
                for d, cs in domain_to_cities.items() if len(cs) > 1]
cross_domain_df = pd.DataFrame(cross_domain).sort_values("n_cities", ascending=False) if cross_domain \
    else pd.DataFrame(columns=["domain", "n_cities", "cities"])
save_table(cross_domain_df, "05c_cross_city_domains", "Domains appearing in more than one city's raw file")

# 5c. Coordinates from city A falling inside city B's bounding box
cross_bbox_rows = []
for cA, df in CLEANED_DF.items():
    have = df[(df["lat"].notna()) & (df["lng"].notna())]
    for cB, bboxB in bboxes.items():
        if cA == cB:
            continue
        n_inside = int(sum(in_bbox(r["lat"], r["lng"], bboxB) for _, r in have.iterrows()))
        if n_inside > 0:
            cross_bbox_rows.append({"city_of_point": cA, "bbox_of_city": cB, "n_points_inside": n_inside})
cross_bbox_df = pd.DataFrame(cross_bbox_rows) if cross_bbox_rows else pd.DataFrame(
    columns=["city_of_point", "bbox_of_city", "n_points_inside"])
save_table(cross_bbox_df, "05d_cross_city_bbox_overlap",
           "Coordinates from city A's cleaned corpus falling inside city B's bounding box")

STATS["cross_city"] = {
    "urls_exact": cross_url_exact, "urls_normalised": cross_url_norm,
    "domains": cross_domain, "bbox_overlap": cross_bbox_rows,
}

# ═══════════════════════════════════════════════════════════════════════
# SECTION 6 — External validation against the baseline screening ledger
# ═══════════════════════════════════════════════════════════════════════
md("\n## 6. External validation against the baseline screening ledger\n")
md(
    "`appendix_reports/report_<city>_baseline.pdf` contains a per-URL "
    "screening table (columns `# | Candidate | Decision | Type | "
    "Operational model | Activity evidence or discard reason | Location`). "
    "No `_chatgpt.html` cross-check export exists on disk for any of these "
    "five cities, so the two-parser cross-check used elsewhere in this repo "
    "(`src/phase_6_v2/lib/ledger_parser.py`) cannot run here. Instead this "
    "script reads the PDF directly with PyMuPDF: row-number tokens in the "
    "left-hand column give each row's page and y-position; the row's "
    "'Candidate' hyperlink (a PDF link annotation, not the wrapped/hyphenated "
    "display text) gives its source URL; 'Retained'/'Discarded' is read from "
    "the row's text band. This was validated against all five PDFs before "
    "being folded into this script: extracted row counts equal the raw-seed "
    "count for every city (Barcelona 261/261, Brighton 81/81, Dublin 83/83, "
    "London 115/115, Milan 163/163), and Barcelona's retained count from "
    "this extraction is 44/261, matching the '44 retained initiatives' "
    "stated in the report's own prose."
)


def find_ledger_columns(doc: fitz.Document) -> tuple[tuple[float, float], tuple[float, float], int]:
    for pno in range(len(doc)):
        words = doc[pno].get_text("words")
        hsh = [w for w in words if w[4] == "#"]
        cand = [w for w in words if w[4] == "Candidate"]
        dec = [w for w in words if w[4] == "Decision"]
        if hsh and cand and dec:
            hx, cx, dx = hsh[0][0], cand[0][0], dec[0][0]
            return (hx - 4, cx - 4), (cx - 4, dx - 4), pno
    raise RuntimeError("ledger header row not found")


DISCARD_REASON_PATTERNS = [
    ("duplicate", re.compile(r"duplicate", re.I)),
    ("outside_city_or_region", re.compile(r"outside .*(city|region)|regional service", re.I)),
    ("archival_or_ended", re.compile(r"archival|explicitly ended|one-off", re.I)),
    ("inaccessible_or_directory_generic",
     re.compile(r"inaccessible.*directory, policy, campaign, commercial", re.I | re.S)),
    ("inaccessible_only", re.compile(r"inaccessible", re.I)),
    ("directory_policy_only", re.compile(r"directory|umbrella|policy|campaign", re.I)),
]


def classify_discard_reason(text: str) -> str:
    for label, pat in DISCARD_REASON_PATTERNS:
        if pat.search(text):
            return label
    return "other_or_unclassified"


def parse_ledger_pdf(city: str) -> pd.DataFrame:
    doc = fitz.open(BASELINE_PDF[city])
    rownum_x, cand_x, _ = find_ledger_columns(doc)
    page_h = doc[0].rect.height

    anchors = []
    expect = 1
    for pno in range(len(doc)):
        words = doc[pno].get_text("words")
        nums = [w for w in words if w[4].isdigit() and rownum_x[0] <= w[0] <= rownum_x[1]]
        nums.sort(key=lambda w: w[1])
        for w in nums:
            if int(w[4]) == expect:
                anchors.append([pno, expect, w[1]])
                expect += 1

    bands = []
    for i, (pno, rid, y0) in enumerate(anchors):
        if i + 1 < len(anchors):
            npno, _, ny0 = anchors[i + 1]
            y1 = ny0 if npno == pno else page_h
        else:
            y1 = page_h
        bands.append((pno, rid, y0, y1))

    rows = []
    for pno, rid, y0, y1 in bands:
        words = doc[pno].get_text("words")
        band_words = [w for w in words if y0 <= w[1] < y1]
        text = " ".join(w[4] for w in band_words)
        decision = "Retained" if "Retained" in text else ("Discarded" if "Discarded" in text else "UNKNOWN")
        links = doc[pno].get_links()
        cand_uris = set()
        for l in links:
            rect = l.get("from")
            uri = l.get("uri")
            if not uri or rect is None:
                continue
            lyc = (rect.y0 + rect.y1) / 2
            lxc = (rect.x0 + rect.x1) / 2
            if y0 <= lyc < y1 and cand_x[0] <= lxc <= cand_x[1]:
                cand_uris.add(uri)
        reason = classify_discard_reason(text) if decision == "Discarded" else None
        rows.append({
            "row_id": rid, "decision": decision,
            "url": sorted(cand_uris)[0] if len(cand_uris) == 1 else None,
            "n_candidate_urls_found": len(cand_uris),
            "discard_reason_bucket": reason,
        })
    doc.close()
    return pd.DataFrame(rows)


ledger_dfs = {}
ledger_summary_rows = []
for city in CITIES:
    if not BASELINE_PDF[city].exists():
        continue
    ldf = parse_ledger_pdf(city)
    ledger_dfs[city] = ldf
    n_total = len(ldf)
    n_retained = int((ldf["decision"] == "Retained").sum())
    n_discarded = int((ldf["decision"] == "Discarded").sum())
    n_unknown = int((ldf["decision"] == "UNKNOWN").sum())
    n_zero_url = int((ldf["n_candidate_urls_found"] == 0).sum())
    n_multi_url = int((ldf["n_candidate_urls_found"] > 1).sum())
    expected = len(RAW_DF[city]) if city in RAW_DF else None
    ledger_summary_rows.append({
        "city": city, "n_ledger_rows": n_total, "n_retained": n_retained,
        "n_discarded": n_discarded, "n_decision_unknown": n_unknown,
        "n_rows_no_url_extracted": n_zero_url, "n_rows_ambiguous_multi_url": n_multi_url,
        "n_raw_seed_urls": expected,
        "row_count_matches_raw_seed": (n_total == expected) if expected is not None else None,
    })
    if n_unknown:
        add_anomaly("6. Ledger validation",
                     f"{city}: {n_unknown} ledger row(s) with neither 'Retained' nor 'Discarded' detected",
                     f"appendix_reports/report_{city}_baseline.pdf, row_id in "
                     f"{ldf.loc[ldf['decision']=='UNKNOWN','row_id'].tolist()}")
    if n_zero_url:
        add_anomaly("6. Ledger validation",
                     f"{city}: {n_zero_url} ledger row(s) had no extractable candidate-column hyperlink",
                     f"appendix_reports/report_{city}_baseline.pdf, row_id in "
                     f"{ldf.loc[ldf['n_candidate_urls_found']==0,'row_id'].tolist()}")

ledger_summary_df = pd.DataFrame(ledger_summary_rows)
save_table(ledger_summary_df, "06a_ledger_extraction_summary", "Baseline ledger extraction summary per city")

# Cross-tabulate: my cleaning decision (kept/discarded) x ledger decision
confusion_rows = []
reason_breakdown_rows = []
for city, ldf in ledger_dfs.items():
    raw_set = set(RAW_DF[city]["url"]) if city in RAW_DF else set()
    raw_set_norm = set(RAW_DF[city]["url_norm"]) if city in RAW_DF else set()
    cleaned_set = set(CLEANED_DF[city]["url"]) if city in CLEANED_DF else set()
    cleaned_set_norm = {normalize_url(u) for u in cleaned_set}

    matched = ldf[ldf["url"].notna()].copy()
    matched["exact_match_in_raw"] = matched["url"].isin(raw_set)
    matched["norm_match_in_raw"] = matched["url"].map(lambda u: normalize_url(u) in raw_set_norm)
    matched["my_decision"] = matched["url"].map(
        lambda u: "kept" if (u in cleaned_set or normalize_url(u) in cleaned_set_norm) else "discarded")

    n_ledger_urls = len(matched)
    n_norm_matched_to_my_raw = int(matched["norm_match_in_raw"].sum())
    match_rate_pct = round(100 * n_norm_matched_to_my_raw / n_ledger_urls, 1) if n_ledger_urls else None

    only_matched = matched[matched["norm_match_in_raw"]]
    for my_dec in ["kept", "discarded"]:
        for ledger_dec in ["Retained", "Discarded"]:
            n = int(((only_matched["my_decision"] == my_dec) & (only_matched["decision"] == ledger_dec)).sum())
            confusion_rows.append({"city": city, "my_decision": my_dec, "ledger_decision": ledger_dec, "n": n})

    kept_but_ledger_discarded = only_matched[(only_matched["my_decision"] == "kept") &
                                              (only_matched["decision"] == "Discarded")]
    reason_counts = kept_but_ledger_discarded["discard_reason_bucket"].value_counts()
    for reason, cnt in reason_counts.items():
        reason_breakdown_rows.append({"city": city, "reason_bucket": reason, "n_urls_i_kept": int(cnt)})

    STATS.setdefault("ledger_match_rates", {})[city] = {
        "n_ledger_urls_extracted": n_ledger_urls,
        "n_matched_normalised_to_my_raw_seed": n_norm_matched_to_my_raw,
        "match_rate_pct": match_rate_pct,
    }

confusion_df = pd.DataFrame(confusion_rows)
save_table(confusion_df, "06b_confusion_matrix",
           "My cleaning decision x ledger decision (normalised-URL-matched rows only)")
md(
    "*Cross-tabulated only over ledger rows whose extracted candidate URL "
    "matches (after normalisation) a URL in my raw-seed file for that city -- "
    "see `06a_ledger_extraction_summary.csv` for the match rate. Neither "
    "source is treated as ground truth; this reports agreement/disagreement "
    "between two independently produced judgements.*"
)

reason_breakdown_df = pd.DataFrame(reason_breakdown_rows) if reason_breakdown_rows else pd.DataFrame(
    columns=["city", "reason_bucket", "n_urls_i_kept"])
save_table(reason_breakdown_df, "06c_discard_reason_breakdown_for_urls_i_kept",
           "Ledger's stated discard reason for URLs my cleaning kept, by reason bucket")
md(
    "*`inaccessible_or_directory_generic` is the ledger's own catch-all "
    "template sentence ('The page was inaccessible or did not provide "
    "sufficient initiative-level evidence; it was a directory, policy, "
    "campaign, commercial or unrelated page...') -- it cites accessibility "
    "and page-type reasons in the same sentence for most discarded rows, so "
    "it cannot be cleanly split into 'dead/inaccessible' vs 'directory/policy' "
    "from the ledger text alone. Rows with a more specific reason (duplicate, "
    "outside-city, archival) are broken out separately; only rows with a "
    "reason string matching neither the generic template nor a specific "
    "pattern fall into `other_or_unclassified`.*"
)

STATS["ledger_extraction_summary"] = ledger_summary_rows
STATS["ledger_confusion_matrix"] = confusion_rows
STATS["ledger_reason_breakdown"] = reason_breakdown_rows

# ═══════════════════════════════════════════════════════════════════════
# SECTION 7 — Corpus size vs reported findings
# ═══════════════════════════════════════════════════════════════════════
md("\n## 7. Corpus size vs reported findings\n")

size_vs_findings_rows = []
for city in CITIES:
    seed_n = len(RAW_DF[city]) if city in RAW_DF else None
    cleaned_n = len(CLEANED_DF[city]) if city in CLEANED_DF else None
    enriched_n = _count_lines(FSI_ENRICHED[city])
    reported_n = reported_totals.get(city)
    district_n = reported_district_counts.get(city)
    n_fsi_types = None
    if FSI_ENRICHED[city].exists():
        types = []
        with open(FSI_ENRICHED[city], encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    types.append(rec.get("fsi_type", "unknown"))
        n_fsi_types = len(set(types))
    ratio = round(reported_n / seed_n, 3) if (reported_n is not None and seed_n) else None
    size_vs_findings_rows.append({
        "city": city, "seed_urls": seed_n, "cleaned_urls": cleaned_n,
        "enriched_records": enriched_n, "reported_fsi_total": reported_n,
        "reported_district_count": district_n, "n_distinct_fsi_types": n_fsi_types,
        "ratio_reported_fsi_to_seed_urls": ratio,
    })
size_vs_findings_df = pd.DataFrame(size_vs_findings_rows)
save_table(size_vs_findings_df, "07a_corpus_size_vs_reported_findings", "Corpus size vs reported findings, per city")

# correlations across the 5 cities (n=5 -- explicitly small-sample)
corr_base = "cleaned_urls"
corr_targets = ["reported_fsi_total", "reported_district_count", "n_distinct_fsi_types"]
corr_rows = []
x = size_vs_findings_df[corr_base].to_numpy(dtype=float)
for target in corr_targets:
    y = size_vs_findings_df[target].to_numpy(dtype=float)
    if np.isnan(x).any() or np.isnan(y).any() or len(x) < 3:
        r = None
    else:
        r = float(np.corrcoef(x, y)[0, 1])
    corr_rows.append({"x": corr_base, "y": target, "n": len(x), "pearson_r": round(r, 3) if r is not None else None})
corr_df = pd.DataFrame(corr_rows)
save_table(corr_df, "07b_correlations", f"Pearson correlation, corpus size ({corr_base}) vs headline reported figures (n=5 cities)")
md("*n=5: any correlation here is descriptive of these five cities only, not a generalisable statistical result.*")

# Evidence for "reported totals are counts of source documents, not initiatives"
doc_vs_initiative_rows = []
for city in CITIES:
    if city not in CLEANED_DF:
        continue
    cdf = CLEANED_DF[city].copy()
    cdf["domain"] = cdf["url"].map(registrable_domain)
    dom_counts = cdf["domain"].value_counts()
    n_multi_url_domains = int((dom_counts > 1).sum())
    n_urls_on_multi_url_domains = int(dom_counts[dom_counts > 1].sum())
    cleaned_n = len(cdf)
    enriched_n = _count_lines(FSI_ENRICHED[city])
    doc_vs_initiative_rows.append({
        "city": city,
        "cleaned_urls": cleaned_n,
        "fsi_enriched_records": enriched_n,
        "urls_equal_records_1to1": (cleaned_n == enriched_n),
        "n_domains_contributing_gt1_url": n_multi_url_domains,
        "n_urls_on_those_domains": n_urls_on_multi_url_domains,
    })
doc_vs_initiative_df = pd.DataFrame(doc_vs_initiative_rows)
save_table(doc_vs_initiative_df, "07c_documents_vs_initiatives_evidence",
           "Evidence: is the reported FSI total a count of source documents or of distinct initiatives?")
md(
    "*Every city's cleaned-URL count equals its `fsi_enriched.jsonl` record "
    "count exactly (see `urls_equal_records_1to1`) -- i.e. `facts['total']` "
    "in the rendered report (`src/phase_5/data_extractor.py:105`, "
    "`total = len(records)`) is one record per surviving URL, with no "
    "URL-to-initiative merging step anywhere in the pipeline. Where "
    "multiple distinct URLs on the same domain both survive cleaning "
    "(`n_domains_contributing_gt1_url` > 0), each contributes its own FSI "
    "record, so a single organisation with several scraped pages can "
    "inflate the reported total by more than one. This is direct, "
    "file-grounded evidence that the reported FSI totals are counts of "
    "surviving source documents, not counts of independently verified "
    "distinct initiatives.*"
)

STATS["size_vs_findings"] = size_vs_findings_rows
STATS["correlations"] = corr_rows
STATS["documents_vs_initiatives_evidence"] = doc_vs_initiative_rows

# ═══════════════════════════════════════════════════════════════════════
# SECTION 8 — Figures
# ═══════════════════════════════════════════════════════════════════════
md("\n## 8. Figures\n")

CITY_COLORS = dict(zip(CITIES, plt.rcParams["axes.prop_cycle"].by_key()["color"]))

# Fig 1: attrition funnel per city (small multiples, log scale)
fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=False)
stage_labels = ["raw", "cleaned", "coords", "scraped", "graphrag", "enriched", "reported"]
stage_keys = ["1_raw_seed", "2_cleaned", "3_usable_coords", "4_scraped_files",
              "5_graphrag_docs", "6_fsi_enriched", "7_reported_fsi_total"]
for ax, row in zip(axes, ledger_rows):
    vals = [row[k] if row[k] is not None else 0 for k in stage_keys]
    ax.bar(stage_labels, vals, color=CITY_COLORS[row["city"]])
    ax.set_title(row["city"])
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.tick_params(axis="x", rotation=90)
    ax.set_ylabel("count (log scale)")
fig.suptitle("Attrition funnel per city (note: 'scraped' and 'graphrag' are not on the same unit as URL-row stages -- see Section 2)")
fig.tight_layout()
save_fig(fig, "fig01_attrition_funnel", "Attrition funnel per city, all seven stages, log-scaled y-axis")

# Fig 2: coordinate coverage bar chart
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(coverage_df["city"], coverage_df["coverage_pct"], color=[CITY_COLORS[c] for c in coverage_df["city"]])
ax.set_ylabel("% rows with complete lat/lon (cleaned corpus)")
ax.set_ylim(0, 100)
for i, v in enumerate(coverage_df["coverage_pct"]):
    ax.text(i, v + 1, f"{v}%", ha="center")
fig.tight_layout()
save_fig(fig, "fig02_coordinate_coverage", "Coordinate coverage per city, cleaned corpus")

# Fig 3: decimal precision histogram (small multiples)
fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=False)
for ax, (c, dp_counts) in zip(axes, precision_all_dp):
    dp_counts = dp_counts.sort_index()
    ax.bar(dp_counts.index.astype(str), dp_counts.values, color=CITY_COLORS[c])
    ax.set_title(c)
    ax.set_xlabel("decimal places")
    ax.set_ylabel("count (lat+lng values)")
fig.suptitle("Coordinate decimal-precision distribution per city")
fig.tight_layout()
save_fig(fig, "fig03_decimal_precision_histogram", "Distribution of decimal places in lat/lng values, per city")

# Fig 4: points-per-unique-coordinate distribution (small multiples)
fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=False)
for ax, c in zip(axes, CITIES):
    if c not in cluster_size_dist:
        continue
    vc = cluster_size_dist[c].value_counts().sort_index()  # cluster_size -> n_such_clusters
    ax.bar(vc.index.astype(str), vc.values, color=CITY_COLORS[c])
    ax.set_title(c)
    ax.set_xlabel("rows sharing one coordinate")
    ax.set_ylabel("number of such coordinate clusters")
fig.suptitle("Points-per-unique-coordinate distribution per city")
fig.tight_layout()
save_fig(fig, "fig04_points_per_unique_coordinate", "How many rows share each distinct coordinate, per city")

# Fig 5: domain concentration curve
fig, ax = plt.subplots(figsize=(7, 5))
for c, curve in domain_concentration_curves.items():
    x_pct = 100 * np.arange(1, len(curve) + 1) / len(curve)
    ax.plot(x_pct, 100 * curve, label=c, color=CITY_COLORS[c])
ax.plot([0, 100], [0, 100], linestyle="--", color="grey", linewidth=1, label="equal share (reference)")
ax.set_xlabel("domains, ranked by URL count (percentile)")
ax.set_ylabel("cumulative share of raw-seed URLs (%)")
ax.legend(fontsize=8)
fig.tight_layout()
save_fig(fig, "fig05_domain_concentration_curve", "Domain concentration curve, raw-seed file, per city")

# Fig 6: per-city scatter of geocoded points with bounding box
# layout: 2 cols x 2 rows, 5th city centered in its own 3rd row
fig = plt.figure(figsize=(9, 11))
gs = fig.add_gridspec(3, 2)
grid_positions = [gs[0, 0], gs[0, 1], gs[1, 0], gs[1, 1]]
axes = [fig.add_subplot(pos) for pos in grid_positions]
axes.append(fig.add_subplot(gs[2, :]))  # 5th panel: centered, spans both columns

for ax, c in zip(axes, CITIES):
    if c not in CLEANED_DF or c not in bboxes:
        continue
    have = CLEANED_DF[c][(CLEANED_DF[c]["lat"].notna()) & (CLEANED_DF[c]["lng"].notna())]
    ax.scatter(have["lng"], have["lat"], s=8, alpha=0.6, color=CITY_COLORS[c])
    minx, miny, maxx, maxy = bboxes[c]
    ax.plot([minx, maxx, maxx, minx, minx], [miny, miny, maxy, maxy, miny], color="black", linewidth=1)
    ax.set_title(c)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
fig.suptitle("Geocoded points per city with generous bounding box (districts.geojson bounds + 25%)")
fig.tight_layout()
save_fig(fig, "fig06_geocoded_points_scatter", "Per-city scatter of geocoded points with bounding box")

# Fig 7: seed URLs vs reported FSIs scatter with identity line
fig, ax = plt.subplots(figsize=(6, 6))
sv = size_vs_findings_df.dropna(subset=["seed_urls", "reported_fsi_total"])
for _, r in sv.iterrows():
    ax.scatter(r["seed_urls"], r["reported_fsi_total"], color=CITY_COLORS[r["city"]], s=60, label=r["city"])
    ax.annotate(r["city"], (r["seed_urls"], r["reported_fsi_total"]), textcoords="offset points", xytext=(5, 5))
lim = max(sv["seed_urls"].max(), sv["reported_fsi_total"].max()) * 1.1
ax.plot([0, lim], [0, lim], linestyle="--", color="grey", linewidth=1, label="y = x (identity)")
ax.set_xlabel("seed URLs (raw-seed file)")
ax.set_ylabel("reported FSI total")
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.legend(fontsize=8)
fig.tight_layout()
save_fig(fig, "fig07_seed_vs_reported_scatter", "Seed URLs vs reported FSI total, with y=x identity line")

# ═══════════════════════════════════════════════════════════════════════
# SECTION 9 — Data quality issues found
# ═══════════════════════════════════════════════════════════════════════
md("\n## Data quality issues found\n")
md(
    "Every anomaly below was produced by the checks in this script; each "
    "is reported as found, not corrected, imputed, or dropped."
)
if ANOMALIES:
    anomalies_df = pd.DataFrame(ANOMALIES)
    save_table(anomalies_df, "09_data_quality_issues", "All anomalies found, with evidence")
else:
    md("\n*No anomalies were flagged by the checks in this script.*\n")

STATS["anomalies"] = ANOMALIES
STATS["n_anomalies"] = len(ANOMALIES)

# ─────────────────────────────────────────────────────────────────────────
# Write outputs
# ─────────────────────────────────────────────────────────────────────────
STATS_PATH.write_text(json.dumps(STATS, indent=2, default=_json_default), encoding="utf-8")
MD_PATH.write_text("\n".join(MD), encoding="utf-8")

print("\n" + "=" * 70)
print(f"Wrote {len(ANOMALIES)} anomalies.")
print(f"Wrote {STATS_PATH.relative_to(ROOT)}")
print(f"Wrote {MD_PATH.relative_to(ROOT)}")
print(f"Tables in {TABLES_DIR.relative_to(ROOT)}/, figures in {FIGURES_DIR.relative_to(ROOT)}/")
print("=" * 70)
