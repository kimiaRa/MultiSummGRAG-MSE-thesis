#!/usr/bin/env python3
"""
E1 — Claim auditability and E1-geo — geospatial verification.

Standalone, read-only evaluation of whether each report-generation system's
own numeric claims can be checked against a persistent, machine-readable
evidence base it retained -- NOT an evaluation of whether those claims are
correct. See AUDITABLE_DIRECT / AUDITABLE_DERIVED / NOT_MACHINE_AUDITABLE
below, and the "What this does and does not show" section of
E1_SUMMARY_v2.md, for why that distinction matters and why the two systems'
results are not a correctness comparison. Makes no network calls, runs no
pipeline stage,
does not re-scrape.

Reused modules (per explicit instruction, nothing else from the old
evaluation is imported):
  - src/phase_5/data_extractor.py::extract_facts()   -- pipeline evidence base
  - parse_ledger_html_v2() (this file)               -- baseline_v2 evidence
    base, parsed straight from report_<city>_baseline_v2.html's own
    Screening Ledger <table>. This is the CURRENT baseline condition and
    the one run_city()/main() score by default.

RETIRED: analysis/corpus_analysis.py::parse_ledger_pdf() -- the pre-
baseline_v2 ledger evidence base (loaded via a partial AST exec of just
that function and its direct helpers, so importing this script does not
also re-run the full corpus-characterisation analysis and its file/figure
side effects). Still loaded and still exposed as parse_ledger_pdf /
BASELINE_PDF for evaluation/scripts/run_cs1_cs4_widen_e2.py's archival
diff, but no longer read by run_city()/main() -- see BASELINE_V2_HTML.

Read-only exceptions (see e1_regex_comparison.csv / markdown discussion):
  - src/phase_6/factual_precision_scorer.py: only its number regex is
    reproduced here (textually, as a constant), for comparison against the
    corrected tokeniser. That file is not imported or otherwise reused.
  - data/<city>/output/eval_<city>.json: only the stored, already-computed
    factual_precision.hallucinated_numbers / counts are read, to report them
    as a retired-baseline comparison. Not recomputed or extended.

Run from the repo root with the project venv active:
    venv/bin/python3 evaluation/e1_evidence_consistency.py
"""
from __future__ import annotations

import ast
import itertools
import json
import re
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF
import geopandas as gpd
import pandas as pd
from bs4 import BeautifulSoup
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
APPENDIX_REPORTS = ROOT / "appendix_reports"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CITIES = ["barcelona", "brighton", "dublin", "london", "milan"]

REPORT_HTML = {c: DATA / c / "output" / f"report_{c}.html" for c in CITIES}
FSI_ENRICHED = {c: DATA / c / "output" / "fsi_enriched.jsonl" for c in CITIES}

# RETIRED CONDITION -- report_<city>_baseline.pdf in appendix_reports/ is the
# superseded, pre-baseline_v2 baseline. No result in the thesis rests on it
# (see chapter4, "An earlier baseline"). Kept only so
# evaluation/scripts/run_cs1_cs4_widen_e2.py can still diff against it for
# the record; run_city()/main() below no longer read from it -- see
# BASELINE_V2_HTML and extract_baseline_v2_prose()/parse_ledger_html_v2()
# for the current baseline condition.
BASELINE_PDF = {c: APPENDIX_REPORTS / f"report_{c}_baseline.pdf" for c in CITIES}

# CURRENT baseline condition (the one the thesis reports on).
BASELINE_V2_HTML = {c: DATA / c / "output" / f"report_{c}_baseline_v2.html" for c in CITIES}

URLS_CLEANED = {c: DATA / c / "urls_cleaned.csv" for c in CITIES}
DISTRICTS_GEOJSON = {c: DATA / c / "districts.geojson" for c in CITIES}
STORED_EVAL_JSON = {c: DATA / c / "output" / f"eval_{c}.json" for c in CITIES}

import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
from src.phase_5.data_extractor import extract_facts  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════
# STEP 1 — number tokenisation
# ═══════════════════════════════════════════════════════════════════════
# The old regex is reproduced textually below, per the explicit read-only
# exception, for comparison purposes only -- it is not imported and no other
# code from src/phase_6/factual_precision_scorer.py is used.
# Source: src/phase_6/factual_precision_scorer.py:4
#   _NUMBER_RE = re.compile(r'\b\d+(?:\.\d+)?%?\b')
NUMBER_RE_OLD = re.compile(r'\b\d+(?:\.\d+)?%?\b')

# Corrected tokeniser: tries the thousands-grouped form first so "14,896" is
# captured whole, not split into "14" and "896" at the comma boundary. The
# trailing boundary is a negative lookahead, not \b: '%' is a non-word
# character, so a trailing \b after an optional %? only holds when the '%'
# is the very last character in the string -- in real prose ("(44.3%)",
# "51.9% of") it is essentially always followed by another non-word
# character, so a naive \b there silently drops the '%' from almost every
# match and 44.3% is misread as a non-percentage 44.3. (?!\w) has no such
# blind spot: it only rejects being followed by a further word character
# (so "44.3rd" is still correctly not truncated to a bare "44.3").
NUMBER_RE_NEW = re.compile(
    r'\b\d{1,3}(?:,\d{3})+(?:\.\d+)?%?(?!\w)'   # thousands-grouped, e.g. 14,896 / 331,300
    r'|\b\d+(?:\.\d+)?%?(?!\w)'                  # plain integer/decimal/percentage
)


def tokenize_numbers(text: str, regex: re.Pattern = NUMBER_RE_NEW) -> list[dict]:
    """Find every number token in text. Returns raw string, parsed float
    value, is_percentage flag, and character offsets."""
    out = []
    for m in regex.finditer(text):
        raw = m.group(0)
        is_pct = raw.endswith('%')
        numeric = raw[:-1] if is_pct else raw
        value = float(numeric.replace(',', ''))
        out.append({
            "raw": raw, "value": value, "is_percentage": is_pct,
            "start": m.start(), "end": m.end(),
        })
    return out


# ═══════════════════════════════════════════════════════════════════════
# Reused module loaders
# ═══════════════════════════════════════════════════════════════════════

def load_parse_ledger_pdf():
    """Partial AST exec of analysis/corpus_analysis.py: pulls out only
    parse_ledger_pdf() and its direct helper dependencies
    (find_ledger_columns, classify_discard_reason, DISCARD_REASON_PATTERNS)
    so we reuse that exact extraction logic without executing the rest of
    that script (which runs a full descriptive analysis with file/figure
    side effects as an unrelated consequence of import)."""
    src_path = ROOT / "analysis" / "corpus_analysis.py"
    source = src_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(src_path))
    wanted = {"find_ledger_columns", "DISCARD_REASON_PATTERNS", "classify_discard_reason", "parse_ledger_pdf"}

    nodes = []
    for node in tree.body:
        name = getattr(node, "name", None)
        if name in wanted:
            nodes.append(node)
            continue
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(t in wanted for t in targets):
                nodes.append(node)

    found = set()
    for node in nodes:
        found.add(getattr(node, "name", None) or
                   next(t.id for t in node.targets if isinstance(t, ast.Name)))
    missing = wanted - found
    if missing:
        raise RuntimeError(f"analysis/corpus_analysis.py: could not locate {missing} for reuse")

    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {
        "fitz": fitz,
        "pd": pd,
        "re": re,
        "BASELINE_PDF": BASELINE_PDF,
    }
    exec(compile(module, filename=str(src_path), mode="exec"), ns)
    return ns["parse_ledger_pdf"]


def load_find_ledger_columns():
    """Same partial-exec approach as load_parse_ledger_pdf(), for the one
    small helper extract_baseline_prose() also needs (to locate the ledger
    table's start page so it can be excluded from prose)."""
    src_path = ROOT / "analysis" / "corpus_analysis.py"
    source = src_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(src_path))
    nodes = [n for n in tree.body if getattr(n, "name", None) == "find_ledger_columns"]
    if not nodes:
        raise RuntimeError("analysis/corpus_analysis.py: could not locate find_ledger_columns for reuse")
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {"fitz": fitz}
    exec(compile(module, filename=str(src_path), mode="exec"), ns)
    return ns["find_ledger_columns"]


parse_ledger_pdf = load_parse_ledger_pdf()
find_ledger_columns = load_find_ledger_columns()


# ═══════════════════════════════════════════════════════════════════════
# Prose extraction
# ═══════════════════════════════════════════════════════════════════════

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z"“(‘’])')
_LEDGER_CAPTION_RE = re.compile(
    r'^Evidence( and screening)? ledger for all [\d,]+ candidate URLs\.?$', re.IGNORECASE
)


def split_sentences(text: str) -> list[str]:
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]


def extract_pipeline_prose(city: str) -> tuple[str, list[tuple[str, str]]]:
    """Returns (full_report_text, [(section_heading, sentence), ...]) from
    the pipeline's rendered HTML report. Prose = <p> tags only (the report
    template has no tables; captions/footers live in <div> tags and are
    excluded), grouped under the preceding <h2>."""
    html = REPORT_HTML[city].read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")
    full_text = soup.get_text(" ", strip=True)

    pairs = []
    heading = "(preamble)"
    for h2 in soup.find_all("h2"):
        heading = h2.get_text(strip=True)
        sib = h2.find_next_sibling()
        while sib is not None and sib.name != "h2":
            if sib.name == "p":
                para = sib.get_text(" ", strip=True)
                for sent in split_sentences(para):
                    pairs.append((heading, sent))
            sib = sib.find_next_sibling()
    return full_text, pairs


def _ledger_header_y0(doc: fitz.Document, ledger_pno: int, rownum_x: tuple[float, float]) -> float | None:
    """Y-coordinate of the ledger table's own '#' column header on its
    first page, using the same word-level lookup find_ledger_columns()
    itself uses (word text == '#' inside the row-number column's x-range).
    This is the geometric boundary between prose and table on that page."""
    words = doc[ledger_pno].get_text("words")
    hsh = [w for w in words if w[4] == "#" and rownum_x[0] <= w[0] <= rownum_x[1]]
    if not hsh:
        return None
    return min(w[1] for w in hsh)  # topmost y0 among matches


def extract_baseline_prose(city: str) -> tuple[str, list[tuple[str, str]]]:
    """Returns (full_prose_text, [(section_heading, sentence), ...]) from
    the baseline PDF, covering every page up to and excluding the evidence
    ledger table.

    find_ledger_columns() gives the first page containing the '#'/
    'Candidate'/'Decision' header row -- but that page can *also* contain
    the tail of the prose (e.g. Barcelona's closing "Screening outcome: ..."
    paragraph sits on the same page as where the ledger table begins), so
    excluding that whole page loses real prose. Instead this processes that
    boundary page too and stops exactly at the table's own y-position
    (_ledger_header_y0) -- geometric, not font-size-based.

    An earlier version tried to detect a bold, 15-22pt "Evidence ledger for
    all N candidate URLs" heading span as the stop marker. That title text
    turned out to render at ~11.8pt in every one of the 5 baseline PDFs --
    inside the body-text size range, not the heading range -- so the check
    silently never fired, and the ledger table's own header row and
    following data rows (both also small/bold, e.g. the per-row 'Retained'/
    'Discarded' tags) were captured as if they were ordinary body prose on
    every city's boundary page, always landing in whatever section heading
    was last active (in every case: 'Notable Initiatives', the final
    section before the ledger). This was caught via a downstream sample
    (evaluation/e5_sample_for_annotation.py) turning up a claim sentence
    that was actually a fragment of ledger-table text -- not caught by
    visual spot-checking during Part 1, since the leak fragments into many
    short, individually unremarkable pseudo-sentences for most cities (only
    Brighton and Dublin's fragments were long enough to stand out by eye).
    The geometric y-position check has no such blind spot: it doesn't care
    what size or style the table's surrounding text happens to use."""
    doc = fitz.open(BASELINE_PDF[city])
    rownum_x, _cand_x, ledger_pno = find_ledger_columns(doc)
    header_y0 = _ledger_header_y0(doc, ledger_pno, rownum_x)

    pairs = []
    full_text_parts = []
    heading = "(title/intro)"
    block_text = ""

    def flush_block():
        nonlocal block_text
        if block_text.strip():
            for sent in split_sentences(block_text):
                if _LEDGER_CAPTION_RE.match(sent):
                    # the table's own caption, e.g. "Evidence ledger for all 83
                    # candidate URLs" -- sits just above header_y0, in the same
                    # small-bold style as the rest of the table furniture, so
                    # the y-position cutoff alone doesn't exclude it
                    continue
                pairs.append((heading, sent))
        block_text = ""

    for pno in range(ledger_pno + 1):
        page = doc[pno]
        d = page.get_text("dict")
        page_text_parts = []
        stop = False
        for block in d["blocks"]:
            if stop:
                break
            if "lines" not in block:
                continue
            for line in block["lines"]:
                if stop:
                    break
                for span in line["spans"]:
                    size = span["size"]
                    bold = bool(span["flags"] & 16)
                    text = span["text"]
                    if not text.strip():
                        continue
                    if pno == ledger_pno and header_y0 is not None and span["bbox"][1] >= header_y0 - 1.0:
                        stop = True
                        break
                    is_heading = bold and 15.0 <= size <= 22.0
                    is_title = bold and size > 22.0
                    if is_heading and "ledger" in text.lower():
                        stop = True
                        break
                    if is_heading:
                        flush_block()
                        heading = text.strip()
                    elif is_title:
                        pass
                    else:
                        block_text += text
                    page_text_parts.append(text)
                if not stop:
                    block_text += " "
        full_text_parts.append(" ".join(page_text_parts))
    flush_block()
    doc.close()
    return "".join(full_text_parts), pairs


def extract_baseline_v2_prose(city: str) -> tuple[str, list[tuple[str, str]]]:
    """Returns (full_report_text, [(section_heading, sentence), ...]) from
    report_<city>_baseline_v2.html -- the current baseline condition.

    baseline_v2 is rendered HTML with the same <h2>/<p> structure as the
    pipeline's own report template (see evaluation/baseline_v2_prompt.txt,
    step 6: "wrap each section title in <h2></h2> and its prose in <p></p>
    tags"), so this reuses extract_pipeline_prose()'s exact logic rather
    than duplicating it -- the only difference is which file is read. The
    Screening Ledger lives in a <table> sibling, not a <p>, so it is
    already excluded by that logic's `if sib.name == "p"` check -- the
    same behaviour the thesis's Experimental Procedure section describes
    ("the evidence ledger tables in the baseline files are excluded from
    the prose the consistency checks scan").

    Validated by evaluation/scripts/run_baseline_v2_e1_e2.py, which used
    exactly this redirect-and-reuse approach to produce the baseline_v2
    rows in evaluation/results/e1_summary.csv."""
    html = BASELINE_V2_HTML[city].read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")
    full_text = soup.get_text(" ", strip=True)

    pairs = []
    heading = "(preamble)"
    for h2 in soup.find_all("h2"):
        heading = h2.get_text(strip=True)
        sib = h2.find_next_sibling()
        while sib is not None and sib.name != "h2":
            if sib.name == "p":
                para = sib.get_text(" ", strip=True)
                for sent in split_sentences(para):
                    pairs.append((heading, sent))
            sib = sib.find_next_sibling()
    return full_text, pairs


def parse_ledger_html_v2(city: str) -> pd.DataFrame:
    """Parse the Screening Ledger <table> directly out of
    report_<city>_baseline_v2.html. Columns match parse_ledger_pdf()'s
    output shape (decision/discard_reason_bucket) so downstream code
    (_flatten_ledger, EvidenceIndex) needs no changes: row_id, url,
    decision ('Retained'/'Discarded'/'UNKNOWN'), discard_reason_bucket
    (slugified), reason_text (literal cell text, unslugified).

    Ported verbatim from evaluation/scripts/run_baseline_v2_e3.py's
    parse_ledger_html(), which produced the already-verified
    e3_v2_ledger_validation.csv (ledger row/retained/discarded counts
    reconcile exactly against each report's own stated outcome line, for
    all 5 cities)."""
    html = BASELINE_V2_HTML[city].read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table")
    tbody = table.find("tbody")
    rows = []
    for tr in tbody.find_all("tr"):
        tds = tr.find_all("td")
        row_id = tds[0].get_text(strip=True) if len(tds) > 0 else None
        a = tds[1].find("a") if len(tds) > 1 else None
        url = (a.get("href") if a else tds[1].get_text(strip=True)) if len(tds) > 1 else None
        decision_text = tds[2].get_text(strip=True) if len(tds) > 2 else ""
        decision = decision_text if decision_text in ("Retained", "Discarded") else "UNKNOWN"
        reason_text = tds[3].get_text(strip=True) if len(tds) > 3 else ""
        bucket = None
        if decision == "Discarded" and reason_text and reason_text != "—":
            bucket = reason_text.strip().lower().replace(" ", "_")
        rows.append({
            "row_id": row_id, "url": url, "decision": decision,
            "discard_reason_bucket": bucket,
            "reason_text": reason_text if decision == "Discarded" else None,
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# STEP 3 — Evidence bases and claim labelling
# ═══════════════════════════════════════════════════════════════════════

def _flatten_facts(facts: dict) -> tuple[list[float], dict[str, list[float]]]:
    """Flatten extract_facts() output into (flat_numbers, grouped_numbers)
    for AUDITABLE_DIRECT / AUDITABLE_DERIVED lookups."""
    flat: list[float] = []
    groups: dict[str, list[float]] = {}

    def add(v, group):
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            flat.append(float(v))
            groups.setdefault(group, []).append(float(v))

    for key, val in facts.items():
        if isinstance(val, dict):
            for v in val.values():
                add(v, key)
        elif isinstance(val, (int, float)):
            add(val, "_scalar")
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    add(item[1], key)
                elif isinstance(item, (int, float)):
                    add(item, key)
    return flat, groups


def _flatten_ledger(ledger_df: pd.DataFrame) -> tuple[list[float], dict[str, list[float]]]:
    n_total = len(ledger_df)
    n_retained = int((ledger_df["decision"] == "Retained").sum())
    n_discarded = int((ledger_df["decision"] == "Discarded").sum())
    n_unknown = int((ledger_df["decision"] == "UNKNOWN").sum())
    reason_counts = ledger_df.loc[ledger_df["decision"] == "Discarded", "discard_reason_bucket"].value_counts()

    flat = [float(n_total), float(n_retained), float(n_discarded)]
    if n_unknown:
        flat.append(float(n_unknown))
    groups = {"_scalar": list(flat), "discard_reason_bucket": [float(v) for v in reason_counts.values]}
    flat.extend(groups["discard_reason_bucket"])
    return flat, groups


class EvidenceIndex:
    """Precomputed AUDITABLE_DIRECT/AUDITABLE_DERIVED lookup structures for
    one evidence base. Built once per (city, system); each claim label
    lookup is then cheap. This checks only whether a claimed value CAN be
    located or derived in the evidence base the system itself retained --
    not whether that evidence base is the right source for that particular
    claim (see label() and E1_SUMMARY_v2.md's "What this does and does not
    show" section for why AUDITABLE_DIRECT is not the same as "correct").

    AUDITABLE_DERIVED-percentage candidates are deliberately restricted to
    sums over the evidence base *divided by an explicit 'total' value* (per
    spec wording: "percentages of the total"), not an unrestricted num/den
    sweep over every pair of evidence numbers. An unrestricted sweep was
    tried first and produces false positives: with ~60-90 evidence numbers
    per city there are ~8,000 arbitrary pairs, and small integers
    coincidentally land within 0.55pp of *some* pair's ratio far too often
    (e.g. a claimed "15" matched 45/3=15.0 in early testing, though nothing
    in the report actually derives 15 that way). The same reasoning
    excludes a generic non-percentage ratio matcher entirely -- it was
    tried and dropped for the same false-positive reason, so
    non-percentage AUDITABLE_DERIVED is exact-sum only, with no
    tolerance-based fallback."""

    PCT_TOL = 0.55       # percentage points, per spec

    def __init__(self, flat: list[float], groups: dict[str, list[float]], totals: list[float]):
        self.flat = flat
        self.number_set = set(round(v, 6) for v in flat)
        self.groups = groups
        self.totals = [t for t in totals if t and t > 0]

        self.sum_lookup: dict[float, str] = {}
        n = len(flat)
        for i in range(n):
            for j in range(i + 1, n):
                s = round(flat[i] + flat[j], 6)
                self.sum_lookup.setdefault(s, f"sum({flat[i]:g}+{flat[j]:g})")
        for gname, values in groups.items():
            for r in (2, 3):
                if r > len(values):
                    continue
                for combo in itertools.combinations(values, r):
                    s = round(sum(combo), 6)
                    combo_str = "+".join(f"{v:g}" for v in combo)
                    self.sum_lookup.setdefault(s, f"sum({combo_str}) [group={gname}]")

        # percentage-of-total candidates: 100 * num / total, single values
        # and bounded group-subset sums, for each candidate 'total'.
        self.pct_candidates: list[tuple[float, str]] = []
        for total in self.totals:
            for num in flat:
                self.pct_candidates.append((100.0 * num / total, f"{num:g}/{total:g}(total)*100"))
            for gname, values in groups.items():
                for r in (2, 3):
                    if r > len(values):
                        continue
                    for combo in itertools.combinations(values, r):
                        s = sum(combo)
                        combo_str = "+".join(f"{v:g}" for v in combo)
                        self.pct_candidates.append(
                            (100.0 * s / total, f"sum({combo_str})/{total:g}(total)*100 [group={gname}]")
                        )

    def label(self, value: float, is_percentage: bool) -> tuple[str, str]:
        """Returns (auditability_label, justification). auditability_label
        is one of AUDITABLE_DIRECT, AUDITABLE_DERIVED, NOT_MACHINE_AUDITABLE
        -- see module docstring. Note AUDITABLE_DIRECT means "this exact
        value exists somewhere in the evidence base", checked by value
        alone, with no verification that it is the evidence base entry the
        sentence actually means -- a real value can be found for the wrong
        quantity (slot-level misattribution). This method cannot detect
        that; it is a value-presence check, not a claim-meaning check."""
        if round(value, 6) in self.number_set:
            return "AUDITABLE_DIRECT", f"exact match to evidence value {value:g}"

        if is_percentage:
            for pct, just in self.pct_candidates:
                if abs(pct - value) <= self.PCT_TOL:
                    return "AUDITABLE_DERIVED", f"{just}={pct:.2f} (within {self.PCT_TOL}pp of {value:g})"
            return "NOT_MACHINE_AUDITABLE", "no exact or derived (percentage-of-total) match in evidence base"
        else:
            key = round(value, 6)
            if key in self.sum_lookup:
                return "AUDITABLE_DERIVED", f"{self.sum_lookup[key]}={value:g}"
            return "NOT_MACHINE_AUDITABLE", "no exact-sum match in evidence base"


# ═══════════════════════════════════════════════════════════════════════
# STEP 4 — E1-geo: geospatial verification
# ═══════════════════════════════════════════════════════════════════════

def read_urls_cleaned(city: str) -> pd.DataFrame:
    df = pd.read_csv(URLS_CLEANED[city])
    df.columns = [c.strip() for c in df.columns]
    return df


def known_district_names(city: str) -> list[str]:
    """District names from districts.geojson, longest first (see
    find_name_spans for why longest-first matters). Empty list if the city
    has no districts.geojson."""
    if not DISTRICTS_GEOJSON[city].exists():
        return []
    gdf = gpd.read_file(DISTRICTS_GEOJSON[city])
    return sorted(set(gdf["name"].dropna().astype(str)), key=len, reverse=True)


def recompute_district_membership(city: str) -> tuple[Counter, int]:
    """Point-in-polygon recompute of district membership from
    urls_cleaned.csv + districts.geojson (the shared geospatial ground
    truth specified for E1-geo, independent of either system's own
    retained-URL subset)."""
    gdf = gpd.read_file(DISTRICTS_GEOJSON[city])
    df = read_urls_cleaned(city)
    lat_col = "Lat" if "Lat" in df.columns else "lat"
    lng_col = "Lng" if "Lng" in df.columns else ("Lon" if "Lon" in df.columns else "lng")
    have = df[df[lat_col].notna() & df[lng_col].notna()]

    counts = Counter()
    n_outside = 0
    for _, row in have.iterrows():
        pt = Point(float(row[lng_col]), float(row[lat_col]))
        matched = None
        for _, drow in gdf.iterrows():
            if drow.geometry.contains(pt) or drow.geometry.touches(pt):
                matched = drow.get("name")
                break
        if matched is None:
            n_outside += 1
        else:
            counts[matched] += 1
    return counts, n_outside


_STOPWORDS_CAND = {
    "The", "This", "These", "Those", "Notably", "Yellow", "Green", "Red",
    "In", "No", "Among", "Comparing", "Green Districts", "Red Districts",
}


NAME_PROXIMITY_MAX_CHARS = 20  # tight on purpose: see find_nearest_number docstring


_ORG_NAME_PREFIX_RE = re.compile(r'\b(?:de|di|del|della|du)\s+$', re.IGNORECASE)


def _district_name_prefixes(known_names: list[str]) -> set[str]:
    """Word-prefixes of 'numbered' district names (e.g. 'Dublin' from
    'Dublin 7', 'Municipio' from 'Municipio 1'), used to catch elliptical
    enumerations like '(Municipio 1, 5, 9)' where only the first item
    repeats the prefix."""
    prefixes = set()
    for name in known_names:
        m = re.match(r'^(.+?)\s+\d+[A-Za-z]*$', name)
        if m:
            prefixes.add(m.group(1))
    return prefixes


def find_name_spans(sentence: str, known_names: list[str]) -> list[tuple[int, int]]:
    """All (start, end) spans in sentence matched by a known district name,
    plus elliptical enumerations sharing its prefix (see
    _district_name_prefixes) so e.g. 'Municipio 3, 2, 6' masks all three
    numbers, not just the '3' that repeats the word 'Municipio'. Longest
    names first so e.g. 'Dublin 12' is not also registered as a spurious
    partial 'Dublin 1' span (word boundaries already prevent the partial
    match; longest-first just keeps span list free of nesting)."""
    spans = []
    for name in known_names:
        for m in re.finditer(r'\b' + re.escape(name) + r'\b', sentence):
            spans.append((m.start(), m.end()))
    for prefix in _district_name_prefixes(known_names):
        list_re = re.compile(
            r'\b' + re.escape(prefix) + r'\s+\d+[A-Za-z]*'
            r'(?:\s*(?:,|and|&)\s*\d+[A-Za-z]*)+\b'
        )
        for m in list_re.finditer(sentence):
            spans.append((m.start(), m.end()))
    return spans


def tokenize_numbers_excluding_names(sentence: str, known_names: list[str]) -> list[dict]:
    """tokenize_numbers(), dropping any number token that is itself part of
    a known district name (e.g. the '7' in 'Dublin 7', the '1' in
    'Municipio 1'): those are administrative-unit identifiers, not
    quantitative claims, and must not be mistaken for a nearby district's
    stated count either."""
    name_spans = find_name_spans(sentence, known_names)
    out = []
    for num in tokenize_numbers(sentence):
        if any(ns <= num["start"] and num["end"] <= ne for ns, ne in name_spans):
            continue
        out.append(num)
    return out


_CLAUSE_BREAK_RE = re.compile(
    r'\b(?:whereas|while|but|however|although|though)\b|[;—]', re.IGNORECASE
)


def _is_org_name_context(sentence: str, name_start: int) -> bool:
    """True if the district name is immediately preceded by 'de'/'di'/'del'/
    etc, i.e. very likely part of an organisation's proper name (e.g.
    'Rebost Solidari de Gràcia') rather than a reference to the district
    itself -- a common Romance-language naming pattern that isn't also how
    genuine location statements are phrased in this corpus (those say 'in'/
    'at', not 'de'/'di')."""
    return bool(_ORG_NAME_PREFIX_RE.search(sentence[:name_start]))


def find_nearest_number(sentence: str, name: str, numbers: list[dict]) -> dict | None:
    """Nearest number token (by character distance) to a district-name
    occurrence in a sentence. Deliberately tight (NAME_PROXIMITY_MAX_CHARS):
    a wide window over-eagerly attaches an unrelated nearby number (e.g. an
    aggregate percentage stated earlier in the same sentence) to a district
    whose own count was spelled out as a word ('only one FSI') rather than a
    digit, which a digit-only tokeniser can never find correctly. Missing a
    match is the safe failure mode here; grabbing the wrong number is not.
    A candidate is also rejected if the text between name and number crosses
    a clause boundary ('whereas', 'while', ';', an em dash, ...): that
    number belongs to the other clause even when it lands inside the
    character window (e.g. "...population of 220,000), whereas Lewisham's
    single FSI...": 220,000 is Westminster's population, not Lewisham's
    count, despite being only ~10 characters away)."""
    best = None
    best_dist = None
    for m in re.finditer(r'\b' + re.escape(name) + r'\b', sentence):
        name_end = m.end()
        name_start = m.start()
        if _is_org_name_context(sentence, name_start):
            continue
        for num in numbers:
            if num["start"] >= name_end:
                dist = num["start"] - name_end
                between = sentence[name_end:num["start"]]
            elif num["end"] <= name_start:
                dist = name_start - num["end"]
                between = sentence[num["end"]:name_start]
                # a bare 4-digit number directly preceding a name ("the 2020
                # Lewisham Food Hub report") reads as a calendar year folded
                # into a title/proper-noun, not that district's FSI count --
                # no city here has anywhere near 1900-2100 initiatives.
                if 1900 <= num["value"] <= 2100 and len(num["raw"]) == 4 and not between.strip():
                    continue
            else:
                continue
            if dist > NAME_PROXIMITY_MAX_CHARS:
                continue
            if _CLAUSE_BREAK_RE.search(between):
                continue
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = num
    return best


CAND_NAME_RE = re.compile(
    r"([A-Z][\w&'’.À-ſ-]*(?:\s+(?:and\s+|&\s+)?[A-Z0-9][\w&'’.À-ſ-]*){0,3})"
    r"\s*(?:\(|,\s+(?:which\s+)?(?:alone\s+)?(?:accounts?\s+for|hosts?|contains?)\s+|\s+leading\s+at\s+)"
    r"\s*(\d[\d,]*\.?\d*)"
)


def find_named_candidates(sentence: str) -> list[str]:
    out = []
    for m in CAND_NAME_RE.finditer(sentence):
        name = m.group(1).strip()
        if name in _STOPWORDS_CAND or len(name) < 2:
            continue
        out.append(name)
    return out


def extract_stated_total(text: str, system: str) -> int | None:
    if system == "pipeline":
        m = re.search(r'([\d,]+)\s+Food Sharing Initiatives \(FSIs\)', text)
        return int(m.group(1).replace(",", "")) if m else None
    # baseline_v2 phrasing: "Screening outcome: N of M URLs retained."
    # (same STATED_OUTCOME_RE as evaluation/scripts/run_baseline_v2_e3.py)
    m = re.search(r'Screening outcome:\s*([\d,]+)\s+of\s+([\d,]+)\s+URLs retained', text, re.IGNORECASE)
    if m:
        return int(m.group(1).replace(",", ""))
    # retired baseline (v1) phrasing: "Screening outcome: N of the M input URLs were retained."
    m = re.search(
        r'Screening outcome:\s*([\d,]+)\s+of the\s+([\d,]+)\s+input URLs were retained',
        text
    )
    return int(m.group(1).replace(",", "")) if m else None


# ═══════════════════════════════════════════════════════════════════════
# Per-city, per-system pipeline
# ═══════════════════════════════════════════════════════════════════════

def load_pipeline_records(city: str) -> list[dict]:
    records = []
    with open(FSI_ENRICHED[city], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_city(city: str, claims_rows: list, geo_rows: list, summary_rows: list) -> None:
    print(f"=== {city} ===")

    have_geo = DISTRICTS_GEOJSON[city].exists() and URLS_CLEANED[city].exists()
    known_names = known_district_names(city) if have_geo else []

    def tok(sentence: str) -> list[dict]:
        # district-name-embedded digits (e.g. the '7' in 'Dublin 7') are
        # identifiers, not quantitative claims -- excluded everywhere,
        # claim extraction included, not just the geo checks.
        return tokenize_numbers_excluding_names(sentence, known_names) if known_names \
            else tokenize_numbers(sentence)

    # ---- Pipeline system ----
    records = load_pipeline_records(city)
    facts = extract_facts(records)
    flat, groups = _flatten_facts(facts)
    pipeline_index = EvidenceIndex(flat, groups, totals=[float(facts["total"])])

    full_text, pairs = extract_pipeline_prose(city)
    n_direct = n_derived = n_not_auditable = 0
    for heading, sentence in pairs:
        for num in tok(sentence):
            label, justification = pipeline_index.label(num["value"], num["is_percentage"])
            claims_rows.append({
                "city": city, "system": "pipeline", "section_heading": heading,
                "sentence": sentence, "raw_number": num["raw"], "parsed_value": num["value"],
                "is_percentage": num["is_percentage"], "label": label, "justification": justification,
            })
            n_direct += label == "AUDITABLE_DIRECT"
            n_derived += label == "AUDITABLE_DERIVED"
            n_not_auditable += label == "NOT_MACHINE_AUDITABLE"
    n_claims = n_direct + n_derived + n_not_auditable
    summary_rows.append({
        "city": city, "system": "pipeline", "n_claims": n_claims,
        "n_auditable_direct": n_direct, "n_auditable_derived": n_derived,
        "n_not_machine_auditable": n_not_auditable,
        "pct_auditable_direct": round(100 * n_direct / n_claims, 1) if n_claims else None,
        "pct_auditable_derived": round(100 * n_derived / n_claims, 1) if n_claims else None,
        "pct_not_machine_auditable": round(100 * n_not_auditable / n_claims, 1) if n_claims else None,
        "auditability_rate": round((n_direct + n_derived) / n_claims, 4) if n_claims else None,
    })

    # ---- Baseline system (baseline_v2 -- the current, reported condition;
    # the retired pre-baseline_v2 PDF is no longer scored by default here) ----
    ledger_df = parse_ledger_html_v2(city)
    bflat, bgroups = _flatten_ledger(ledger_df)
    n_total_ledger = len(ledger_df)
    n_retained_ledger = int((ledger_df["decision"] == "Retained").sum())
    baseline_index = EvidenceIndex(bflat, bgroups, totals=[float(n_total_ledger), float(n_retained_ledger)])

    bfull_text, bpairs = extract_baseline_v2_prose(city)
    n_direct = n_derived = n_not_auditable = 0
    for heading, sentence in bpairs:
        for num in tok(sentence):
            label, justification = baseline_index.label(num["value"], num["is_percentage"])
            claims_rows.append({
                "city": city, "system": "baseline_v2", "section_heading": heading,
                "sentence": sentence, "raw_number": num["raw"], "parsed_value": num["value"],
                "is_percentage": num["is_percentage"], "label": label, "justification": justification,
            })
            n_direct += label == "AUDITABLE_DIRECT"
            n_derived += label == "AUDITABLE_DERIVED"
            n_not_auditable += label == "NOT_MACHINE_AUDITABLE"
    n_claims = n_direct + n_derived + n_not_auditable
    summary_rows.append({
        "city": city, "system": "baseline_v2", "n_claims": n_claims,
        "n_auditable_direct": n_direct, "n_auditable_derived": n_derived,
        "n_not_machine_auditable": n_not_auditable,
        "pct_auditable_direct": round(100 * n_direct / n_claims, 1) if n_claims else None,
        "pct_auditable_derived": round(100 * n_derived / n_claims, 1) if n_claims else None,
        "pct_not_machine_auditable": round(100 * n_not_auditable / n_claims, 1) if n_claims else None,
        "auditability_rate": round((n_direct + n_derived) / n_claims, 4) if n_claims else None,
    })

    # ---- E1-geo ----
    if not have_geo:
        geo_rows.append({
            "city": city, "system": "both", "check": "recompute_unavailable",
            "district_name": None, "stated_value": None, "recomputed_value": None,
            "match": None, "delta": None,
            "note": "districts.geojson or urls_cleaned.csv missing; E1-geo not run for this city",
        })
        return

    recomputed_counts, n_outside = recompute_district_membership(city)

    for system, text, pairs_ in (("pipeline", full_text, pairs), ("baseline_v2", bfull_text, bpairs)):
        # (a) per-district stated count vs recomputed
        stated_by_district: dict[str, dict] = {}
        for heading, sentence in pairs_:
            nums = tok(sentence)
            if not nums:
                continue
            for name in known_names:
                if re.search(r'\b' + re.escape(name) + r'\b', sentence) is None:
                    continue
                hit = find_nearest_number(sentence, name, nums)
                if hit is not None and not hit["is_percentage"]:
                    prev = stated_by_district.get(name)
                    if prev is None:
                        stated_by_district[name] = {"value": hit["value"], "sentence": sentence}

        for name, info in stated_by_district.items():
            recomputed = recomputed_counts.get(name)
            geo_rows.append({
                "city": city, "system": system, "check": "per_district_count",
                "district_name": name, "stated_value": info["value"],
                "recomputed_value": recomputed,
                "match": (recomputed == info["value"]) if recomputed is not None else None,
                "delta": (info["value"] - recomputed) if recomputed is not None else None,
                "note": info["sentence"][:200] if recomputed is None else "",
            })

        # (b) sum of stated district counts vs report's own stated total
        stated_total = extract_stated_total(text, system)
        sum_stated = sum(v["value"] for v in stated_by_district.values())
        n_matched = len(stated_by_district)
        n_known = len(known_names)
        geo_rows.append({
            "city": city, "system": system, "check": "district_sum_vs_total",
            "district_name": None,
            "stated_value": sum_stated if stated_by_district else None,
            "recomputed_value": stated_total,
            "match": (stated_total == sum_stated) if (stated_total is not None and stated_by_district) else None,
            "delta": (stated_total - sum_stated) if (stated_total is not None and stated_by_district) else None,
            "note": f"delta = stated_total - sum(stated per-district counts); "
                    f"per-district extractor paired {n_matched}/{n_known} known districts with a "
                    f"nearby count -- a nonzero delta with low coverage reflects extractor "
                    f"coverage as much as report accuracy, not a clean shortfall finding; "
                    f"empty if either total could not be extracted",
        })

        # (c) districts named in prose (Geographic Distribution heading) not in geojson
        geo_heading_pairs = [(h, s) for h, s in pairs_ if "geographic" in h.lower()]
        candidates: set[str] = set()
        for _, sentence in geo_heading_pairs:
            for cand in find_named_candidates(sentence):
                is_known = any(
                    re.fullmatch(re.escape(kn), cand, re.IGNORECASE) or kn.lower() in cand.lower()
                    for kn in known_names
                )
                if not is_known:
                    candidates.add(cand)
        for cand in sorted(candidates):
            geo_rows.append({
                "city": city, "system": system, "check": "district_named_not_in_geojson",
                "district_name": cand, "stated_value": None, "recomputed_value": None,
                "match": False, "delta": None,
                "note": "capitalised name adjacent to a number in the Geographic Distribution "
                        "prose, not found among districts.geojson names (heuristic extraction)",
            })

    geo_rows.append({
        "city": city, "system": "ground_truth", "check": "n_points_outside_all_districts",
        "district_name": None, "stated_value": None, "recomputed_value": n_outside,
        "match": n_outside == 0, "delta": None,
        "note": "point-in-polygon recompute from urls_cleaned.csv against districts.geojson",
    })


# ═══════════════════════════════════════════════════════════════════════
# STEP 1 (output) — old vs corrected tokeniser comparison
# ═══════════════════════════════════════════════════════════════════════

def build_regex_comparison_rows() -> list[dict]:
    rows = []
    for city in CITIES:
        stored_path = STORED_EVAL_JSON[city]
        if not stored_path.exists():
            rows.append({
                "city": city, "old_claimed_count": None, "old_hallucinated_count": None,
                "old_hallucinated_numbers": None, "new_hallucinated_count": None,
                "new_hallucinated_numbers": None, "phantom_fragments_resolved": None,
                "note": "no stored eval_<city>.json found",
            })
            continue
        stored = json.loads(stored_path.read_text(encoding="utf-8"))
        fp = stored["detail"]["factual_precision"]
        old_hallucinated_numbers = fp["hallucinated_numbers"]

        records = load_pipeline_records(city)
        facts = extract_facts(records)
        full_text, _ = extract_pipeline_prose(city)

        old_claimed_recomputed = set(NUMBER_RE_OLD.findall(full_text))
        old_allowed_recomputed = set(NUMBER_RE_OLD.findall(str(facts)))
        old_hallucinated_recomputed = old_claimed_recomputed - old_allowed_recomputed

        new_claimed = set(NUMBER_RE_NEW.findall(full_text))
        new_allowed = set(NUMBER_RE_NEW.findall(str(facts)))
        new_hallucinated = new_claimed - new_allowed

        phantom_fragments = [n for n in old_hallucinated_numbers if n in ("000", "896", "308")]
        resolved = [n for n in phantom_fragments if n not in new_hallucinated]

        rows.append({
            "city": city,
            "old_claimed_count": fp["claimed_count"],
            "old_hallucinated_count": fp["hallucinated_count"],
            "old_hallucinated_numbers": ";".join(old_hallucinated_numbers),
            "old_hallucinated_count_recomputed_today": len(old_hallucinated_recomputed),
            "new_claimed_count": len(new_claimed),
            "new_hallucinated_count": len(new_hallucinated),
            "new_hallucinated_numbers": ";".join(sorted(new_hallucinated)),
            "phantom_fragments_in_stored_result": ";".join(phantom_fragments) if phantom_fragments else "",
            "phantom_fragments_resolved_by_fix": ";".join(resolved) if resolved else (
                "n/a - none present" if not phantom_fragments else "none resolved"
            ),
        })
    return rows


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=== E1 input conditions (resolved paths) ===")
    print("  pipeline    :", REPORT_HTML[CITIES[0]].parent, "/ report_<city>.html")
    print("  baseline_v2 :", BASELINE_V2_HTML[CITIES[0]].parent, "/ report_<city>_baseline_v2.html  (current condition)")
    print("  RETIRED     :", BASELINE_PDF[CITIES[0]].parent, "/ report_<city>_baseline.pdf  (superseded -- not scored below)")
    for c in CITIES:
        missing = [str(p) for p in (REPORT_HTML[c], BASELINE_V2_HTML[c]) if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{c}: missing required input(s): {missing}")
    print()

    claims_rows: list[dict] = []
    geo_rows: list[dict] = []
    summary_rows: list[dict] = []

    for city in CITIES:
        run_city(city, claims_rows, geo_rows, summary_rows)

    claims_df = pd.DataFrame(claims_rows)
    summary_df = pd.DataFrame(summary_rows)
    geo_df = pd.DataFrame(geo_rows)
    regex_df = pd.DataFrame(build_regex_comparison_rows())

    claims_df.to_csv(RESULTS_DIR / "e1_claims.csv", index=False)
    summary_df.to_csv(RESULTS_DIR / "e1_summary.csv", index=False)
    geo_df.to_csv(RESULTS_DIR / "e1_geo.csv", index=False)
    regex_df.to_csv(RESULTS_DIR / "e1_regex_comparison.csv", index=False)

    print("\n=== e1_summary.csv ===")
    print(summary_df.to_string(index=False))
    print("\n=== e1_regex_comparison.csv ===")
    print(regex_df.to_string(index=False))

    write_markdown(summary_df, claims_df, geo_df, regex_df)
    print(f"\nWrote outputs to {RESULTS_DIR} and {RESULTS_DIR.parent / 'E1_SUMMARY.md'}")


def write_markdown(summary_df, claims_df, geo_df, regex_df) -> None:
    md = []
    md.append("# E1 — Claim auditability and E1-geo — geospatial verification\n")
    md.append(
        "Read-only evaluation of whether each report-generation system's own numeric claims "
        "can be checked against a persistent, machine-readable evidence base that system "
        "retained. No pipeline stage was run, no network call was made, no number was "
        "estimated. Generated by `evaluation/e1_evidence_consistency.py`.\n\n"
        "**This measures auditability, not accuracy or cross-system consistency of the same "
        "kind of thing** -- see \"What this metric is, and is not\" under Step 3/4 below before "
        "reading the summary table, especially before comparing the two systems' rates "
        "head-to-head.\n"
    )

    md.append("## Step 1 — number tokenisation: old vs corrected\n")
    md.append(
        "The old regex (`src/phase_6/factual_precision_scorer.py:4`, read for comparison "
        "only) is `\\b\\d+(?:\\.\\d+)?%?\\b`, which does not include the comma thousands "
        "separator in its character class, so it splits `14,896` into two tokens, `14` "
        "and `896`. Both fragments are then checked for hallucination independently. Since "
        "`896` rarely also occurs standalone elsewhere in `str(facts)`, it is flagged as a "
        "phantom hallucination, though the actual number `14,896` may be a real, "
        "grounded evidence value. The corrected tokeniser tries a thousands-grouped "
        "pattern first, so `14,896` is captured whole.\n"
    )
    md.append(regex_df.to_markdown(index=False))
    md.append("")
    md.append(
        "*`old_hallucinated_numbers` / `old_claimed_count` / `old_hallucinated_count` "
        "are read verbatim from the stored `data/<city>/output/eval_<city>.json` "
        "(read-only exception 2) -- not recomputed. `old_hallucinated_count_recomputed_today` "
        "reruns the same old-regex/`str(facts)` method against the report and facts as they "
        "exist on disk today, as a sanity check that it is still reproducible (it may differ "
        "slightly from the stored count if the on-disk report or fsi_enriched.jsonl changed "
        "since that eval run). `new_*` columns use the corrected tokeniser on the same "
        "inputs, via the same crude `claimed - allowed` set-difference against `str(facts)` "
        "the old scorer used (reproduced independently here, not imported, to isolate the "
        "regex fix specifically). `phantom_fragments_in_stored_result` lists which of the "
        "three named phantom fragments (`000`, `896`, `308`) appear in this city's stored "
        "hallucinated_numbers; `phantom_fragments_resolved_by_fix` reports whether they "
        "disappear once the tokeniser is corrected -- in every city where they occurred, "
        "they do.*\n\n"
        "*`new_hallucinated_count` is higher than `old_hallucinated_count`, which looks like "
        "the fix made things worse -- it did not. Building the corrected tokeniser surfaced a "
        "second, independent bug in the old pattern: its trailing `\\b` after the optional "
        "`%?` only holds when `%` is the very last character in the whole string, since `%` "
        "is a non-word character -- in real prose (`\"44.3%\"` followed by a space, `)`, or "
        "`,`) that boundary essentially never holds, so the old regex silently drops the `%` "
        "and reads `\"44.3%\"` as a bare `44.3` on almost every real occurrence. `str(facts)` "
        "never contains a literal `%` character, so a bare `44.3` had a chance of matching a "
        "same-valued fact by coincidence; a correctly-captured `\"44.3%\"` token cannot match "
        "`str(facts)` at all under this crude substring method, and every percentage claim "
        "now correctly shows up as claimed-but-not-literally-in-`str(facts)`. That inflates "
        "`new_hallucinated_count` here, but it is this table's method (bare string "
        "containment in `str(facts)`) reaching its limit, not the corrected tokeniser making "
        "wrong calls -- Step 3/4 below replaces that crude method with a real evidence-base "
        "lookup and handles percentages correctly (see `is_percentage` / AUDITABLE_DERIVED).*\n"
    )

    md.append("## Step 3/4 — Claim auditability summary (per city x system)\n")

    md.append("### What this metric is, and is not\n")
    md.append(
        "Each numeric claim in each report is labelled one of three ways against *that "
        "system's own* retained evidence base:\n\n"
        "- **AUDITABLE_DIRECT** -- the claimed value is present, as-is, in a persistent, "
        "machine-readable evidence base the system retained.\n"
        "- **AUDITABLE_DERIVED** -- not present directly, but exactly computable from that "
        "evidence base: an exact sum of evidence-base values, or a percentage *of a candidate "
        "total* (facts['total'] for the pipeline; ledger total-candidates and retained-count "
        "for the baseline) within 0.55 percentage points. Restricted to a candidate 'total' "
        "rather than an unrestricted ratio between any two evidence numbers -- an unrestricted "
        "num/den sweep was tried first and dropped: with ~60-90 evidence numbers per city "
        "there are thousands of arbitrary pairs, and a small claimed integer lands within "
        "tolerance of *some* coincidental pair far too easily (e.g. a claimed \"15\" matched "
        "an unrelated \"45/3=15.0\" in early testing). The same reasoning excludes a generic "
        "non-percentage ratio matcher entirely.\n"
        "- **NOT_MACHINE_AUDITABLE** -- neither. Verifiable, if at all, only against a source "
        "the system did not retain in machine-readable form (for the baseline: the live web "
        "page it read during browsing).\n\n"
        "`auditability_rate = (n_auditable_direct + n_auditable_derived) / n_claims`. Every "
        "AUDITABLE_DERIVED row's `justification` column in `e1_claims.csv` names the exact "
        "evidence values used, so any AUDITABLE_DERIVED label can be manually re-checked "
        "rather than taken on faith.\n\n"
        "**1. This measures verifiability, not correctness.** A NOT_MACHINE_AUDITABLE claim "
        "is not thereby shown to be false -- it means this script cannot check it without "
        "re-fetching a live page, not that the claim is wrong. Inspection of the 164 baseline "
        "NOT_MACHINE_AUDITABLE rows confirms what they actually are: page-sourced operational "
        "facts -- e.g. \"reports 737 families served fortnightly\" (Barcelona, Rebost "
        "Solidari de Gràcia), \"630,612 kilograms recovered\" (Barcelona, Nutrició Sense "
        "Fronteres) -- each carrying the justification \"no exact-sum match in evidence "
        "base\". These are claims the baseline system read directly off a live web page while "
        "browsing; its retained evidence base (the screening ledger) records only retain/"
        "discard decisions and candidate URLs, and structurally cannot contain figures like "
        "these, regardless of whether the figures are accurate.\n\n"
        "**2. An AUDITABLE_DIRECT claim is not necessarily correct either.** The check "
        "confirms the claimed value exists *somewhere* in the evidence base -- not that it is "
        "the evidence-base entry the sentence actually means. A real value can be found for "
        "the wrong quantity (slot-level misattribution), and this method has no way to catch "
        "that: it is a value-presence check, not a claim-meaning check. Worked example -- "
        "Milan pipeline report, `e1_claims.csv` row for `raw_number=\"34%\"`: the sentence "
        "\"Milan hosts 147 Food Sharing Initiatives (FSIs) across nine postal districts, with "
        "the highest concentrations in Municipio 1 (28 FSIs), Municipio 5 (22 FSIs), and "
        "Municipio 9 (15 FSIs), which collectively account for 34% of all initiatives\" is "
        "labelled AUDITABLE_DIRECT with justification \"exact match to evidence value 34\" -- "
        "`facts['type_pct']['other'] == 34.0` for Milan, confirmed by inspecting `extract_facts()`"
        "'s output directly. But that 34.0 is the *FSI-type* share for the 'other' category, "
        "not a district concentration. The sentence's actual quantity -- the combined share of "
        "the three named districts -- is 28+22+15=65 of 147, i.e. 44.2%, not 34%; this exact "
        "arithmetic mismatch is independently flagged by Part 2's CS-4 check (`e2_flags.csv`, "
        "Milan pipeline: \"constituents (28+22+15)=sum 65, total=147: predicted 44.22%, "
        "claimed 34%, diff 10.22pp\"). AUDITABLE_DIRECT and AUDITABLE_DERIVED are therefore "
        "an **upper bound** on accuracy, not a measurement of it -- some fraction of claims "
        "labelled auditable here are auditable to the *wrong* fact.\n\n"
        "**3. The resulting asymmetry between systems is architectural, not a quality gap.** "
        "The pipeline's evidence base (`facts`, from `extract_facts()`) is, by construction, "
        "the exact source the Phase-5 prompt instructed the generator to draw every number "
        "from -- a structured dataset built before generation, so claims drawn from it are "
        "auditable by design. The baseline's evidence base (the screening ledger) records "
        "retain/discard decisions and URLs from an agentic browsing workflow; the figures its "
        "report states were read from live pages during that browsing and were never written "
        "to a persistent, machine-readable store the way `facts` was. That a much larger share "
        "of baseline claims come back NOT_MACHINE_AUDITABLE reflects this structural "
        "difference in what each system kept, not a difference in how careful or truthful "
        "either system's claims are -- and it is exactly why AUDITABLE_DIRECT / "
        "AUDITABLE_DERIVED / NOT_MACHINE_AUDITABLE are not framed here as a correctness "
        "comparison between the two systems.\n"
    )
    md.append(summary_df.to_markdown(index=False))
    md.append("")

    md.append("## Step 4 — E1-geo: geospatial verification\n")
    md.append(
        "Ground truth = point-in-polygon recompute (shapely `contains`/`touches`) of "
        "`data/<city>/urls_cleaned.csv` coordinates against `data/<city>/districts.geojson`, "
        "the single shared file pair specified for this check -- used as ground truth for "
        "*both* systems' stated counts, even though the baseline's own retained-URL set "
        "differs from the pipeline's cleaned set (a documented limitation, not an oversight).\n\n"
        "- **per_district_count**: every `(district name, count)` mention the extractor could "
        "pair (nearest-number-to-known-district-name heuristic, <=20 characters, rejecting a "
        "candidate that crosses a clause boundary such as 'whereas'/';', that is itself a "
        "bare calendar year folded into a title, or where the name reads as part of an "
        "organisation's proper name -- e.g. 'Rebost Solidari de Gràcia' -- rather than a "
        "reference to the district; see `find_nearest_number` in the script for the exact "
        "rules and the false positives each one was added to fix) against the recomputed "
        "count for that district.\n"
        "- **district_sum_vs_total**: sum of all extracted per-district counts vs the report's "
        "own stated city-wide total (`delta` = shortfall; the extraction-coverage caveat in "
        "each row's `note` matters here -- see below).\n"
        "- **district_named_not_in_geojson**: capitalised names adjacent to a number in the "
        "'Geographic Distribution' prose that do not match any `districts.geojson` name "
        "(heuristic regex extraction -- see script `CAND_NAME_RE`; may contain false "
        "positives/negatives, spot-check before treating as a hard finding).\n"
        "- **n_points_outside_all_districts** (system=ground_truth): points in "
        "`urls_cleaned.csv` that fall inside no polygon at all.\n\n"
        "**Known limitations of the district-count extractor** (kept deliberately "
        "conservative -- a missed pairing is the safe failure mode, a wrong one is not): it "
        "cannot recover a count stated as a word rather than a digit (e.g. 'only one FSI'); "
        "it cannot correctly split multi-district parallel lists whose counts are given "
        "afterwards rather than adjacently (e.g. 'Dublin 7 and Dublin 16 host 4 and 3 FSIs "
        "respectively' pairs neither district, since neither has a count within 20 characters "
        "of its own name); and the organisation-proper-name guard only covers the Romance "
        "'de/di/del/della/du X' pattern actually observed in this corpus, not every possible "
        "phrasing. Coverage is therefore a real, reported undercount, not full recall -- "
        "`district_sum_vs_total`'s note states how many of a city's known districts were "
        "successfully paired, and low coverage there means an apparent shortfall reflects "
        "the extractor as much as the report.\n\n"
        "Where a check could not be run for a city, the cell is left empty rather than "
        "imputed.\n"
    )
    md.append(geo_df.to_markdown(index=False))
    md.append("")

    (RESULTS_DIR.parent / "E1_SUMMARY.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
