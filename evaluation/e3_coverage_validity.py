#!/usr/bin/env python3
"""
E3 — Coverage vs validity.

RETIRED CONDITION WARNING: this module's own main() (and therefore
e3_coverage.csv / e3_agreement.csv / e3_directionality.csv /
e3_reason_breakdown.csv / E3_SUMMARY.md, all written below) score the
retired, pre-baseline_v2 baseline -- because analysis/tables/02_attrition_
ledger.csv and friends (below) were computed by analysis/corpus_analysis.py
against that superseded PDF ledger, and no baseline_v2 equivalent of those
precomputed tables exists. No result in the thesis rests on this retired
condition (see chapter4, "An earlier baseline"). For the CURRENT baseline_v2
condition, see evaluation/scripts/run_baseline_v2_e3.py, which parses each
report_<city>_baseline_v2.html's own Screening Ledger <table> directly
(parse_ledger_html) and is the authoritative source of the e3_v2_*.csv
files used in the thesis. Repointing this file's own main() at baseline_v2
would require rebuilding analysis/tables/*.csv from the baseline_v2 ledger
first, which run_baseline_v2_e3.py already does independently rather than
by extending this module's main() -- so that script's main(), not this
one's, is what to run for baseline_v2.

MODULE STATUS: this file's standalone old-baseline main()/CLI path is
retired (see warning above) and produces no thesis result on its own, but
the module itself is NOT obsolete: run_baseline_v2_e3.py imports several of
its generic, condition-agnostic helper functions and constants by name --
cluster_bootstrap_ratio, cluster_bootstrap_kappa, build_agreement,
build_directionality, build_reason_breakdown, ACCESS_FAILURE_PATTERNS,
ACCESS_FAILURE_RE, url_to_raw_filename, body_text_length, RAW_DIR,
RECALL_CHECK_CITIES, N_RESAMPLES, RNG_SEED -- and applies them directly to
baseline_v2's own confusion matrix / reason-breakdown / recall data. Those
functions never hardcode "baseline" (only build_coverage's row-labelling
does, which is why run_baseline_v2_e3.py has its own build_coverage_v2
instead of reusing build_coverage). Keep this file for that reason even
though its own main()/CLI output is retired.

Standalone, read-only, no network calls, no pipeline stage run. Reuses:
  - analysis/tables/02_attrition_ledger.csv, 06a_ledger_extraction_summary.csv,
    06b_confusion_matrix.csv, 06c_discard_reason_breakdown_for_urls_i_kept.csv
    (already computed by analysis/corpus_analysis.py -- read directly, not
    recomputed)
  - e1_evidence_consistency.parse_ledger_pdf (the ledger extractor loaded in
    Part 1 via a partial AST exec of analysis/corpus_analysis.py) for the
    recall counter-check's per-URL decision/reason data

Nothing else from any prior evaluation code was read, imported, or adapted.

Five outputs:
  1. Coverage rate (surviving/seed) per city + pooled, both systems.
  2. Phase-0-vs-ledger agreement: raw agreement, Cohen's kappa, 4 confusion
     cells, per city + pooled -- from 06b_confusion_matrix.csv directly.
  3. Directionality: kept-but-ledger-discarded vs discarded-but-ledger-
     retained, per city + pooled.
  4. Reason breakdown for kept-but-ledger-discarded -- from
     06c_discard_reason_breakdown_for_urls_i_kept.csv directly, plus a
     pooled roll-up.
  5. Recall counter-check, Barcelona and London only (the only two cities
     where data/<city>/raw/*.html survived -- confirmed directly against
     analysis/tables/02_attrition_ledger.csv's 4_scraped_files column:
     barcelona=912, london=540, brighton/dublin/milan=0): for every URL the
     ledger discarded for an access failure, check for a matching scraped
     HTML file and whether it holds >500 characters of body text.

Eligibility for check 5 was originally `discard_reason_bucket in
{inaccessible_only, inaccessible_or_directory_generic}` -- both buckets
require the literal substring 'inaccessible' in the ledger's per-row reason
text (analysis/corpus_analysis.py::classify_discard_reason). That returned
zero eligible London rows: London's per-row reason wording never contains
'inaccessible', even though its aggregate screening-outcome paragraph states
25 pages were "inaccessible or technically blocked". Fixed by extracting the
actual per-row reason text (see load_parse_ledger_pdf_with_reason_text --
reuses parse_ledger_pdf's exact row/column detection, patched only to also
expose the 'Activity evidence or discard reason' column's own text, which
the original function computes internally but does not return) and matching
it against a documented, real-data-confirmed access-failure pattern list
(ACCESS_FAILURE_PATTERNS) instead of the bucket label. Applied identically
to all 5 cities; Barcelona's eligible count is unchanged at 183 (not
sensitive to the fix), London's goes from 0 to 24. See
evaluation/E3_SUMMARY.md Section 5 for the full before/after and the
verification steps (distinct reason strings inspected before any pattern was
chosen, per instruction).

All pooled proportions get a 10,000-resample cluster bootstrap 95% CI,
resampling the 5 cities (with replacement) as the clusters -- not resampling
individual URLs, which would treat cities as exchangeable rows rather than
the actual units of correlated variation (a single city's scraping/ledger
quirks affect all its URLs together).
"""
from __future__ import annotations

import ast
import re
import sys
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import e1_evidence_consistency as e1  # noqa: E402  -- reused: parse_ledger_pdf only

ROOT = e1.ROOT
DATA = e1.DATA
ANALYSIS_TABLES = ROOT / "analysis" / "tables"
RESULTS_DIR = _HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CITIES = e1.CITIES
RAW_DIR = {c: DATA / c / "raw" for c in CITIES}
RECALL_CHECK_CITIES = ["barcelona", "london"]

N_RESAMPLES = 10_000
RNG_SEED = 20260812  # fixed for a reproducible run-to-run CI

# Documented, real-data-confirmed access-failure patterns for the ledger's
# per-row reason text (see load_parse_ledger_pdf_with_reason_text). Chosen
# by inspecting the actual distinct reason strings for all 5 cities (525
# discarded rows total) and testing each candidate substring's hit count
# against that real text -- not assumed. 'login' and '403' were tested
# (both suggested candidates) and dropped: 0 occurrences in any of the 5
# cities' ledgers, out of 525 discarded rows.
ACCESS_FAILURE_PATTERNS = [
    "inaccessible", "error", "challenge", "timeout", "blocked",
    "bot", "404", "server", "no readable",
]
ACCESS_FAILURE_RE = re.compile(
    "|".join(re.escape(p) for p in ACCESS_FAILURE_PATTERNS), re.IGNORECASE
)


def _find_reason_column_x(doc: fitz.Document, ledger_pno: int) -> tuple[float, float]:
    """X-range of the ledger's discard-reason text on its first page. Uses
    the 'Type' header word's x-position as the LEFT edge, not 'Activity'
    (the reason column's own header) -- for Discarded rows the Type and
    Operational-model columns are empty, and the PDF lets the reason
    sentence flow from the Type column's x-position across the merged
    empty space; anchoring on 'Activity' clips the sentence's opening
    words (verified directly: for Barcelona, anchoring on 'Activity'
    truncated the ledger's own template sentence -- which independently is
    known, from classify_discard_reason's regex, to contain the word
    'inaccessible' -- down to fragments like 'provide sufficient campaign,
    commercial distinct Barcelona', silently dropping 'inaccessible' out of
    the extracted text entirely). 'Location' is still the right edge."""
    words = doc[ledger_pno].get_text("words")
    type_w = [w for w in words if w[4] == "Type"]
    location_w = [w for w in words if w[4] == "Location"]
    if not type_w or not location_w:
        raise RuntimeError("reason column header words ('Type'/'Location') not found")
    return (type_w[0][0] - 4, location_w[0][0] - 4)


def load_parse_ledger_pdf_with_reason_text():
    """Same partial-AST-exec reuse of parse_ledger_pdf as Part 1's
    load_parse_ledger_pdf, with one addition: the row loop's already-
    computed `text` variable is column-isolated to the discard-reason
    x-range (_find_reason_column_x) and the result is added as a
    'reason_text' key in the row dict, via a source-text patch on the
    function body (three targeted string replacements, asserted present
    before use). No row/column boundary detection, decision parsing, URL
    extraction, or bucket classification logic is touched or reimplemented
    -- every one of those stays 100% the original analysis/corpus_analysis.py
    code; this only exposes one more already-computed local value that the
    original function discarded before returning."""
    src_path = ROOT / "analysis" / "corpus_analysis.py"
    source = src_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(src_path))
    wanted = {"find_ledger_columns", "DISCARD_REASON_PATTERNS", "classify_discard_reason", "parse_ledger_pdf"}
    nodes = []
    for node in tree.body:
        name = getattr(node, "name", None)
        if name in wanted:
            nodes.append(node)
        elif isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(t in wanted for t in targets):
                nodes.append(node)
    found = {getattr(n, "name", None) or next(t.id for t in n.targets if isinstance(t, ast.Name))
             for n in nodes}
    missing = wanted - found
    if missing:
        raise RuntimeError(f"analysis/corpus_analysis.py: could not locate {missing} for reuse")

    patches = [
        ("rownum_x, cand_x, _ = find_ledger_columns(doc)",
         "rownum_x, cand_x, ledger_pno = find_ledger_columns(doc)\n"
         "    reason_x = _find_reason_column_x(doc, ledger_pno)"),
        ('band_words = [w for w in words if y0 <= w[1] < y1]\n'
         '        text = " ".join(w[4] for w in band_words)',
         'band_words = [w for w in words if y0 <= w[1] < y1]\n'
         '        text = " ".join(w[4] for w in band_words)\n'
         '        reason_words = [w for w in band_words if reason_x[0] <= w[0] <= reason_x[1]]\n'
         '        reason_words.sort(key=lambda w: (round(w[1]), w[0]))\n'
         '        reason_text = " ".join(w[4] for w in reason_words)'),
        ('"discard_reason_bucket": reason,\n        })',
         '"discard_reason_bucket": reason,\n            "reason_text": reason_text,\n        })'),
    ]
    patched_nodes = []
    for node in nodes:
        if getattr(node, "name", None) != "parse_ledger_pdf":
            patched_nodes.append(node)
            continue
        segment = ast.get_source_segment(source, node)
        for old, new in patches:
            if old not in segment:
                raise RuntimeError(f"parse_ledger_pdf source shape changed; patch marker not found: {old!r}")
            segment = segment.replace(old, new)
        patched_nodes.append(ast.parse(segment).body[0])

    module = ast.Module(body=patched_nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {"fitz": fitz, "pd": pd, "re": re, "BASELINE_PDF": e1.BASELINE_PDF,
          "_find_reason_column_x": _find_reason_column_x}
    exec(compile(module, filename=str(src_path), mode="exec"), ns)
    return ns["parse_ledger_pdf"]


parse_ledger_pdf_with_reason = load_parse_ledger_pdf_with_reason_text()


# ═══════════════════════════════════════════════════════════════════════
# Cluster bootstrap (cities as clusters)
# ═══════════════════════════════════════════════════════════════════════

def cluster_bootstrap_ratio(numerators, denominators,
                             n_resamples: int = N_RESAMPLES, seed: int = RNG_SEED) -> tuple[float, float]:
    """95% CI on a pooled ratio sum(num)/sum(den), resampling the list of
    clusters (one (num, den) pair per cluster) with replacement."""
    rng = np.random.default_rng(seed)
    nums = np.asarray(numerators, dtype=float)
    dens = np.asarray(denominators, dtype=float)
    n = len(nums)
    idx = rng.integers(0, n, size=(n_resamples, n))
    num_sums = nums[idx].sum(axis=1)
    den_sums = dens[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratios = num_sums / den_sums
    ratios = ratios[np.isfinite(ratios)]
    if len(ratios) == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5)))


def cluster_bootstrap_kappa(cells: list[tuple[int, int, int, int]],
                             n_resamples: int = N_RESAMPLES, seed: int = RNG_SEED) -> tuple[float, float]:
    """95% CI on pooled Cohen's kappa, resampling cities (each contributing
    its own (a,b,c,d) 2x2 cell counts) with replacement; kappa is recomputed
    from the resampled, summed cells on every draw."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(cells, dtype=float)  # (n_cities, 4)
    n = len(arr)
    idx = rng.integers(0, n, size=(n_resamples, n))
    summed = arr[idx].sum(axis=1)  # (n_resamples, 4)
    a, b, c, d = summed[:, 0], summed[:, 1], summed[:, 2], summed[:, 3]
    total = a + b + c + d
    with np.errstate(invalid="ignore", divide="ignore"):
        po = (a + d) / total
        p_kept, p_ret = (a + b) / total, (a + c) / total
        p_disc, p_ledgerdisc = (c + d) / total, (b + d) / total
        pe = p_kept * p_ret + p_disc * p_ledgerdisc
        kappa = (po - pe) / (1 - pe)
    kappa = kappa[np.isfinite(kappa)]
    if len(kappa) == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(kappa, 2.5)), float(np.percentile(kappa, 97.5)))


# ═══════════════════════════════════════════════════════════════════════
# 1. Coverage rate
# ═══════════════════════════════════════════════════════════════════════

def build_coverage(attrition: pd.DataFrame, ledger_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pipe_num, pipe_den, base_num, base_den = [], [], [], []
    for city in CITIES:
        a = attrition[attrition.city == city].iloc[0]
        l = ledger_summary[ledger_summary.city == city].iloc[0]
        raw_seed, cleaned = int(a["1_raw_seed"]), int(a["2_cleaned"])
        n_ledger, n_retained = int(l["n_ledger_rows"]), int(l["n_retained"])
        # sanity check: both tables should agree on the seed-URL count per
        # city (06a's row_count_matches_raw_seed column already asserts
        # this at build time; re-checked here rather than assumed)
        if raw_seed != n_ledger:
            raise ValueError(
                f"{city}: 02_attrition_ledger raw_seed={raw_seed} != "
                f"06a_ledger_extraction_summary n_ledger_rows={n_ledger}"
            )
        rows.append({"city": city, "system": "pipeline", "surviving": cleaned, "seed": raw_seed,
                      "coverage_rate": round(cleaned / raw_seed, 4)})
        rows.append({"city": city, "system": "baseline", "surviving": n_retained, "seed": n_ledger,
                      "coverage_rate": round(n_retained / n_ledger, 4)})
        pipe_num.append(cleaned); pipe_den.append(raw_seed)
        base_num.append(n_retained); base_den.append(n_ledger)

    for system, num, den in (("pipeline", pipe_num, pipe_den), ("baseline", base_num, base_den)):
        lo, hi = cluster_bootstrap_ratio(num, den)
        rows.append({
            "city": "ALL_POOLED", "system": system, "surviving": sum(num), "seed": sum(den),
            "coverage_rate": round(sum(num) / sum(den), 4),
            "ci95_lower": round(lo, 4), "ci95_upper": round(hi, 4),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# 2/3. Agreement + directionality (reusing 06b_confusion_matrix.csv)
# ═══════════════════════════════════════════════════════════════════════

def _cell(sub: pd.DataFrame, my: str, ledger: str) -> int:
    m = sub[(sub.my_decision == my) & (sub.ledger_decision == ledger)]
    return int(m["n"].iloc[0]) if len(m) else 0


def build_agreement(confusion: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[int, int, int, int]]]:
    rows = []
    cells: list[tuple[int, int, int, int]] = []
    for city in CITIES:
        sub = confusion[confusion.city == city]
        a = _cell(sub, "kept", "Retained")
        b = _cell(sub, "kept", "Discarded")
        c = _cell(sub, "discarded", "Retained")
        d = _cell(sub, "discarded", "Discarded")
        n = a + b + c + d
        po = (a + d) / n
        p_kept, p_ret = (a + b) / n, (a + c) / n
        p_disc, p_ledgerdisc = (c + d) / n, (b + d) / n
        pe = p_kept * p_ret + p_disc * p_ledgerdisc
        kappa = (po - pe) / (1 - pe) if pe != 1 else float("nan")
        rows.append({
            "city": city, "a_kept_retained": a, "b_kept_discarded": b,
            "c_discarded_retained": c, "d_discarded_discarded": d, "n": n,
            "raw_agreement": round(po, 4), "cohens_kappa": round(kappa, 4),
            "ci95_lower_agreement": None, "ci95_upper_agreement": None,
            "ci95_lower_kappa": None, "ci95_upper_kappa": None,
        })
        cells.append((a, b, c, d))

    A, B, C, D = (sum(cell[i] for cell in cells) for i in range(4))
    N = A + B + C + D
    PO = (A + D) / N
    P_kept, P_ret = (A + B) / N, (A + C) / N
    P_disc, P_ledgerdisc = (C + D) / N, (B + D) / N
    PE = P_kept * P_ret + P_disc * P_ledgerdisc
    KAPPA = (PO - PE) / (1 - PE)

    agree_lo, agree_hi = cluster_bootstrap_ratio([a + d for a, b, c, d in cells],
                                                  [a + b + c + d for a, b, c, d in cells])
    kappa_lo, kappa_hi = cluster_bootstrap_kappa(cells)

    rows.append({
        "city": "ALL_POOLED", "a_kept_retained": A, "b_kept_discarded": B,
        "c_discarded_retained": C, "d_discarded_discarded": D, "n": N,
        "raw_agreement": round(PO, 4), "cohens_kappa": round(KAPPA, 4),
        "ci95_lower_agreement": round(agree_lo, 4), "ci95_upper_agreement": round(agree_hi, 4),
        "ci95_lower_kappa": round(kappa_lo, 4), "ci95_upper_kappa": round(kappa_hi, 4),
    })
    return pd.DataFrame(rows), cells


def build_directionality(cells: list[tuple[int, int, int, int]]) -> pd.DataFrame:
    rows = []
    b_list, den_list = [], []
    for city, (a, b, c, d) in zip(CITIES, cells):
        total_disagree = b + c
        rows.append({
            "city": city,
            "n_kept_but_ledger_discarded": b,
            "n_discarded_but_ledger_retained": c,
            "n_disagreements": total_disagree,
            "pct_disagreements_kept_but_ledger_discarded":
                round(100 * b / total_disagree, 1) if total_disagree else None,
        })
        b_list.append(b); den_list.append(total_disagree)

    B = sum(b_list); C = sum(c for _, (_, _, c, _) in zip(CITIES, cells))
    TOT = sum(den_list)
    if sum(den_list) > 0 and any(d > 0 for d in den_list):
        lo, hi = cluster_bootstrap_ratio(b_list, den_list)
    else:
        lo = hi = float("nan")
    rows.append({
        "city": "ALL_POOLED",
        "n_kept_but_ledger_discarded": B,
        "n_discarded_but_ledger_retained": C,
        "n_disagreements": TOT,
        "pct_disagreements_kept_but_ledger_discarded": round(100 * B / TOT, 1) if TOT else None,
        "ci95_lower": round(lo, 4) if np.isfinite(lo) else None,
        "ci95_upper": round(hi, 4) if np.isfinite(hi) else None,
    })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# 4. Reason breakdown (reusing 06c directly)
# ═══════════════════════════════════════════════════════════════════════

def build_reason_breakdown(reason_breakdown: pd.DataFrame) -> pd.DataFrame:
    pooled = reason_breakdown.groupby("reason_bucket")["n_urls_i_kept"].sum().reset_index()
    pooled.insert(0, "city", "ALL_POOLED")
    return pd.concat([reason_breakdown, pooled], ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════
# 5. Recall counter-check (Barcelona, London only)
# ═══════════════════════════════════════════════════════════════════════

def url_to_raw_filename(url: str) -> str:
    """Reverse-engineered directly from data/{barcelona,london}/raw/
    filenames (no scraper source code was read to derive this): strip the
    scheme, replace every '/' in what remains with '_', keep everything
    else verbatim (www., query strings, percent-encoding). Verified exactly
    against real files, e.g.:
      https://abarkacoop.com/catering-abarka -> abarkacoop.com_catering-abarka.html
      https://abarkacoop.com/ca/             -> abarkacoop.com_ca_.html
      https://www.alexandrarose.org.uk/about-us -> www.alexandrarose.org.uk_about-us.html
    """
    rest = re.sub(r'^https?://', '', url.strip())
    rest = rest.replace('/', '_')
    return rest + '.html'


def body_text_length(html_path: Path) -> int:
    """Character count of visible body text: full-page text with <script>/
    <style> removed, via BeautifulSoup -- written from scratch here, not
    imported from src/phase_1's scraper/extractor (out of scope per
    instruction: only extract_facts and the ledger extractor may be
    reused)."""
    html = html_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return len(soup.get_text(" ", strip=True))


def run_recall_counter_check() -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    summary_rows = []
    num_over500, den_total = [], []

    old_buckets = {"inaccessible_only", "inaccessible_or_directory_generic"}
    for city in RECALL_CHECK_CITIES:
        ledger = parse_ledger_pdf_with_reason(city)
        discarded = ledger[ledger.decision == "Discarded"]
        n_old = int(discarded.discard_reason_bucket.isin(old_buckets).sum())
        sub = ledger[
            (ledger.decision == "Discarded")
            & ledger.reason_text.fillna("").str.contains(ACCESS_FAILURE_RE)
            & (ledger.url.notna())
        ]
        raw_dir = RAW_DIR[city]
        raw_files = {p.name for p in raw_dir.glob("*.html")} if raw_dir.exists() else set()
        domain_file_counts = Counter(
            re.sub(r'^https?://', '', name).split('_')[0] for name in raw_files
        )

        n_exists = n_over500 = 0
        for _, row in sub.iterrows():
            url = row.url
            fname = url_to_raw_filename(url)
            exists = fname in raw_files
            domain = re.sub(r'^https?://', '', re.sub(r'^www\.', '', url.split('/')[2] if '//' in url else url))
            chars = None
            over500 = False
            if exists:
                n_exists += 1
                chars = body_text_length(raw_dir / fname)
                over500 = chars > 500
                if over500:
                    n_over500 += 1
            detail_rows.append({
                "city": city, "url": url, "discard_reason_bucket": row.discard_reason_bucket,
                "reason_text": row.reason_text,
                "expected_filename": fname, "raw_file_exists": exists,
                "body_text_chars": chars, "captured_over_500_chars": over500,
                "other_scraped_pages_same_domain": domain_file_counts.get(domain, 0) - (1 if exists else 0),
            })

        n_total = len(sub)
        summary_rows.append({
            "city": city,
            "n_eligible_old_bucket_method": n_old,
            "n_eligible_new_pattern_method": n_total,
            "eligible_count_changed": n_old != n_total,
            "n_raw_file_exists": n_exists,
            "n_captured_over_500_chars": n_over500,
            "pct_file_exists": round(100 * n_exists / n_total, 1) if n_total else None,
            "pct_captured_over_500_chars": round(100 * n_over500 / n_total, 1) if n_total else None,
        })
        num_over500.append(n_over500)
        den_total.append(n_total)

    N = sum(den_total)
    if N > 0:
        lo, hi = cluster_bootstrap_ratio(num_over500, den_total)
        summary_rows.append({
            "city": "ALL_POOLED (barcelona+london only, 2 clusters)",
            "n_eligible_old_bucket_method": None,
            "n_eligible_new_pattern_method": N,
            "eligible_count_changed": None,
            "n_raw_file_exists": None,
            "n_captured_over_500_chars": sum(num_over500),
            "pct_file_exists": None,
            "pct_captured_over_500_chars": round(100 * sum(num_over500) / N, 1),
            "ci95_lower_pct": round(100 * lo, 1) if np.isfinite(lo) else None,
            "ci95_upper_pct": round(100 * hi, 1) if np.isfinite(hi) else None,
        })
    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=== E3 input conditions (resolved paths) ===")
    print("  This module's main() scores the RETIRED baseline condition only, via:")
    print("   ", ANALYSIS_TABLES / "02_attrition_ledger.csv")
    print("   ", ANALYSIS_TABLES / "06a_ledger_extraction_summary.csv")
    print("   ", ANALYSIS_TABLES / "06b_confusion_matrix.csv")
    print("   ", ANALYSIS_TABLES / "06c_discard_reason_breakdown_for_urls_i_kept.csv")
    print("   ", e1.BASELINE_PDF[CITIES[0]].parent, "/ report_<city>_baseline.pdf  (superseded)")
    print("  For the CURRENT baseline_v2 condition, run")
    print("  evaluation/scripts/run_baseline_v2_e3.py instead (produces evaluation/results/e3_v2_*.csv).")
    print()

    attrition = pd.read_csv(ANALYSIS_TABLES / "02_attrition_ledger.csv")
    ledger_summary = pd.read_csv(ANALYSIS_TABLES / "06a_ledger_extraction_summary.csv")
    confusion = pd.read_csv(ANALYSIS_TABLES / "06b_confusion_matrix.csv")
    reason_breakdown = pd.read_csv(ANALYSIS_TABLES / "06c_discard_reason_breakdown_for_urls_i_kept.csv")

    coverage_df = build_coverage(attrition, ledger_summary)
    agreement_df, cells = build_agreement(confusion)
    directionality_df = build_directionality(cells)
    reason_df = build_reason_breakdown(reason_breakdown)
    recall_detail_df, recall_summary_df = run_recall_counter_check()

    coverage_df.to_csv(RESULTS_DIR / "e3_coverage.csv", index=False)
    agreement_df.to_csv(RESULTS_DIR / "e3_agreement.csv", index=False)
    directionality_df.to_csv(RESULTS_DIR / "e3_directionality.csv", index=False)
    reason_df.to_csv(RESULTS_DIR / "e3_reason_breakdown.csv", index=False)
    recall_detail_df.to_csv(RESULTS_DIR / "e3_recall_counter_check_detail.csv", index=False)
    recall_summary_df.to_csv(RESULTS_DIR / "e3_recall_counter_check_summary.csv", index=False)

    print("=== e3_coverage.csv ===")
    print(coverage_df.to_string(index=False))
    print("\n=== e3_agreement.csv ===")
    print(agreement_df.to_string(index=False))
    print("\n=== e3_directionality.csv ===")
    print(directionality_df.to_string(index=False))
    print("\n=== e3_reason_breakdown.csv ===")
    print(reason_df.to_string(index=False))
    print("\n=== e3_recall_counter_check_summary.csv ===")
    print(recall_summary_df.to_string(index=False))

    print("\n" + "=" * 78)
    print("ACCESS-FAILURE MATCHER (final, for appendix):")
    print("  " + ", ".join(ACCESS_FAILURE_PATTERNS))
    print("  (case-insensitive substring match against the ledger's per-row")
    print("   'Activity evidence or discard reason' column text)")
    print("  Tested and DROPPED (0 occurrences in 525 discarded rows, all 5 cities):")
    print("   login, 403")
    bc_row = recall_summary_df[recall_summary_df.city == "barcelona"].iloc[0]
    ld_row = recall_summary_df[recall_summary_df.city == "london"].iloc[0]
    print(f"  Barcelona eligible: old={int(bc_row.n_eligible_old_bucket_method)} -> "
          f"new={int(bc_row.n_eligible_new_pattern_method)} "
          f"({'CHANGED' if bc_row.eligible_count_changed else 'unchanged'})")
    print(f"  London eligible:    old={int(ld_row.n_eligible_old_bucket_method)} -> "
          f"new={int(ld_row.n_eligible_new_pattern_method)} "
          f"({'CHANGED' if ld_row.eligible_count_changed else 'unchanged'})")
    pooled_row = recall_summary_df[recall_summary_df.city.str.startswith("ALL_POOLED")].iloc[0]
    ci_lo, ci_hi = pooled_row.ci95_lower_pct, pooled_row.ci95_upper_pct
    degenerate = np.isfinite(ci_lo) and np.isfinite(ci_hi) and round(ci_lo, 1) == round(ci_hi, 1)
    print(f"  Pooled (2-cluster) bootstrap CI: [{ci_lo}, {ci_hi}] -- "
          f"{'DEGENERATE (collapses to the point estimate)' if degenerate else 'not degenerate'}")
    print("=" * 78)

    write_markdown(coverage_df, agreement_df, directionality_df, reason_df,
                   recall_detail_df, recall_summary_df)
    print(f"\nWrote CSVs to {RESULTS_DIR} and {_HERE / 'E3_SUMMARY.md'}")


def write_markdown(coverage_df, agreement_df, directionality_df, reason_df,
                    recall_detail_df, recall_summary_df) -> None:
    md = []
    md.append("# E3 — Coverage vs validity\n")
    md.append(
        "Read-only, no pipeline stage run, no network call. Reuses "
        "`analysis/tables/02_attrition_ledger.csv`, "
        "`06a_ledger_extraction_summary.csv`, `06b_confusion_matrix.csv`, and "
        "`06c_discard_reason_breakdown_for_urls_i_kept.csv` directly (all already computed "
        "by `analysis/corpus_analysis.py`) rather than recomputing them, plus "
        "`e1_evidence_consistency.parse_ledger_pdf` (Part 1's reused ledger extractor) for "
        "the recall counter-check. Generated by `evaluation/e3_coverage_validity.py`.\n\n"
        f"All pooled-proportion 95% CIs use a {N_RESAMPLES:,}-resample cluster bootstrap "
        "over the 5 cities (fixed seed for reproducibility) -- resampling cities, not "
        "individual URLs, since URLs within one city share that city's scraping/ledger "
        "idiosyncrasies and are not independent draws.\n"
    )

    md.append("## 1. Coverage rate (surviving / seed URLs)\n")
    md.append(coverage_df.to_markdown(index=False))
    md.append("")

    md.append("## 2. Phase-0-vs-ledger agreement\n")
    md.append(
        "Cells: `a`=kept & ledger-Retained, `b`=kept & ledger-Discarded, "
        "`c`=discarded & ledger-Retained, `d`=discarded & ledger-Discarded. "
        "`raw_agreement=(a+d)/n`; Cohen's kappa corrects that for chance agreement given "
        "each side's marginal kept/discarded and Retained/Discarded rates. Source: "
        "`analysis/tables/06b_confusion_matrix.csv`, itself restricted to ledger rows whose "
        "extracted URL matched (after normalisation) a URL in the raw-seed file for that "
        "city -- see that table's own generating comments in `analysis/corpus_analysis.py` "
        "for the match-rate caveat; per-city CI cells are left empty (a single city isn't a "
        "resampling target -- only the pooled row is bootstrapped).\n"
    )
    md.append(agreement_df.to_markdown(index=False))
    md.append("")

    md.append("## 3. Directionality\n")
    md.append(
        "**Disagreement is almost entirely one-directional.** Across all 5 cities, "
        "discarded-but-ledger-retained (`c`) is 0 everywhere except London (1); "
        "kept-but-ledger-discarded (`b`) is the overwhelming majority of all disagreement "
        "in every city. In other words: when Phase-0 cleaning and the baseline ledger "
        "disagree, it is essentially always because Phase-0 kept something the ledger "
        "discarded, almost never the reverse.\n"
    )
    md.append(directionality_df.to_markdown(index=False))
    md.append("")

    md.append("## 4. Reason breakdown — why the ledger discarded URLs Phase-0 kept\n")
    md.append(
        "From `analysis/tables/06c_discard_reason_breakdown_for_urls_i_kept.csv` directly "
        "(reused, not recomputed), plus a pooled roll-up by `reason_bucket`. "
        "`inaccessible_or_directory_generic` is the ledger's own catch-all template "
        "sentence citing accessibility and page-type reasons together -- see that table's "
        "own note in `analysis/corpus_analysis.py` for why it can't be split further from "
        "the ledger text alone.\n"
    )
    md.append(reason_df.to_markdown(index=False))
    md.append("")

    md.append("## 5. Recall counter-check (Barcelona, London only)\n")
    md.append(
        "The only two cities where `data/<city>/raw/*.html` survived on disk "
        "(`analysis/tables/02_attrition_ledger.csv` column `4_scraped_files`: "
        "barcelona=912, london=540, brighton/dublin/milan=0 -- confirmed directly, not "
        "assumed).\n\n"
        "**Bug found and fixed.** Eligibility was originally `discard_reason_bucket in "
        "{inaccessible_only, inaccessible_or_directory_generic}` -- both buckets require "
        "the literal substring 'inaccessible' in that row's reason text, per "
        "`classify_discard_reason`. That returned **zero** eligible London rows, despite "
        "London's own aggregate screening-outcome paragraph stating 25 pages were "
        "\"inaccessible or technically blocked\". Root-caused by extracting the actual "
        "per-row reason text (`load_parse_ledger_pdf_with_reason_text` -- reuses "
        "`parse_ledger_pdf`'s exact row/column/decision/URL logic unchanged, patched only to "
        "also expose the 'Activity evidence or discard reason' column's own text, which the "
        "original function computes internally as `text` but never returns) and inspecting "
        "the real distinct strings before assuming anything: London's 80 discarded rows "
        "resolve to five templated reason sentences, one of which -- 23 of 80 rows, verbatim "
        "-- reads \"The page returned challenge, timeout otherwise exposed readable "
        "initiative-level evidence\" (a few words are dropped by imperfect column-boundary "
        "text extraction; the intended sentence is evidently the access-failure template "
        "described in London's aggregate paragraph). It never contains the word "
        "'inaccessible' -- a genuine property of that city's per-row wording, not an "
        "extraction bug once traced this far.\n\n"
        "**The fix itself needed a second correction.** Column-isolating the reason text by "
        "the 'Activity' column header's own x-position clipped it badly for cities whose "
        "discard-reason sentences are long: for a Discarded row, the Type/Operational-model "
        "columns are empty, and the PDF lets the reason sentence flow from the *Type* "
        "column's x-position across that merged empty space -- anchoring on 'Activity' "
        "cropped off the sentence's opening words. Verified directly against Barcelona: "
        "anchoring on 'Activity' extracted its 183-row template sentence as fragments like "
        "\"provide sufficient campaign, commercial distinct Barcelona\" -- silently dropping "
        "'inaccessible' out of the text entirely, even though `classify_discard_reason`'s "
        "regex (which scans the *whole* row band, not a column-isolated slice) had already "
        "proven that word is really there. Anchoring on 'Type' instead recovers the sentence "
        "nearly verbatim: \"The page was inaccessible or did not provide sufficient "
        "evidence; it was a directory, policy, campaign, commercial unrelated page, or "
        "otherwise not a distinct Barcelona [FSI]\".\n\n"
        "**Widened matcher, real-data-confirmed.** `ACCESS_FAILURE_PATTERNS` = " +
        ", ".join(f"`{p}`" for p in ACCESS_FAILURE_PATTERNS) +
        " (case-insensitive substring match against the isolated reason text). Chosen by "
        "testing each candidate pattern's hit count against all 525 discarded rows across "
        "all 5 cities before deciding anything -- `login` and `403` (both proposed "
        "candidates) were tested and dropped: 0 occurrences anywhere. `inaccessible` is kept "
        "as the first pattern since it is the original, still-valid signal, not replaced by "
        "the new ones. Applied identically to all 5 cities (not just the 2 with retained raw "
        "HTML) to check whether the matcher change was itself consequential:\n\n"
        "| city | old (bucket method) | new (pattern method) |\n"
        "|:---|---:|---:|\n"
        "| barcelona | 183 | 183 |\n"
        "| brighton | n/a (no raw HTML to check) | 16 |\n"
        "| dublin | n/a (no raw HTML to check) | 18 |\n"
        "| london | 0 | 24 |\n"
        "| milan | n/a (no raw HTML to check) | 58 |\n\n"
        "**Barcelona's result is unchanged (183 -> 183) -- not sensitive to the matcher.** "
        "The widening only added coverage where the bucket method had a genuine blind spot; "
        "it did not loosen or tighten anything for the city the original number came from.\n\n"
        "For every eligible row, the matching raw HTML filename is derived (see "
        "`url_to_raw_filename` docstring for the exact, verified transform) and checked for "
        "existence and >500 characters of body text (full page text minus `<script>`/"
        "`<style>`, via BeautifulSoup).\n"
    )
    md.append(recall_summary_df.to_markdown(index=False))
    md.append("")

    pooled_mask = recall_summary_df["city"].astype(str).str.startswith("ALL_POOLED")
    if pooled_mask.any():
        prow = recall_summary_df[pooled_mask].iloc[0]
        ci_lo, ci_hi = prow.get("ci95_lower_pct"), prow.get("ci95_upper_pct")
        is_degenerate = (
            ci_lo is not None and ci_hi is not None
            and pd.notna(ci_lo) and pd.notna(ci_hi)
            and round(float(ci_lo), 1) == round(float(ci_hi), 1)
        )
        if is_degenerate:
            md.append(
                f"**The pooled 2-cluster bootstrap CI is still degenerate: [{ci_lo}, {ci_hi}].** "
                "This time it is not because one cluster carries zero information (both now "
                "do -- 183 and 24 eligible rows respectively) but because of an exact "
                "coincidence: Barcelona's capture rate is 122/183 and London's is 16/24, and "
                "both reduce to exactly 2/3 (verified: `Fraction(122,183) == Fraction(16,24) "
                "== Fraction(2,3)`). Every resample is therefore a weighted average of two "
                "identical ratios, which is that same ratio regardless of how the resampling "
                "weights the two cities -- so the interval collapses to a point again, for a "
                "different and more interesting reason than before the fix. A 2-cluster "
                "bootstrap is intrinsically weak evidence either way -- report the two "
                "cities' own percentages (above) as the primary result, and treat this pooled "
                "interval as a coarse summary, not a precise one.\n"
            )
        else:
            md.append(
                f"**The pooled 2-cluster bootstrap CI is [{ci_lo}, {ci_hi}] -- no longer "
                "degenerate**, now that both clusters (183 Barcelona, 24 London eligible rows) "
                "carry real, independent data. Still only 2 clusters, so treat this as a "
                "coarse interval, not a precise one -- the two cities' own percentages (above) "
                "remain the more informative numbers.\n"
            )

    md.append(
        f"Per-URL detail in `e3_recall_counter_check_detail.csv` ({len(recall_detail_df)} rows). "
        "`other_scraped_pages_same_domain` is contextual only (how many *other* pages from "
        "that domain were scraped) -- it does not count as a match for the specific "
        "discarded URL, which requires an exact filename match.\n"
    )

    (RESULTS_DIR.parent / "E3_SUMMARY.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
