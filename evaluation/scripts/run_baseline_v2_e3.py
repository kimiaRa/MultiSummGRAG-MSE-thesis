#!/usr/bin/env python3
"""
Re-run E3 (coverage vs validity) against the 5 baseline_v2 reports
(report_<city>_baseline_v2.html), which now embed a per-URL "Screening
Ledger" <table> directly in the HTML (no separate appendix PDF ledger).

Read-only apart from NEW outputs:
  evaluation/results/e3_v2_ledger_validation.csv
  evaluation/results/e3_v2_coverage.csv
  evaluation/results/e3_v2_agreement.csv
  evaluation/results/e3_v2_directionality.csv
  evaluation/results/e3_v2_reason_breakdown.csv
  evaluation/results/e3_v2_recall_counter_check_detail.csv
  evaluation/results/e3_v2_recall_counter_check_summary.csv
The original e3_*.csv files (and E3_SUMMARY.md) are never opened for
writing, only e3_coverage_validity.py's already-computed CONSTANTS and
generic (city-agnostic) helper FUNCTIONS are imported and reused.

── What's reused verbatim vs rebuilt ──────────────────────────────────────
Reused, unmodified, by import:
  - e3_coverage_validity.cluster_bootstrap_ratio / cluster_bootstrap_kappa
    (identical 10,000-resample cluster-over-cities bootstrap, same
    N_RESAMPLES=10_000 and RNG_SEED, imported not retyped)
  - e3_coverage_validity.build_agreement / build_directionality /
    build_reason_breakdown -- these three take only a confusion_df /
    reason_breakdown_df / cell list as input and never hardcode "baseline"
    anywhere in their logic (only build_coverage's row-labelling does), so
    they are called directly on baseline_v2's own confusion matrix, exactly
    the same functions the original baseline system's numbers came from.
  - e3_coverage_validity.ACCESS_FAILURE_PATTERNS / ACCESS_FAILURE_RE /
    url_to_raw_filename / body_text_length / RAW_DIR / RECALL_CHECK_CITIES
    for the recall counter-check (step 3) -- identical matcher, identical
    raw-HTML-filename derivation, identical >500-char body-text rule.

Rebuilt (not reusable as-is, but same source data / same method):
  - Ledger extraction: baseline_v2 has no PDF, so e1.parse_ledger_pdf /
    e3.parse_ledger_pdf_with_reason (both fitz-based) do not apply. Parsed
    directly from each report's own <table> (see parse_ledger_html below) --
    columns No./Number, URL, Decision, Reason, one row per candidate URL.
    The "Reason" cell IS the per-row reason text already (baseline_v2's
    ledger states a closed 6-value vocabulary directly -- no PDF
    column-x-position patch needed to recover it, unlike the original
    baseline's free-text sentences).
  - normalize_url() / read_url_csv(): reused via the same partial-AST-exec
    technique e1/e3 already use for parse_ledger_pdf -- pulls those two
    self-contained functions (and nothing else) straight out of
    analysis/corpus_analysis.py's source, so the URL-normalisation and
    raw/cleaned-CSV-reading logic used for the confusion matrix is
    byte-identical to what built 06b_confusion_matrix.csv originally, not a
    reimplementation.
  - build_coverage(): the original hardcodes system="pipeline"/"baseline"
    row labels, so it is not called directly; a small local version
    (build_coverage_v2) mirrors its arithmetic exactly (surviving/seed,
    cluster_bootstrap_ratio import reused) but labels the new system
    "baseline_v2".

── Step 1: ledger-vs-stated-outcome validation ────────────────────────────
Per instruction: parse each ledger, print rows extracted / retained /
discarded / the report's own "Screening outcome: N of M" line, and these
must agree -- rows extracted == M, retained-count == N. Any city that fails
this check is excluded from every step below ("stop for that city"), not
silently included with bad data.

── Step 3: recall counter-check ───────────────────────────────────────────
Eligibility is recomputed against baseline_v2's OWN per-row "Reason" text
using the same ACCESS_FAILURE_RE substring matcher as before (see
e3_coverage_validity.py's ACCESS_FAILURE_PATTERNS) -- but where the
original baseline's ledger used one shared, long template sentence per
discard bucket (so a single "inaccessible"-containing template sentence
could cover very different underlying reasons at once), baseline_v2's
"Reason" cells are already a closed 6-value vocabulary (duplicate /
outside the city / activity ended / inaccessible / directory or policy
page / not an FSI) with no combined template -- so eligibility here comes
down, almost by construction, to reason == "inaccessible" (the only
category value matching ACCESS_FAILURE_RE, checked directly against all
six values, not assumed).

Run from the repo root with the project venv active:
    venv/bin/python3 evaluation/scripts/run_baseline_v2_e3.py
"""
from __future__ import annotations

import ast
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

_HERE = Path(__file__).resolve().parent
_EVAL_DIR = _HERE.parent
sys.path.insert(0, str(_EVAL_DIR))
import e1_evidence_consistency as e1        # noqa: E402
import e3_coverage_validity as e3            # noqa: E402 -- imports only; main() never called

ROOT = e1.ROOT
DATA = e1.DATA
CITIES = e1.CITIES
RESULTS_DIR = _EVAL_DIR / "results"
ANALYSIS_TABLES = ROOT / "analysis" / "tables"

BASELINE_V2_HTML = {c: DATA / c / "output" / f"report_{c}_baseline_v2.html" for c in CITIES}
for c in CITIES:
    if not BASELINE_V2_HTML[c].exists():
        raise FileNotFoundError(f"baseline_v2 report missing for {c}: {BASELINE_V2_HTML[c]}")

N_RESAMPLES = e3.N_RESAMPLES
RNG_SEED = e3.RNG_SEED
RECALL_CHECK_CITIES = e3.RECALL_CHECK_CITIES  # ["barcelona", "london"]
ACCESS_FAILURE_RE = e3.ACCESS_FAILURE_RE
ACCESS_FAILURE_PATTERNS = e3.ACCESS_FAILURE_PATTERNS
url_to_raw_filename = e3.url_to_raw_filename
body_text_length = e3.body_text_length
RAW_DIR = e3.RAW_DIR
cluster_bootstrap_ratio = e3.cluster_bootstrap_ratio
cluster_bootstrap_kappa = e3.cluster_bootstrap_kappa
build_agreement = e3.build_agreement
build_directionality = e3.build_directionality
build_reason_breakdown = e3.build_reason_breakdown

# Same, user-confirmed raw/cleaned file choices as analysis/corpus_analysis.py
# (barcelona has no raw_urls.csv on this branch; urls.csv is its documented
# raw-seed proxy -- reused as a path constant, not a new methodological call).
RAW_FILE = {
    "barcelona": DATA / "barcelona" / "urls.csv",
    "brighton": DATA / "brighton" / "raw_urls.csv",
    "dublin": DATA / "dublin" / "raw_urls.csv",
    "london": DATA / "london" / "raw_urls.csv",
    "milan": DATA / "milan" / "raw_urls.csv",
}
CLEANED_FILE = {c: DATA / c / "urls_cleaned.csv" for c in CITIES}


def load_url_helpers():
    """Partial AST exec of analysis/corpus_analysis.py: pulls out only
    read_url_csv() and normalize_url() (both self-contained -- csv/pandas
    and urllib.parse.urlparse respectively), the same reuse technique
    e1_evidence_consistency.py already uses for parse_ledger_pdf. Does not
    execute the rest of that script."""
    src_path = ROOT / "analysis" / "corpus_analysis.py"
    source = src_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(src_path))
    wanted = {"read_url_csv", "normalize_url"}
    nodes = [n for n in tree.body if getattr(n, "name", None) in wanted]
    found = {n.name for n in nodes}
    missing = wanted - found
    if missing:
        raise RuntimeError(f"analysis/corpus_analysis.py: could not locate {missing} for reuse")
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    import csv
    from urllib.parse import urlparse
    ns = {"csv": csv, "pd": pd, "urlparse": urlparse, "Path": Path}
    exec(compile(module, filename=str(src_path), mode="exec"), ns)
    return ns["read_url_csv"], ns["normalize_url"]


read_url_csv, normalize_url = load_url_helpers()

RAW_DF = {c: read_url_csv(RAW_FILE[c]) for c in CITIES if RAW_FILE[c].exists()}
for _df in RAW_DF.values():
    _df["url_norm"] = _df["url"].map(normalize_url)
CLEANED_DF = {c: read_url_csv(CLEANED_FILE[c]) for c in CITIES if CLEANED_FILE[c].exists()}


# ═══════════════════════════════════════════════════════════════════════
# Step 1 — ledger extraction + agreement-with-stated-outcome validation
# ═══════════════════════════════════════════════════════════════════════

STATED_OUTCOME_RE = re.compile(r'Screening outcome:\s*([\d,]+)\s+of\s+([\d,]+)\s+URLs retained', re.IGNORECASE)


def parse_ledger_html(city: str) -> pd.DataFrame:
    """Parse the Screening Ledger <table> directly out of
    report_<city>_baseline_v2.html. Columns: row_id, url, decision
    ('Retained'/'Discarded'/'UNKNOWN'), discard_reason_bucket (slugified:
    lower-case, spaces->underscores), reason_text (literal cell text,
    unslugified -- used for the recall counter-check's substring match)."""
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


def extract_stated_outcome(city: str) -> tuple[int | None, int | None]:
    """Returns (N, M) from the report's own 'Screening outcome: N of M URLs
    retained.' line -- searched over the full rendered HTML text, not just
    extracted prose, so it's found regardless of prose-extraction rules."""
    html = BASELINE_V2_HTML[city].read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True)
    m = STATED_OUTCOME_RE.search(text)
    if not m:
        return None, None
    return int(m.group(1).replace(",", "")), int(m.group(2).replace(",", ""))


def run_ledger_validation() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    ledgers: dict[str, pd.DataFrame] = {}
    rows = []
    print("=== Step 1: ledger extraction + agreement with stated outcome ===")
    for city in CITIES:
        ldf = parse_ledger_html(city)
        n_rows = len(ldf)
        n_retained = int((ldf.decision == "Retained").sum())
        n_discarded = int((ldf.decision == "Discarded").sum())
        n_unknown = int((ldf.decision == "UNKNOWN").sum())
        stated_n, stated_m = extract_stated_outcome(city)
        agree_total = (stated_m == n_rows)
        agree_retained = (stated_n == n_retained)
        ok = agree_total and agree_retained and stated_n is not None
        print(f"{city}: rows_extracted={n_rows} retained={n_retained} discarded={n_discarded} "
              f"unknown={n_unknown} | stated 'Screening outcome: {stated_n} of {stated_m}' | "
              f"{'AGREE' if ok else 'MISMATCH -- STOPPING FOR THIS CITY'}")
        rows.append({
            "city": city, "n_rows_extracted": n_rows, "n_retained": n_retained,
            "n_discarded": n_discarded, "n_unknown": n_unknown,
            "stated_retained_n": stated_n, "stated_total_m": stated_m,
            "rows_match_stated_total": agree_total,
            "retained_matches_stated_n": agree_retained,
            "agree": ok,
        })
        if ok:
            ledgers[city] = ldf
        else:
            print(f"  ** {city} EXCLUDED from all downstream steps (ledger disagrees with its own stated outcome) **")
    validation_df = pd.DataFrame(rows)
    return ledgers, validation_df


# ═══════════════════════════════════════════════════════════════════════
# Step 2a — coverage rate (pipeline unchanged; baseline_v2 new)
# ═══════════════════════════════════════════════════════════════════════

def build_coverage_v2(cities: list[str], ledgers: dict[str, pd.DataFrame]) -> pd.DataFrame:
    attrition = pd.read_csv(ANALYSIS_TABLES / "02_attrition_ledger.csv")
    rows = []
    pipe_num, pipe_den, bv2_num, bv2_den = [], [], [], []
    for city in cities:
        a = attrition[attrition.city == city].iloc[0]
        raw_seed, cleaned = int(a["1_raw_seed"]), int(a["2_cleaned"])
        ldf = ledgers[city]
        n_ledger = len(ldf)
        n_retained = int((ldf.decision == "Retained").sum())
        rows.append({"city": city, "system": "pipeline", "surviving": cleaned, "seed": raw_seed,
                      "coverage_rate": round(cleaned / raw_seed, 4)})
        rows.append({"city": city, "system": "baseline_v2", "surviving": n_retained, "seed": n_ledger,
                      "coverage_rate": round(n_retained / n_ledger, 4)})
        pipe_num.append(cleaned); pipe_den.append(raw_seed)
        bv2_num.append(n_retained); bv2_den.append(n_ledger)

    for system, num, den in (("pipeline", pipe_num, pipe_den), ("baseline_v2", bv2_num, bv2_den)):
        lo, hi = cluster_bootstrap_ratio(num, den, n_resamples=N_RESAMPLES, seed=RNG_SEED)
        rows.append({
            "city": "ALL_POOLED", "system": system, "surviving": sum(num), "seed": sum(den),
            "coverage_rate": round(sum(num) / sum(den), 4),
            "ci95_lower": round(lo, 4), "ci95_upper": round(hi, 4),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# Step 2b — confusion matrix (Phase-0 cleaning vs baseline_v2 ledger)
# ═══════════════════════════════════════════════════════════════════════

def build_confusion_v2(cities: list[str], ledgers: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mirrors analysis/corpus_analysis.py's Section-6 cross-tab block
    (my_decision = kept/discarded from Phase-0's urls_cleaned.csv, matched
    to the ledger's Retained/Discarded by normalised URL) verbatim, applied
    to baseline_v2's ledger instead of the PDF-parsed baseline ledger."""
    confusion_rows = []
    match_rows = []
    for city in cities:
        ldf = ledgers[city]
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
        n_norm_matched = int(matched["norm_match_in_raw"].sum())
        match_rows.append({
            "city": city, "n_ledger_urls_extracted": n_ledger_urls,
            "n_matched_normalised_to_raw_seed": n_norm_matched,
            "match_rate_pct": round(100 * n_norm_matched / n_ledger_urls, 1) if n_ledger_urls else None,
        })

        only_matched = matched[matched["norm_match_in_raw"]]
        for my_dec in ["kept", "discarded"]:
            for ledger_dec in ["Retained", "Discarded"]:
                n = int(((only_matched["my_decision"] == my_dec) & (only_matched["decision"] == ledger_dec)).sum())
                confusion_rows.append({"city": city, "my_decision": my_dec, "ledger_decision": ledger_dec, "n": n})

    return pd.DataFrame(confusion_rows), pd.DataFrame(match_rows)


# ═══════════════════════════════════════════════════════════════════════
# Step 3 — recall counter-check, Barcelona + London
# ═══════════════════════════════════════════════════════════════════════

def run_recall_counter_check_v2(ledgers: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    summary_rows = []
    num_over500, den_total = [], []
    old_eligible = {"barcelona": 183, "london": 24}  # e3_recall_counter_check_summary.csv, post-fix numbers

    for city in RECALL_CHECK_CITIES:
        if city not in ledgers:
            continue
        ledger = ledgers[city]
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
        n_old = old_eligible.get(city)
        summary_rows.append({
            "city": city,
            "n_eligible_old_e3_baseline_v1": n_old,
            "n_eligible_baseline_v2": n_total,
            "eligible_count_changed": (n_old != n_total) if n_old is not None else None,
            "n_raw_file_exists": n_exists,
            "n_captured_over_500_chars": n_over500,
            "pct_file_exists": round(100 * n_exists / n_total, 1) if n_total else None,
            "pct_captured_over_500_chars": round(100 * n_over500 / n_total, 1) if n_total else None,
        })
        num_over500.append(n_over500)
        den_total.append(n_total)

    N = sum(den_total)
    if N > 0 and len(den_total) > 0:
        lo, hi = cluster_bootstrap_ratio(num_over500, den_total, n_resamples=N_RESAMPLES, seed=RNG_SEED)
        summary_rows.append({
            "city": f"ALL_POOLED ({'+'.join(RECALL_CHECK_CITIES)} only, {len(den_total)} clusters)",
            "n_eligible_old_e3_baseline_v1": sum(v for v in old_eligible.values()),
            "n_eligible_baseline_v2": N,
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
    ledgers, validation_df = run_ledger_validation()
    ok_cities = [c for c in CITIES if c in ledgers]
    if len(ok_cities) < len(CITIES):
        print(f"\nProceeding with {len(ok_cities)}/{len(CITIES)} cities: {ok_cities}")
    validation_df.to_csv(RESULTS_DIR / "e3_v2_ledger_validation.csv", index=False)

    coverage_df = build_coverage_v2(ok_cities, ledgers)
    confusion_df, match_df = build_confusion_v2(ok_cities, ledgers)
    agreement_df, cells = build_agreement(confusion_df)  # reused verbatim from e3_coverage_validity
    directionality_df = build_directionality(cells)       # reused verbatim

    reason_breakdown_rows = []
    for city in ok_cities:
        ldf = ledgers[city]
        raw_set_norm = set(RAW_DF[city]["url_norm"]) if city in RAW_DF else set()
        cleaned_set = set(CLEANED_DF[city]["url"]) if city in CLEANED_DF else set()
        cleaned_set_norm = {normalize_url(u) for u in cleaned_set}
        matched = ldf[ldf["url"].notna()].copy()
        matched["norm_match_in_raw"] = matched["url"].map(lambda u: normalize_url(u) in raw_set_norm)
        matched["my_decision"] = matched["url"].map(
            lambda u: "kept" if (u in cleaned_set or normalize_url(u) in cleaned_set_norm) else "discarded")
        only_matched = matched[matched["norm_match_in_raw"]]
        kept_but_ledger_discarded = only_matched[(only_matched["my_decision"] == "kept") &
                                                  (only_matched["decision"] == "Discarded")]
        reason_counts = kept_but_ledger_discarded["discard_reason_bucket"].value_counts()
        for reason, cnt in reason_counts.items():
            reason_breakdown_rows.append({"city": city, "reason_bucket": reason, "n_urls_i_kept": int(cnt)})
    reason_breakdown_input = pd.DataFrame(reason_breakdown_rows) if reason_breakdown_rows else pd.DataFrame(
        columns=["city", "reason_bucket", "n_urls_i_kept"])
    reason_df = build_reason_breakdown(reason_breakdown_input)  # reused verbatim

    recall_detail_df, recall_summary_df = run_recall_counter_check_v2(ledgers)

    coverage_df.to_csv(RESULTS_DIR / "e3_v2_coverage.csv", index=False)
    agreement_df.to_csv(RESULTS_DIR / "e3_v2_agreement.csv", index=False)
    directionality_df.to_csv(RESULTS_DIR / "e3_v2_directionality.csv", index=False)
    reason_df.to_csv(RESULTS_DIR / "e3_v2_reason_breakdown.csv", index=False)
    recall_detail_df.to_csv(RESULTS_DIR / "e3_v2_recall_counter_check_detail.csv", index=False)
    recall_summary_df.to_csv(RESULTS_DIR / "e3_v2_recall_counter_check_summary.csv", index=False)
    match_df.to_csv(RESULTS_DIR / "e3_v2_url_match_rates.csv", index=False)

    print("\n=== e3_v2_coverage.csv ===")
    print(coverage_df.to_string(index=False))
    print("\n=== e3_v2_agreement.csv ===")
    print(agreement_df.to_string(index=False))
    print("\n=== e3_v2_directionality.csv ===")
    print(directionality_df.to_string(index=False))
    print("\n=== e3_v2_reason_breakdown.csv ===")
    print(reason_df.to_string(index=False))
    print("\n=== e3_v2_recall_counter_check_summary.csv ===")
    print(recall_summary_df.to_string(index=False))

    print("\n" + "=" * 78)
    print("Wrote all e3_v2_*.csv files to", RESULTS_DIR)
    print("Original e3_*.csv / E3_SUMMARY.md were not opened for writing.")
    print("=" * 78)


if __name__ == "__main__":
    main()
