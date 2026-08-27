#!/usr/bin/env python3
"""
Run E1 (auditability) and E2 (internal consistency) over the 5 baseline_v2
reports (report_<city>_baseline_v2.html), labelling every new row
system="baseline_v2" and adding it as a fourth system alongside pipeline,
baseline, and b1.

Read-only except for:
  evaluation/results/e1_claims.csv   -- baseline_v2 claim rows appended
  evaluation/results/e1_summary.csv  -- baseline_v2 summary rows appended
  evaluation/results/e2_flags.csv    -- baseline_v2 flag rows appended (VERDICT/NOTE left blank)
  evaluation/results/e2_summary.csv  -- baseline_v2 summary rows appended

Every write below is an APPEND (mode="a", header=False), never a
read-modify-rewrite of the whole file -- the existing pipeline/baseline/b1
rows, including e2_flags.csv's hand-filled VERDICT/NOTE columns, are never
touched, only read (to sanity-check column order before appending).

Explicitly NOT run: e1_evidence_consistency.main() / e2_internal_consistency.main().
Both would regenerate E1_SUMMARY.md / E2_SUMMARY.md from scratch and silently
delete their hand-written sections. This script imports both modules and
calls their functions directly, but never calls either main() and never
writes either SUMMARY.md.

── Evidence base: "as with the original baseline" ────────────────────────
Per instruction, baseline_v2's evidence base is each report's OWN screening
ledger -- the same evidence-base *philosophy* the original baseline system
uses (its retained/discarded ledger, not the pipeline's fsi_enriched.jsonl
facts). It is NOT read via e1.parse_ledger_pdf(): baseline_v2 has no
appendix PDF ledger, only the HTML report itself (the instruction is
explicit: "Use the HTML, not the PDF"), so the ledger is parsed directly out
of each report's own <table> (inside its "Screening Ledger" <h2> section) --
columns No./Number, URL, Decision, Reason (if discarded), one row per
candidate URL, 261/81/83/115/163 rows across the 5 cities (matches each
city's seed URL CSV row count exactly, checked separately). decision is
"Retained" or "Discarded" in every row of every city -- no UNKNOWN rows, so
n_unknown is always 0 here (unlike the PDF baseline's parser, which allows
for a row where neither word was detected).

discard_reason_bucket is built directly from the literal "Reason (if
discarded)" cell text (lower-cased, spaces -> underscores; e.g. "directory
or policy page" -> "directory_or_policy_page"), NOT by reusing the original
baseline's classify_discard_reason() regex classifier. That classifier was
built to bucket free-text explanations scraped off live pages during
baseline browsing; baseline_v2's ledger instead already states a closed,
six-value controlled vocabulary per city (duplicate / outside the city /
activity ended / inaccessible / directory or policy page / not an FSI) --
running it through the old free-text classifier would actually lose
information (e.g. "activity ended" and "not an FSI" both fall through every
regex in DISCARD_REASON_PATTERNS to "other_or_unclassified", collapsing two
categories baseline_v2 keeps distinct). Reading the ledger's own stated
categories directly is a closer match to "the evidence base is each report's
own screening ledger" than forcing it through a classifier built for a
different, messier text format.

Once built, ledger_df (columns "decision", "discard_reason_bucket") is fed
straight into e1._flatten_ledger() and e1.EvidenceIndex -- the identical
functions/classes the original baseline system's evidence base uses.
totals=[n_total, n_retained], same convention as the original baseline.

── Prose extraction ────────────────────────────────────────────────────
baseline_v2's HTML uses the same <h1>/<h2>/<p> document structure as the
pipeline's rendered report (unlike the original baseline, a PDF requiring
font-size/boldness heuristics to find headings) -- so this reuses
e1.extract_pipeline_prose() verbatim, via the same temporary REPORT_HTML
redirect trick evaluation/scripts/run_b1_e1_e2.py uses for b1, restored
immediately after each city. That function only collects text from <p>
sibling tags under each <h2> -- the ledger <table> and the
"Retained: N Inaccessible: N ..." summary <div> are structurally excluded
without any extra filtering (neither is a <p> tag), exactly as that
function already excludes pipeline-report captions/footers living in <div>
tags. The one <p> tag inside the "Screening Ledger" section that DOES get
picked up as prose is the closing "Screening outcome: N of M URLs retained."
line -- this matches the original baseline, where the same sentence (there,
"Screening outcome: N of the M input URLs were retained...") was likewise
captured as a claim-bearing prose sentence, not excluded as ledger
furniture (see e1_claims.csv, system=baseline, heading="Notable
Initiatives").

extract_stated_total() is NOT reused for baseline_v2: its baseline branch
matches the phrasing "N of the M input URLs were retained", which
baseline_v2's ledger states differently ("Screening outcome: N of M URLs
retained.") -- a plain regex swap, not a change to any existing system's
logic. The rest of the pipeline (E1 claim labelling, E2's four checks) is
byte-identical code reuse.

Run from the repo root with the project venv active:
    venv/bin/python3 evaluation/scripts/run_baseline_v2_e1_e2.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

_HERE = Path(__file__).resolve().parent
_EVAL_DIR = _HERE.parent
sys.path.insert(0, str(_EVAL_DIR))
import e1_evidence_consistency as e1        # noqa: E402 -- reused directly, main() never called
import e2_internal_consistency as e2mod     # noqa: E402 -- reused directly, main() never called

RESULTS_DIR = _EVAL_DIR / "results"
CITIES = e1.CITIES
BASELINE_V2_HTML = {c: e1.DATA / c / "output" / f"report_{c}_baseline_v2.html" for c in CITIES}

for c in CITIES:
    if not BASELINE_V2_HTML[c].exists():
        raise FileNotFoundError(f"baseline_v2 report missing for {c}: {BASELINE_V2_HTML[c]}")


STATED_TOTAL_RE = re.compile(r'Screening outcome:\s*([\d,]+)\s+of\s+([\d,]+)\s+URLs retained', re.IGNORECASE)


def extract_stated_total_baseline_v2(text: str) -> int | None:
    """baseline_v2's own phrasing ('Screening outcome: N of M URLs
    retained.') differs from the original baseline's ('N of the M input
    URLs were retained'), so e1.extract_stated_total()'s baseline regex
    does not match it -- this is a new regex, not an edit to that function.
    Returns N (the retained count), the same quantity
    e1.extract_stated_total() returns for the original baseline (used as
    the percentage-of-total basis in E2's CS-1/CS-2/CS-4)."""
    m = STATED_TOTAL_RE.search(text)
    return int(m.group(1).replace(",", "")) if m else None


def parse_ledger_html(city: str) -> pd.DataFrame:
    """Parse the Screening Ledger <table> straight out of
    report_<city>_baseline_v2.html -- see module docstring for why this is
    not routed through e1.parse_ledger_pdf() or classify_discard_reason()."""
    html = BASELINE_V2_HTML[city].read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table")
    tbody = table.find("tbody")
    rows = []
    for tr in tbody.find_all("tr"):
        tds = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        decision = tds[2] if len(tds) > 2 and tds[2] in ("Retained", "Discarded") else "UNKNOWN"
        reason_raw = tds[3] if len(tds) > 3 else ""
        bucket = None
        if decision == "Discarded" and reason_raw and reason_raw != "—":
            bucket = reason_raw.strip().lower().replace(" ", "_")
        rows.append({"decision": decision, "discard_reason_bucket": bucket})
    return pd.DataFrame(rows)


def _append(path: Path, new_df: pd.DataFrame) -> None:
    """Append-only write: reads just the header line to confirm column order
    matches, then appends with pandas -- never reads or rewrites existing
    data rows."""
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert list(new_df.columns) == header, (
        f"{path.name}: column order mismatch.\nexisting header: {header}\nnew_df columns: {list(new_df.columns)}"
    )
    new_df.to_csv(path, mode="a", header=False, index=False)
    print(f"Appended {len(new_df)} row(s) to {path}")


def run_e1_baseline_v2(city: str, claims_rows: list[dict], summary_rows: list[dict]):
    have_geo = e1.DISTRICTS_GEOJSON[city].exists() and e1.URLS_CLEANED[city].exists()
    known_names = e1.known_district_names(city) if have_geo else []

    def tok(sentence: str) -> list[dict]:
        return e1.tokenize_numbers_excluding_names(sentence, known_names) if known_names \
            else e1.tokenize_numbers(sentence)

    ledger_df = parse_ledger_html(city)          # report's own screening ledger, from HTML
    bflat, bgroups = e1._flatten_ledger(ledger_df)
    n_total_ledger = len(ledger_df)
    n_retained_ledger = int((ledger_df["decision"] == "Retained").sum())
    bv2_index = e1.EvidenceIndex(bflat, bgroups, totals=[float(n_total_ledger), float(n_retained_ledger)])

    orig_path = e1.REPORT_HTML[city]
    e1.REPORT_HTML[city] = BASELINE_V2_HTML[city]
    try:
        full_text, pairs = e1.extract_pipeline_prose(city)   # same function, baseline_v2 HTML
    finally:
        e1.REPORT_HTML[city] = orig_path                     # restore module state immediately

    n_direct = n_derived = n_not_auditable = 0
    for heading, sentence in pairs:
        for num in tok(sentence):
            label, justification = bv2_index.label(num["value"], num["is_percentage"])
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
    return full_text, pairs


def run_e2_baseline_v2(city: str, full_text: str, pairs: list[tuple[str, str]],
                        flags_rows: list[dict], summary_rows: list[dict]) -> None:
    total = extract_stated_total_baseline_v2(full_text)
    n_words, counts = e2mod.run_report(city, "baseline_v2", pairs, total, flags_rows)   # verbatim reuse
    n_flags = sum(counts.values())
    exposure = n_words / 1000.0
    ci_lo, ci_hi = e2mod.poisson_exact_ci(n_flags, exposure)
    summary_rows.append({
        "city": city, "system": "baseline_v2", "stated_total": total,
        "n_words": n_words, "n_flags": n_flags,
        "n_cs1": counts["CS-1"], "n_cs2": counts["CS-2"],
        "n_cs4": counts["CS-4"], "n_cs5": counts["CS-5"],
        "defects_per_1000_words": round(n_flags / exposure, 3) if exposure else None,
        "ci95_lower": round(ci_lo, 3) if exposure else None,
        "ci95_upper": round(ci_hi, 3) if exposure else None,
    })


def applicability_check(city: str, pairs: list[tuple[str, str]]) -> dict:
    """Per-city report requested alongside the run: word count of prose,
    whether '%' appears anywhere in that prose, whether 'population' or
    'capita' appears -- the same three structural preconditions the
    now-removed, superseded E2_SUMMARY.md had historically reported were
    zero for all five original (retired) baseline PDFs (which made CS-1,
    CS-4, and CS-5 structurally unable to fire there; E2_SUMMARY_v2.md does
    not cover that retired condition, so no current document restates this
    specific historical finding)."""
    prose_text = " ".join(s for _, s in pairs)
    n_words = sum(len(s.split()) for _, s in pairs)
    has_pct = "%" in prose_text
    has_pop_capita = bool(re.search(r'\bpopulation\b|\bcapita\b', prose_text, re.IGNORECASE))
    return {
        "city": city, "n_words_prose": n_words,
        "has_percent": has_pct, "n_percent_chars": prose_text.count("%"),
        "has_population_or_capita": has_pop_capita,
    }


def main() -> None:
    e1_claims_rows: list[dict] = []
    e1_summary_rows: list[dict] = []
    e2_flags_rows: list[dict] = []
    e2_summary_rows: list[dict] = []
    applicability_rows: list[dict] = []

    for city in CITIES:
        print(f"=== {city} (baseline_v2) ===")
        full_text, pairs = run_e1_baseline_v2(city, e1_claims_rows, e1_summary_rows)
        run_e2_baseline_v2(city, full_text, pairs, e2_flags_rows, e2_summary_rows)
        applicability_rows.append(applicability_check(city, pairs))

    e1_claims_cols = ["city", "system", "section_heading", "sentence", "raw_number",
                       "parsed_value", "is_percentage", "label", "justification"]
    e1_summary_cols = ["city", "system", "n_claims", "n_auditable_direct", "n_auditable_derived",
                         "n_not_machine_auditable", "pct_auditable_direct", "pct_auditable_derived",
                         "pct_not_machine_auditable", "auditability_rate"]
    e2_flags_cols = ["city", "system", "check", "section_heading", "sentence", "detail", "VERDICT", "NOTE"]
    e2_summary_cols = ["city", "system", "stated_total", "n_words", "n_flags", "n_cs1", "n_cs2",
                        "n_cs4", "n_cs5", "defects_per_1000_words", "ci95_lower", "ci95_upper"]

    _append(RESULTS_DIR / "e1_claims.csv", pd.DataFrame(e1_claims_rows, columns=e1_claims_cols))
    _append(RESULTS_DIR / "e1_summary.csv", pd.DataFrame(e1_summary_rows, columns=e1_summary_cols))
    _append(RESULTS_DIR / "e2_flags.csv", pd.DataFrame(e2_flags_rows, columns=e2_flags_cols))
    _append(RESULTS_DIR / "e2_summary.csv", pd.DataFrame(e2_summary_rows, columns=e2_summary_cols))

    print("\n=== e1_summary.csv (baseline_v2 rows) ===")
    print(pd.DataFrame(e1_summary_rows, columns=e1_summary_cols).to_string(index=False))
    print("\n=== e2_summary.csv (baseline_v2 rows) ===")
    print(pd.DataFrame(e2_summary_rows, columns=e2_summary_cols).to_string(index=False))
    print("\n=== applicability check (prose scope, same definition as e2 n_words) ===")
    print(pd.DataFrame(applicability_rows).to_string(index=False))

    pd.DataFrame(applicability_rows).to_csv(RESULTS_DIR / "baseline_v2_applicability.csv", index=False)
    print(f"\nWrote {RESULTS_DIR / 'baseline_v2_applicability.csv'}")

    print("\n" + "=" * 78)
    print("baseline_v2 rows appended to e1_claims.csv / e1_summary.csv / e2_flags.csv / e2_summary.csv.")
    print("e2_flags.csv baseline_v2 rows have VERDICT/NOTE left blank, same as an unverified")
    print("pipeline/baseline/b1 run -- do not cite baseline_v2's defects_per_1000_words as a")
    print("verified rate until hand-reviewed.")
    print("E1_SUMMARY.md and E2_SUMMARY.md were NOT regenerated (see module docstring).")
    print("=" * 78)


if __name__ == "__main__":
    main()
