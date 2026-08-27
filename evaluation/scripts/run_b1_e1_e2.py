#!/usr/bin/env python3
"""
Run E1 (auditability) and E2 (internal consistency) over the 5 B1 ablation
reports (report_<city>_b1.html -- {instruction} forced empty, same
{data_points}/model/temperature as the pipeline condition; see
evaluation/B1_report.md), labelling every new row system="b1", and add it as
a third system alongside pipeline and baseline.

Read-only except for:
  evaluation/results/e1_claims.csv     -- b1 claim rows appended
  evaluation/results/e1_summary.csv    -- b1 summary rows appended
  evaluation/results/e2_flags.csv      -- b1 flag rows appended (VERDICT/NOTE left blank)
  evaluation/results/e2_summary.csv    -- b1 summary rows appended
  evaluation/results/b1_comparison.csv -- new file, pipeline vs b1 per city

Every write below is an APPEND (`mode="a", header=False`), never a
read-modify-rewrite of the whole file -- the existing pipeline/baseline rows,
including e2_flags.csv's hand-filled VERDICT/NOTE columns from the manual
E2/E6 review, are never touched, only read (to build b1_comparison.csv's
pipeline-side columns, and to sanity-check column order before appending).

Explicitly NOT run: e1_evidence_consistency.main() / e2_internal_consistency.main().
Both would regenerate E1_SUMMARY.md / E2_SUMMARY.md from scratch and silently
delete their hand-written sections (E1's Step 5 human-validation writeup,
E2's closing pointer to E6) -- neither script's own markdown-writer
reproduces those sections, they were added by hand in prior sessions. This
script imports both modules (safe: both guard their pipeline in
`if __name__ == "__main__"`) and calls their functions directly, but never
calls either main() and never writes either SUMMARY.md.

"Same code, same evidence base as the pipeline condition":
  - E1: reuses e1.extract_facts / e1._flatten_facts / e1.EvidenceIndex /
    e1.known_district_names / e1.tokenize_numbers(_excluding_names) --
    identical calls to the ones run_city() makes for the pipeline system.
    The evidence base is extract_facts(load_pipeline_records(city)) against
    fsi_enriched.jsonl -- byte-identical to the pipeline arm, since B1 only
    ablates the Phase-5 prompt's {instruction} field, never the underlying
    data. Prose extraction reuses e1.extract_pipeline_prose(city) verbatim
    (the same function, not a reimplementation) via a temporary redirect of
    e1.REPORT_HTML[city] to the b1 HTML path, restored immediately after
    each city so module state is left exactly as found.
  - E2: reuses e2.run_report() verbatim -- the exact function pipeline and
    baseline rows are already produced by, called here with system="b1".
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_EVAL_DIR = _HERE.parent
sys.path.insert(0, str(_EVAL_DIR))
import e1_evidence_consistency as e1        # noqa: E402 -- reused directly, main() never called
import e2_internal_consistency as e2mod     # noqa: E402 -- reused directly, main() never called

RESULTS_DIR = _EVAL_DIR / "results"
CITIES = e1.CITIES
B1_HTML = {c: e1.DATA / c / "output" / f"report_{c}_b1.html" for c in CITIES}

for c in CITIES:
    if not B1_HTML[c].exists():
        raise FileNotFoundError(f"B1 report missing for {c}: {B1_HTML[c]}")


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


def run_e1_b1(city: str, claims_rows: list[dict], summary_rows: list[dict]) -> None:
    """Mirrors run_city()'s pipeline-system block in e1_evidence_consistency.py
    exactly (same functions, same evidence base), pointed at the b1 HTML."""
    have_geo = e1.DISTRICTS_GEOJSON[city].exists() and e1.URLS_CLEANED[city].exists()
    known_names = e1.known_district_names(city) if have_geo else []

    def tok(sentence: str) -> list[dict]:
        return e1.tokenize_numbers_excluding_names(sentence, known_names) if known_names \
            else e1.tokenize_numbers(sentence)

    records = e1.load_pipeline_records(city)          # fsi_enriched.jsonl -- same file as pipeline
    facts = e1.extract_facts(records)                  # same evidence base as pipeline
    flat, groups = e1._flatten_facts(facts)
    b1_index = e1.EvidenceIndex(flat, groups, totals=[float(facts["total"])])

    orig_path = e1.REPORT_HTML[city]
    e1.REPORT_HTML[city] = B1_HTML[city]
    try:
        full_text, pairs = e1.extract_pipeline_prose(city)   # same function, b1 HTML
    finally:
        e1.REPORT_HTML[city] = orig_path                     # restore module state immediately

    n_direct = n_derived = n_not_auditable = 0
    for heading, sentence in pairs:
        for num in tok(sentence):
            label, justification = b1_index.label(num["value"], num["is_percentage"])
            claims_rows.append({
                "city": city, "system": "b1", "section_heading": heading,
                "sentence": sentence, "raw_number": num["raw"], "parsed_value": num["value"],
                "is_percentage": num["is_percentage"], "label": label, "justification": justification,
            })
            n_direct += label == "AUDITABLE_DIRECT"
            n_derived += label == "AUDITABLE_DERIVED"
            n_not_auditable += label == "NOT_MACHINE_AUDITABLE"
    n_claims = n_direct + n_derived + n_not_auditable
    summary_rows.append({
        "city": city, "system": "b1", "n_claims": n_claims,
        "n_auditable_direct": n_direct, "n_auditable_derived": n_derived,
        "n_not_machine_auditable": n_not_auditable,
        "pct_auditable_direct": round(100 * n_direct / n_claims, 1) if n_claims else None,
        "pct_auditable_derived": round(100 * n_derived / n_claims, 1) if n_claims else None,
        "pct_not_machine_auditable": round(100 * n_not_auditable / n_claims, 1) if n_claims else None,
        "auditability_rate": round((n_direct + n_derived) / n_claims, 4) if n_claims else None,
    })
    return full_text, pairs


def run_e2_b1(city: str, full_text: str, pairs: list[tuple[str, str]],
              flags_rows: list[dict], summary_rows: list[dict]) -> None:
    total = e1.extract_stated_total(full_text, "pipeline")   # b1 uses the pipeline HTML template
    n_words, counts = e2mod.run_report(city, "b1", pairs, total, flags_rows)   # verbatim reuse
    n_flags = sum(counts.values())
    exposure = n_words / 1000.0
    ci_lo, ci_hi = e2mod.poisson_exact_ci(n_flags, exposure)
    summary_rows.append({
        "city": city, "system": "b1", "stated_total": total,
        "n_words": n_words, "n_flags": n_flags,
        "n_cs1": counts["CS-1"], "n_cs2": counts["CS-2"],
        "n_cs4": counts["CS-4"], "n_cs5": counts["CS-5"],
        "defects_per_1000_words": round(n_flags / exposure, 3) if exposure else None,
        "ci95_lower": round(ci_lo, 3) if exposure else None,
        "ci95_upper": round(ci_hi, 3) if exposure else None,
    })


def main() -> None:
    e1_claims_rows: list[dict] = []
    e1_summary_rows: list[dict] = []
    e2_flags_rows: list[dict] = []
    e2_summary_rows: list[dict] = []

    for city in CITIES:
        print(f"=== {city} (b1) ===")
        full_text, pairs = run_e1_b1(city, e1_claims_rows, e1_summary_rows)
        run_e2_b1(city, full_text, pairs, e2_flags_rows, e2_summary_rows)

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

    # ── b1_comparison.csv: pipeline vs b1, per city ─────────────────────
    e1_summary_full = pd.read_csv(RESULTS_DIR / "e1_summary.csv")
    e2_summary_full = pd.read_csv(RESULTS_DIR / "e2_summary.csv")

    comparison_rows = []
    for city in CITIES:
        p1 = e1_summary_full[(e1_summary_full.city == city) & (e1_summary_full.system == "pipeline")].iloc[0]
        b1s = e1_summary_full[(e1_summary_full.city == city) & (e1_summary_full.system == "b1")].iloc[0]
        p2 = e2_summary_full[(e2_summary_full.city == city) & (e2_summary_full.system == "pipeline")].iloc[0]
        b2 = e2_summary_full[(e2_summary_full.city == city) & (e2_summary_full.system == "b1")].iloc[0]
        comparison_rows.append({
            "city": city,
            "pipeline_n_claims": int(p1["n_claims"]), "b1_n_claims": int(b1s["n_claims"]),
            "pipeline_auditability_rate": p1["auditability_rate"], "b1_auditability_rate": b1s["auditability_rate"],
            "pipeline_n_flags": int(p2["n_flags"]), "b1_n_flags": int(b2["n_flags"]),
            "pipeline_n_cs1": int(p2["n_cs1"]), "b1_n_cs1": int(b2["n_cs1"]),
            "pipeline_n_cs2": int(p2["n_cs2"]), "b1_n_cs2": int(b2["n_cs2"]),
            "pipeline_n_cs4": int(p2["n_cs4"]), "b1_n_cs4": int(b2["n_cs4"]),
            "pipeline_n_cs5": int(p2["n_cs5"]), "b1_n_cs5": int(b2["n_cs5"]),
        })
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(RESULTS_DIR / "b1_comparison.csv", index=False)
    print(f"\nWrote {RESULTS_DIR / 'b1_comparison.csv'}")
    print(comparison_df.to_string(index=False))

    print("\n" + "=" * 78)
    print("b1 rows appended to e1_claims.csv / e1_summary.csv / e2_flags.csv / e2_summary.csv.")
    print("e2_flags.csv b1 rows have VERDICT/NOTE left blank, same as an unverified pipeline")
    print("or baseline run -- do not cite b1's defects_per_1000_words as a verified rate until")
    print("hand-reviewed, exactly as instructed for pipeline/baseline in the original E2 run.")
    print("E1_SUMMARY.md and E2_SUMMARY.md were NOT regenerated (see module docstring).")
    print("=" * 78)


if __name__ == "__main__":
    main()
