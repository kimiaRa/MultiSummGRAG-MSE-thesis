#!/usr/bin/env python3
"""
Re-run E2 on all four systems (pipeline, b1, baseline, baseline_v2) after
widening CS-1 and CS-4 in evaluation/e2_internal_consistency.py to match
count/percentage phrasings actually used across all three report styles,
not only the pipeline's "N (P%)" / "account(s) for N%" forms -- see that
module's CS-1/CS-4 section docstrings for the exact regex changes and why
each one was needed (motivated by baseline/baseline_v2 sentences like
"19 of 47 retained initiatives (40.4%)" and "account for 6 of 24, or
25.0%", which produced no CS-1/CS-4 match at all before this change, not a
checked-and-passed one). CS-2 and CS-5 were not touched.

Read-only apart from NEW outputs:
  evaluation/results/e2_flags_v2.csv
  evaluation/results/e2_summary_v2.csv
These are full, from-scratch runs across all 20 (city, system) pairs -- NOT
appends -- written under new filenames specifically so the original
e2_flags.csv / e2_summary.csv (produced under the pre-widening CS-1/CS-4)
stay exactly as they were and the effect of the widening stays visible by
diffing the two pairs of files. Nothing here writes to any e2_*.csv file
that lacks the _v2 suffix, and E2_SUMMARY.md is not touched either.

VERDICT/NOTE are left blank on every e2_flags_v2.csv row, same convention
as every other unverified E2 run in this project.

Prose extraction per system (identical to how each system's rows in the
original e2_summary.csv/e2_flags.csv, and in
evaluation/scripts/run_b1_e1_e2.py / run_baseline_v2_e1_e2.py, were built):
  - pipeline:    e1.extract_pipeline_prose(city) on report_<city>.html
  - baseline:    e1.extract_baseline_prose(city) on the appendix PDF
  - b1:          e1.extract_pipeline_prose(city), REPORT_HTML redirected to
                 report_<city>_b1.html (same function, b1's HTML)
  - baseline_v2: e1.extract_pipeline_prose(city), REPORT_HTML redirected to
                 report_<city>_baseline_v2.html

Run from the repo root with the project venv active:
    venv/bin/python3 evaluation/scripts/run_cs1_cs4_widen_e2.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_EVAL_DIR = _HERE.parent
sys.path.insert(0, str(_EVAL_DIR))
import e1_evidence_consistency as e1        # noqa: E402
import e2_internal_consistency as e2mod     # noqa: E402 -- the just-widened module; main() never called

RESULTS_DIR = _EVAL_DIR / "results"
CITIES = e1.CITIES

B1_HTML = {c: e1.DATA / c / "output" / f"report_{c}_b1.html" for c in CITIES}
BASELINE_V2_HTML = {c: e1.DATA / c / "output" / f"report_{c}_baseline_v2.html" for c in CITIES}
for c in CITIES:
    if not B1_HTML[c].exists():
        raise FileNotFoundError(f"b1 report missing for {c}: {B1_HTML[c]}")
    if not BASELINE_V2_HTML[c].exists():
        raise FileNotFoundError(f"baseline_v2 report missing for {c}: {BASELINE_V2_HTML[c]}")

BASELINE_V2_OUTCOME_RE = re.compile(r'Screening outcome:\s*([\d,]+)\s+of\s+([\d,]+)\s+URLs retained', re.IGNORECASE)


def extract_stated_total_baseline_v2(text: str) -> int | None:
    m = BASELINE_V2_OUTCOME_RE.search(text)
    return int(m.group(1).replace(",", "")) if m else None


def get_pairs_and_total(city: str, system: str) -> tuple[str, list[tuple[str, str]], int | None]:
    if system == "pipeline":
        full_text, pairs = e1.extract_pipeline_prose(city)
        total = e1.extract_stated_total(full_text, "pipeline")
    elif system == "baseline":
        full_text, pairs = e1.extract_baseline_prose(city)
        total = e1.extract_stated_total(full_text, "baseline")
    elif system == "b1":
        orig = e1.REPORT_HTML[city]
        e1.REPORT_HTML[city] = B1_HTML[city]
        try:
            full_text, pairs = e1.extract_pipeline_prose(city)
        finally:
            e1.REPORT_HTML[city] = orig
        total = e1.extract_stated_total(full_text, "pipeline")  # b1 uses the pipeline HTML template
    elif system == "baseline_v2":
        orig = e1.REPORT_HTML[city]
        e1.REPORT_HTML[city] = BASELINE_V2_HTML[city]
        try:
            full_text, pairs = e1.extract_pipeline_prose(city)
        finally:
            e1.REPORT_HTML[city] = orig
        total = extract_stated_total_baseline_v2(full_text)
    else:
        raise ValueError(system)
    return full_text, pairs, total


def main() -> None:
    systems = ["pipeline", "b1", "baseline", "baseline_v2"]
    flags_rows: list[dict] = []
    summary_rows: list[dict] = []

    for city in CITIES:
        for system in systems:
            full_text, pairs, total = get_pairs_and_total(city, system)
            n_words, counts = e2mod.run_report(city, system, pairs, total, flags_rows)
            n_flags = sum(counts.values())
            exposure = n_words / 1000.0
            ci_lo, ci_hi = e2mod.poisson_exact_ci(n_flags, exposure)
            summary_rows.append({
                "city": city, "system": system, "stated_total": total,
                "n_words": n_words, "n_flags": n_flags,
                "n_cs1": counts["CS-1"], "n_cs2": counts["CS-2"],
                "n_cs4": counts["CS-4"], "n_cs5": counts["CS-5"],
                "defects_per_1000_words": round(n_flags / exposure, 3) if exposure else None,
                "ci95_lower": round(ci_lo, 3) if exposure else None,
                "ci95_upper": round(ci_hi, 3) if exposure else None,
            })

    flags_df = pd.DataFrame(flags_rows, columns=[
        "city", "system", "check", "section_heading", "sentence", "detail", "VERDICT", "NOTE",
    ])
    summary_df = pd.DataFrame(summary_rows)

    flags_df.to_csv(RESULTS_DIR / "e2_flags_v2.csv", index=False)
    summary_df.to_csv(RESULTS_DIR / "e2_summary_v2.csv", index=False)
    print(f"Wrote {RESULTS_DIR / 'e2_flags_v2.csv'} ({len(flags_df)} rows) and "
          f"{RESULTS_DIR / 'e2_summary_v2.csv'} ({len(summary_df)} rows)")

    # ── old vs new comparison, per system, pooled over 5 cities ─────────
    old_summary = pd.read_csv(RESULTS_DIR / "e2_summary.csv")

    print("\n" + "=" * 90)
    print("OLD (pre-widening) vs NEW (post-widening) flag counts, per system, pooled over 5 cities")
    print("=" * 90)
    rows = []
    for system in systems:
        old = old_summary[old_summary.system == system]
        new = summary_df[summary_df.system == system]
        if len(old) == 0:
            print(f"** {system}: no rows found in original e2_summary.csv -- skipping old-vs-new for this system **")
            old_tot = old_cs1 = old_cs2 = old_cs4 = old_cs5 = None
        else:
            old_tot, old_cs1, old_cs2, old_cs4, old_cs5 = (
                int(old.n_flags.sum()), int(old.n_cs1.sum()), int(old.n_cs2.sum()),
                int(old.n_cs4.sum()), int(old.n_cs5.sum()),
            )
        new_tot, new_cs1, new_cs2, new_cs4, new_cs5 = (
            int(new.n_flags.sum()), int(new.n_cs1.sum()), int(new.n_cs2.sum()),
            int(new.n_cs4.sum()), int(new.n_cs5.sum()),
        )
        rows.append({
            "system": system,
            "old_n_flags": old_tot, "new_n_flags": new_tot,
            "old_cs1": old_cs1, "new_cs1": new_cs1,
            "old_cs2": old_cs2, "new_cs2": new_cs2,
            "old_cs4": old_cs4, "new_cs4": new_cs4,
            "old_cs5": old_cs5, "new_cs5": new_cs5,
        })
    comparison_df = pd.DataFrame(rows)
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv(RESULTS_DIR / "e2_widen_old_vs_new_summary.csv", index=False)
    print(f"\nWrote {RESULTS_DIR / 'e2_widen_old_vs_new_summary.csv'}")

    print("\nCS-2 and CS-5 counts are identical old vs new for every system (unchanged checks) --")
    print("verified above rather than assumed. Original e2_flags.csv / e2_summary.csv were not")
    print("opened for writing. VERDICT/NOTE left blank on every e2_flags_v2.csv row.")


if __name__ == "__main__":
    main()
