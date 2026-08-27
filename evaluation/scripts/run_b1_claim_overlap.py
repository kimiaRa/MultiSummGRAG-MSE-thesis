#!/usr/bin/env python3
"""
Compare pipeline vs b1 numeric claims from evaluation/results/e1_claims.csv.
Read-only apart from evaluation/results/b1_claim_overlap.csv.

A "claim" is the pair (section_heading, raw_number) -- e.g. ("Geographic
Distribution", "44%"). Two rows count as the same claim if they share both,
regardless of which sentence produced them or their AUDITABLE_DIRECT/
DERIVED/NOT_MACHINE_AUDITABLE label (not part of the key). Per city:
  pipeline_only = claims whose (heading, raw_number) appear in pipeline's
                  rows for that city but no b1 row for that city
  b1_only       = same, reversed
  both          = appear in both systems' rows for that city
  overlap_pct   = 100 * |both| / |union|  (union = pipeline_only + b1_only + both)
Pooled row sums the four per-city counts (not a re-union across cities --
the same raw_number in two different cities is not the same claim) and
recomputes overlap_pct from the summed counts.

Output: one CSV, two row shapes distinguished by `row_type` (same mixed-
row-type convention as e1_geo.csv / e2_flags.csv elsewhere in this repo):
  - row_type=summary: city (or ALL_POOLED), n_pipeline_only, n_b1_only,
    n_both, n_union, overlap_pct populated; claim/sentence columns blank.
  - row_type=diff: city, status (pipeline_only/b1_only), section_heading,
    raw_number, sentence populated; summary-count columns blank. One row
    per (claim, sentence) pair actually seen for that claim in the system
    that has it -- a claim restated in more than one sentence within the
    same section produces more than one diff row, so nothing is silently
    collapsed to a single example sentence.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
RESULTS_DIR = _HERE.parent / "results"
CLAIMS_CSV = RESULTS_DIR / "e1_claims.csv"
OUT_CSV = RESULTS_DIR / "b1_claim_overlap.csv"

CITIES = ["barcelona", "brighton", "dublin", "london", "milan"]


def main() -> None:
    df = pd.read_csv(CLAIMS_CSV)
    df = df[df.system.isin(["pipeline", "b1"])].copy()
    df["raw_number"] = df["raw_number"].astype(str)

    summary_rows = []
    diff_rows = []
    pooled = {"n_pipeline_only": 0, "n_b1_only": 0, "n_both": 0, "n_union": 0}

    for city in CITIES:
        csub = df[df.city == city]
        p = csub[csub.system == "pipeline"]
        b = csub[csub.system == "b1"]

        p_claims = set(zip(p.section_heading, p.raw_number))
        b_claims = set(zip(b.section_heading, b.raw_number))
        pipeline_only = p_claims - b_claims
        b1_only = b_claims - p_claims
        both = p_claims & b_claims
        union = p_claims | b_claims
        overlap_pct = round(100 * len(both) / len(union), 1) if union else None

        summary_rows.append({
            "row_type": "summary", "city": city,
            "n_pipeline_only": len(pipeline_only), "n_b1_only": len(b1_only),
            "n_both": len(both), "n_union": len(union), "overlap_pct": overlap_pct,
        })
        for k in pooled:
            pooled[k] += len(pipeline_only) if k == "n_pipeline_only" else \
                         len(b1_only) if k == "n_b1_only" else \
                         len(both) if k == "n_both" else len(union)

        # diff rows: every (claim, sentence) pair actually observed, not
        # collapsed to one example -- a claim restated in >1 sentence in
        # the same section produces >1 row.
        for status, claims, src in (("pipeline_only", pipeline_only, p), ("b1_only", b1_only, b)):
            for heading, raw_number in sorted(claims):
                sentences = src[(src.section_heading == heading) & (src.raw_number == raw_number)]["sentence"].unique()
                for sentence in sentences:
                    diff_rows.append({
                        "row_type": "diff", "city": city, "status": status,
                        "section_heading": heading, "raw_number": raw_number, "sentence": sentence,
                    })

    pooled_overlap = round(100 * pooled["n_both"] / pooled["n_union"], 1) if pooled["n_union"] else None
    summary_rows.append({
        "row_type": "summary", "city": "ALL_POOLED",
        "n_pipeline_only": pooled["n_pipeline_only"], "n_b1_only": pooled["n_b1_only"],
        "n_both": pooled["n_both"], "n_union": pooled["n_union"], "overlap_pct": pooled_overlap,
    })

    cols = ["row_type", "city", "status", "section_heading", "raw_number", "sentence",
            "n_pipeline_only", "n_b1_only", "n_both", "n_union", "overlap_pct"]
    summary_df = pd.DataFrame(summary_rows).reindex(columns=cols)
    diff_df = pd.DataFrame(diff_rows).reindex(columns=cols)
    out_df = pd.concat([summary_df, diff_df], ignore_index=True)
    out_df.to_csv(OUT_CSV, index=False)

    print("=== summary (per city + pooled) ===")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"\nWrote {OUT_CSV} ({len(out_df)} rows: {len(summary_df)} summary + {len(diff_df)} diff)")


if __name__ == "__main__":
    main()
