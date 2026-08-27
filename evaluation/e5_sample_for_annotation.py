#!/usr/bin/env python3
"""
E5 — Human validation sampling.

Standalone, no model calls, no network calls. Reuses only
e1_evidence_consistency.parse_ledger_pdf (the ledger extractor, Part 1's
reused loader) for Set B; Set A is drawn directly from
evaluation/results/e1_claims.csv (Part 1's own output, not the retired
evaluation). Nothing from any retired evaluation code was read, imported,
or adapted.

Draws two stratified random samples, seed=42:

  Set A (50 items): rows from e1_claims.csv, stratified across
    system x city x label (2 x 5 x 3 = up to 30 strata). Several strata are
    naturally near-empty or empty (e.g. pipeline UNSUPPORTED is 0 for 4 of
    5 cities) -- allocation uses a capacity-aware "water-filling" scheme
    (allocate_equal) so every non-empty stratum gets as close to an equal
    share as its size allows, and any shortfall from small/empty strata
    flows to the others rather than reducing the total below 50.

  Set B (40 items): 8 per city from the ledger (parse_ledger_pdf), each
    city's 8 stratified evenly across decision in {Retained, Discarded}
    (rows with decision=UNKNOWN or no extractable URL are excluded from the
    sampling frame -- ledger-extraction failures, not a decision category).

For both sets, the automated label the sample is meant to validate
(e1_claims.csv's `label` for Set A, the ledger's `decision` for Set B) is
withheld from the CSV you annotate and written only to a separate
evaluation/hidden_answers/ directory, so annotation is genuinely blind.

evaluation/scripts/score_annotation.py (an earlier, generic pooled scorer)
was removed during the final-repository cleanup -- it was never used for any
final thesis number (see evaluation/E5_SUMMARY.md's "Scoring" section) and
nothing else depended on it. evaluation/scripts/e5_final_stats.py is the
current scorer; it reads Set B against evaluation/hidden_answers/e5_set_b_hidden.csv
where present. Set A is not part of the final thesis evaluation and is no
longer scored by anything in this repository.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import e1_evidence_consistency as e1  # noqa: E402 -- reused: parse_ledger_pdf only

RESULTS_DIR = _HERE / "results"
HIDDEN_DIR = _HERE / "hidden_answers"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
HIDDEN_DIR.mkdir(parents=True, exist_ok=True)

CITIES = e1.CITIES
SEED = 42
N_SET_A = 50
N_SET_B_PER_CITY = 8
DECISION_CATEGORIES = ["Retained", "Discarded"]


# ═══════════════════════════════════════════════════════════════════════
# Capacity-aware equal allocation ("water-filling")
# ═══════════════════════════════════════════════════════════════════════

def allocate_equal(capacities: dict, total: int, rng: np.random.Generator) -> dict:
    """Allocate `total` items across strata as equally as possible, each
    stratum capped at its own capacity. Any shortfall from a stratum
    hitting its cap is redistributed across the remaining uncapped strata
    in further rounds ("water-filling"). When the exact split isn't
    divisible, the +1 remainder items go to strata chosen by `rng` (fixed
    seed => deterministic), in a fixed (sorted-key) candidate order so the
    draw itself is reproducible."""
    keys = sorted(capacities.keys())
    remaining_cap = {k: capacities[k] for k in keys if capacities[k] > 0}
    alloc = {k: 0 for k in keys}
    remaining_total = min(total, sum(remaining_cap.values()))

    while remaining_total > 0 and remaining_cap:
        active = sorted(remaining_cap.keys())
        n = len(active)
        share, rem = divmod(remaining_total, n)
        if share == 0:
            chosen_idx = rng.choice(n, size=rem, replace=False)
            for i in sorted(chosen_idx):
                k = active[i]
                alloc[k] += 1
                remaining_cap[k] -= 1
                if remaining_cap[k] == 0:
                    del remaining_cap[k]
            remaining_total -= rem
            break
        progressed = False
        for k in active:
            take = min(share, remaining_cap[k])
            if take:
                alloc[k] += take
                remaining_total -= take
                remaining_cap[k] -= take
                progressed = True
            if remaining_cap[k] == 0:
                del remaining_cap[k]
        if not progressed:
            break
    return alloc


def sample_within_strata(df: pd.DataFrame, strata_col_values: pd.Series,
                          alloc: dict, rng: np.random.Generator) -> pd.DataFrame:
    picked_idx = []
    for key in sorted(alloc.keys()):
        k = alloc[key]
        if k == 0:
            continue
        pool = df.index[strata_col_values == key].to_numpy()
        chosen = rng.choice(pool, size=k, replace=False)
        picked_idx.extend(chosen.tolist())
    return df.loc[picked_idx]


# ═══════════════════════════════════════════════════════════════════════
# Set A — e1_claims.csv, stratified system x city x label
# ═══════════════════════════════════════════════════════════════════════

def build_set_a(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    claims = pd.read_csv(RESULTS_DIR / "e1_claims.csv")
    strata_key = list(zip(claims["system"], claims["city"], claims["label"]))
    capacities: dict = {}
    for key in strata_key:
        capacities[key] = capacities.get(key, 0) + 1

    alloc = allocate_equal(capacities, N_SET_A, rng)
    n_allocated = sum(alloc.values())
    if n_allocated < N_SET_A:
        print(f"WARNING: Set A could only allocate {n_allocated}/{N_SET_A} "
              f"(only {claims.shape[0]} claims exist in total across all strata).")

    strata_series = pd.Series(strata_key, index=claims.index)
    sampled = sample_within_strata(claims, strata_series, alloc, rng)

    # final presentation order: shuffle so strata aren't grouped block-by-block
    order = rng.permutation(len(sampled))
    sampled = sampled.iloc[order].reset_index(drop=True)
    sampled.insert(0, "item_id", [f"A{i+1:03d}" for i in range(len(sampled))])

    visible = pd.DataFrame({
        "item_id": sampled["item_id"],
        "claim": sampled["raw_number"],
        "sentence": sampled["sentence"],
        "section": sampled["section_heading"],
        "city": sampled["city"],
        "system": sampled["system"],
        "evidence_excerpt": sampled["justification"],
        "MY_LABEL": "",
        "MY_NOTE": "",
    })
    hidden = pd.DataFrame({
        "item_id": sampled["item_id"],
        "city": sampled["city"],
        "system": sampled["system"],
        "sentence": sampled["sentence"],
        "label": sampled["label"],
        "raw_number": sampled["raw_number"],
        "parsed_value": sampled["parsed_value"],
        "is_percentage": sampled["is_percentage"],
        "justification": sampled["justification"],
    })
    return visible, hidden, alloc, capacities


# ═══════════════════════════════════════════════════════════════════════
# Set B — the ledger, 8 per city, stratified across decision
# ═══════════════════════════════════════════════════════════════════════

def build_set_b(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_sampled = []
    per_city_alloc = {}
    per_city_capacity = {}

    for city in CITIES:
        ledger = e1.parse_ledger_pdf(city)
        frame = ledger[ledger["decision"].isin(DECISION_CATEGORIES) & ledger["url"].notna()].copy()
        frame = frame.reset_index(drop=True)
        capacities = {d: int((frame["decision"] == d).sum()) for d in DECISION_CATEGORIES}
        alloc = allocate_equal(capacities, N_SET_B_PER_CITY, rng)
        per_city_alloc[city] = alloc
        per_city_capacity[city] = capacities
        n_allocated = sum(alloc.values())
        if n_allocated < N_SET_B_PER_CITY:
            print(f"WARNING: {city}: Set B could only allocate {n_allocated}/{N_SET_B_PER_CITY} "
                  f"(capacity {capacities}).")

        strata_series = frame["decision"]
        sampled = sample_within_strata(frame, strata_series, alloc, rng)
        sampled = sampled.copy()
        sampled["city"] = city
        all_sampled.append(sampled)

    combined = pd.concat(all_sampled, ignore_index=True)
    order = rng.permutation(len(combined))
    combined = combined.iloc[order].reset_index(drop=True)
    combined.insert(0, "item_id", [f"B{i+1:03d}" for i in range(len(combined))])

    visible = pd.DataFrame({
        "item_id": combined["item_id"],
        "url": combined["url"],
        "city": combined["city"],
        "MY_DECISION": "",
        "MY_NOTE": "",
    })
    hidden = pd.DataFrame({
        "item_id": combined["item_id"],
        "url": combined["url"],
        "city": combined["city"],
        "decision": combined["decision"],
        "discard_reason_bucket": combined["discard_reason_bucket"],
    })
    return visible, hidden, per_city_alloc, per_city_capacity


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    rng = np.random.default_rng(SEED)

    a_visible, a_hidden, a_alloc, a_capacity = build_set_a(rng)
    b_visible, b_hidden, b_alloc, b_capacity = build_set_b(rng)

    a_visible.to_csv(RESULTS_DIR / "e5_set_a.csv", index=False)
    b_visible.to_csv(RESULTS_DIR / "e5_set_b.csv", index=False)
    a_hidden.to_csv(HIDDEN_DIR / "e5_set_a_hidden.csv", index=False)
    b_hidden.to_csv(HIDDEN_DIR / "e5_set_b_hidden.csv", index=False)
    (HIDDEN_DIR / "README.md").write_text(
        "# Do not open until after annotating\n\n"
        "This directory holds the automated labels for evaluation/results/e5_set_a.csv "
        "and e5_set_b.csv, withheld so annotation can be genuinely blind. Fill in "
        "MY_LABEL/MY_NOTE (Set A) and MY_DECISION/MY_NOTE (Set B) in the results/ files "
        "first. evaluation/scripts/score_annotation.py, which used to score both sets "
        "here, was removed during the final-repository cleanup (never used for any final "
        "thesis number, and Set A is not part of the final thesis evaluation); "
        "evaluation/scripts/e5_final_stats.py now scores Set B. Opening these files "
        "before annotating defeats the point of the exercise.\n",
        encoding="utf-8",
    )

    print(f"Wrote {RESULTS_DIR / 'e5_set_a.csv'} ({len(a_visible)} rows)")
    print(f"Wrote {RESULTS_DIR / 'e5_set_b.csv'} ({len(b_visible)} rows)")
    print(f"Wrote hidden answers to {HIDDEN_DIR}/ (do not open before annotating)")

    write_markdown(a_visible, a_alloc, a_capacity, b_visible, b_alloc, b_capacity)
    print(f"Wrote {_HERE / 'E5_SUMMARY.md'}")


def write_markdown(a_visible, a_alloc, a_capacity, b_visible, b_alloc, b_capacity) -> None:
    md = []
    md.append("# E5 — Human validation sampling\n")
    md.append(
        "No model calls, no network calls. Set A drawn from `evaluation/results/e1_claims.csv` "
        "(Part 1's own output); Set B drawn via `e1_evidence_consistency.parse_ledger_pdf` "
        "(Part 1's reused ledger extractor). Fixed seed 42 throughout -- both the stratum "
        "allocation's remainder tie-breaks and the within-stratum sampling draws. Generated "
        "by `evaluation/e5_sample_for_annotation.py`.\n"
    )

    md.append("## Set A — 50 claims, stratified system x city x label\n")
    md.append(
        "Allocation uses a capacity-aware equal split (`allocate_equal`, a water-filling "
        "scheme): every non-empty (system, city, label) stratum gets as close to an equal "
        "share of 50 as its own size allows; any shortfall from strata too small to take "
        "their equal share (several are -- pipeline UNSUPPORTED is 0 rows for 4 of 5 cities) "
        "flows to the remaining strata in further rounds, rather than shrinking the sample "
        "below 50 or leaving it unevenly weighted toward the largest strata. This is a "
        "deliberate choice for *coverage* (every combination that has any data gets checked) "
        "over strict population-proportional sampling, which would have handed nearly the "
        "whole sample to the largest strata (pipeline SUPPORTED) and left several conditions "
        "unchecked.\n"
    )
    alloc_df = pd.DataFrame([
        {"system": k[0], "city": k[1], "label": k[2], "available": a_capacity[k], "sampled": v}
        for k, v in sorted(a_alloc.items())
    ])
    md.append(alloc_df.to_markdown(index=False))
    md.append("")
    md.append(
        "Visible columns (`evaluation/results/e5_set_a.csv`): `item_id`, `claim` "
        "(the raw claimed number/percentage), `sentence`, `section`, `city`, `system`, "
        "`evidence_excerpt` (the automated checker's justification string -- what evidence "
        "it matched against, *not* the SUPPORTED/DERIVED/UNSUPPORTED verdict itself), "
        "`MY_LABEL`, `MY_NOTE` (both empty, yours to fill). The automated `label` is withheld "
        "in `evaluation/hidden_answers/e5_set_a_hidden.csv`, joined on `item_id`.\n"
    )

    md.append("## Set B — 40 ledger rows, 8 per city, stratified across decision\n")
    md.append(
        "Per city: `allocate_equal({Retained: n, Discarded: n}, 8, rng)` -- as close to 4/4 "
        "as that city's ledger allows (all 5 cities have far more than 4 of each, so every "
        "city gets an exact 4/4 split). Rows with `decision=UNKNOWN` or no extractable "
        "Candidate-column URL are excluded from the sampling frame before allocation -- "
        "ledger-extraction failures (2 rows in Barcelona, 1 in London, 0 elsewhere), not a "
        "genuine decision category.\n"
    )
    b_alloc_rows = []
    for city in CITIES:
        for dec in DECISION_CATEGORIES:
            b_alloc_rows.append({
                "city": city, "decision": dec,
                "available": b_capacity[city].get(dec, 0),
                "sampled": b_alloc[city].get(dec, 0),
            })
    md.append(pd.DataFrame(b_alloc_rows).to_markdown(index=False))
    md.append("")
    md.append(
        "Visible columns (`evaluation/results/e5_set_b.csv`): `item_id`, `url`, `city`, "
        "`MY_DECISION`, `MY_NOTE` (empty, yours to fill with Retained/Discarded). The "
        "automated `decision` (and `discard_reason_bucket` for context) is withheld in "
        "`evaluation/hidden_answers/e5_set_b_hidden.csv`, joined on `item_id`.\n"
    )

    md.append("## Scoring\n")
    md.append(
        "evaluation/scripts/score_annotation.py, an earlier generic pooled scorer for "
        "both sets, was removed during the final-repository cleanup -- it was never used "
        "for any final thesis number, since Set A's weighted sample would misrepresent a "
        "pooled figure and Set A is not part of the final thesis evaluation. "
        "`venv/bin/python3 evaluation/scripts/e5_final_stats.py` is the current scorer; "
        "it computes Set B's raw agreement and Cohen's kappa and writes "
        "`evaluation/results/e5_final_stats.csv` / `.md`.\n"
    )

    (RESULTS_DIR.parent / "E5_SUMMARY.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
