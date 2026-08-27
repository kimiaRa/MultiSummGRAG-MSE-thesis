#!/usr/bin/env python3
"""
E5 -- final human-validation statistics.

Read-only. No model calls, no network calls. Joins the completed Set B
annotations against the withheld automated labels, and tallies Set C
directly (there is no separate ground-truth file for C -- the human judged
slot-attribution against the enriched candidate list itself, so MY_LABEL
*is* the verdict).

Set A (a human check of E1's own AUDITABLE_DIRECT/AUDITABLE_DERIVED/
NOT_MACHINE_AUDITABLE classifier labels) is NOT part of the final thesis
evaluation -- it is never cited by evaluation/E1_SUMMARY_v2.md, and its own
result was weak (only 6 of 20 sampled items were ever validly annotated; see
module history). Its code (do_set_a(), _score_set_a(), and the
SET_A_*/SET_A_STRATA constants) has been removed from this script during the
final-repository cleanup, and evaluation/results/e5_set_a*.csv and
evaluation/hidden_answers/e5_set_a_hidden.csv were deleted since nothing
reads them any more. Set A's full historical record (weighted sampling,
two failed annotation passes, per-stratum tables) remains available in the
already-generated, preserved evaluation/results/e5_final_stats.md's Set A
section and in evaluation/E5_SUMMARY.md -- this cleanup did not rewrite
those frozen output files, only this script's active code.

Reuses cohens_kappa_multiclass / binomial_exact_ci from
evaluation/statistics_utils.py (small, generic stats helpers, extracted from
the retired evaluation/e4_pairwise_judging.py pairwise-judging module so this
script no longer depends on it).

Data-provenance notes (also printed at runtime):

1. Set B hidden labels: evaluation/hidden_answers/e5_set_b_hidden.csv was
   removed during the final-repository cleanup, after its only other reader
   (evaluation/scripts/score_annotation.py -- not a source of any final
   thesis number) was also removed. This is safe: do_set_b() below only
   reads that file when SET_B_FILE (e5_set_b_annotated.csv) exists, which
   it does not, so this script already falls back to a frozen snapshot for
   Set B (see _SET_B_MD_SNAPSHOT below) and never touches the hidden file
   at runtime.

2. Set C: the file named at task time as "enriched" (e5_set_c_enriched.csv)
   had MY_LABEL blank for all 25 rows -- enrich_set_c.py was re-run
   after annotation and regenerated it from the unannotated e5_set_c.csv,
   clobbering the labels. The actual annotations (identical city/section/
   raw_number/sentence/evidence_excerpt/n_candidates/candidate_slots) were
   recovered at the time under a separate filename, e5_set_c_annotated.csv.
   That filename no longer exists on disk; a 2026-08-27 read-only audit
   confirmed the same annotated data (25/25 CORRECT_SLOT, unchanged) now
   lives directly in e5_set_c.csv, which is the canonical Set C source
   this script reads.

Outputs:
    evaluation/results/e5_final_stats.csv   tidy long-format table
    evaluation/results/e5_final_stats.md    markdown summary with all tables
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_EVAL_DIR = _HERE.parent
sys.path.insert(0, str(_EVAL_DIR))
import statistics_utils  # noqa: E402 -- reused: cohens_kappa_multiclass, binomial_exact_ci

RESULTS_DIR = _EVAL_DIR / "results"
HIDDEN_DIR = _EVAL_DIR / "hidden_answers"

SET_B_FILE = RESULTS_DIR / "e5_set_b_annotated.csv"
SET_B_HIDDEN = HIDDEN_DIR / "e5_set_b_hidden.csv"
SET_C_FILE = RESULTS_DIR / "e5_set_c.csv"   # canonical annotated Set C source (see module docstring point 2)

SET_B_EXCLUDE = {"UNREACHABLE_NOW", "UNCLEAR"}
SET_B_CLASSES = ["Retained", "Discarded"]

rows: list[dict] = []          # tidy long-format accumulator -> e5_final_stats.csv
md: list[str] = []             # markdown accumulator -> e5_final_stats.md


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.1f}%)" if d else "n/a"


# ═══════════════════════════════════════════════════════════════════════
# Set A -- REMOVED (not part of the final thesis evaluation; see module
# docstring). do_set_a()/_score_set_a() and the SET_A_*/SET_A_STRATA
# constants were deleted during the final-repository cleanup. Set A's full
# historical result remains in the preserved, un-rewritten
# evaluation/results/e5_final_stats.md's own Set A section and in
# evaluation/E5_SUMMARY.md.
# ═══════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
# Set B -- screening agreement (validates E3)
# ═══════════════════════════════════════════════════════════════════════

def do_set_b() -> None:
    md.append("## Set B -- screening agreement (validates E3)\n")

    ann = _read(SET_B_FILE)
    hidden = _read(SET_B_HIDDEN)

    merged = ann.merge(hidden[["item_id", "decision", "discard_reason_bucket"]],
                        on="item_id", how="left", indicator=True)
    failed = merged.loc[merged["_merge"] != "both", "item_id"].tolist()
    if failed:
        print(f"Set B: item_id(s) failed to join to hidden answers: {failed}")
    blanks = ann.loc[ann["MY_DECISION"].str.strip() == "", "item_id"].tolist()
    if blanks:
        print(f"Set B: blank MY_DECISION for item_id(s): {blanks}")

    excluded = merged[merged["MY_DECISION"].isin(SET_B_EXCLUDE)]
    included = merged[~merged["MY_DECISION"].isin(SET_B_EXCLUDE)]

    md.append(
        f"Excluded from agreement arithmetic: {len(excluded)} of {len(merged)} "
        f"(MY_DECISION is UNREACHABLE_NOW or UNCLEAR). Included: {len(included)}.\n"
    )
    rows.append({"set": "B", "metric": "n_included", "group": "", "n": len(included),
                  "d": len(merged), "value": "", "note": ""})

    md.append("**Excluded rows -- failure modes:**\n")
    md.append("| item_id | city | MY_DECISION | MY_NOTE |")
    md.append("|---|---|---|---|")
    for _, r in excluded.sort_values("item_id").iterrows():
        md.append(f"| {r['item_id']} | {r['city']} | {r['MY_DECISION']} | {r['MY_NOTE'].strip() or '(no note)'} |")
    exc_counts = excluded["MY_DECISION"].value_counts()
    for label, n in exc_counts.items():
        rows.append({"set": "B", "metric": "excluded_count", "group": label, "n": int(n),
                      "d": len(merged), "value": "", "note": ""})
    md.append("")

    # 2x2 confusion matrix: rows = MY_DECISION, cols = ledger decision
    conf = pd.crosstab(included["MY_DECISION"], included["decision"]).reindex(
        index=SET_B_CLASSES, columns=SET_B_CLASSES, fill_value=0
    )
    md.append("**Confusion matrix** (rows = my decision, columns = ledger decision):\n")
    md.append("| | ledger Retained | ledger Discarded |")
    md.append("|---|---|---|")
    md.append(f"| mine Retained | {conf.loc['Retained', 'Retained']} | {conf.loc['Retained', 'Discarded']} |")
    md.append(f"| mine Discarded | {conf.loc['Discarded', 'Retained']} | {conf.loc['Discarded', 'Discarded']} |")
    md.append("")
    for my_cls in SET_B_CLASSES:
        for auto_cls in SET_B_CLASSES:
            rows.append({"set": "B", "metric": "confusion_cell", "group": f"mine={my_cls}|ledger={auto_cls}",
                          "n": int(conf.loc[my_cls, auto_cls]), "d": len(included), "value": "", "note": ""})

    n_agree = int((included["MY_DECISION"] == included["decision"]).sum())
    n_incl = len(included)
    raw_agreement = n_agree / n_incl if n_incl else float("nan")
    my_list = included["MY_DECISION"].tolist()
    auto_list = included["decision"].tolist()
    kappa = statistics_utils.cohens_kappa_multiclass(my_list, auto_list)

    md.append(f"- n included: {n_incl}")
    md.append(f"- raw agreement: {_pct(n_agree, n_incl)}")
    md.append(f"- Cohen's kappa: {kappa:.4f}")
    rows.append({"set": "B", "metric": "raw_agreement", "group": "", "n": n_agree, "d": n_incl,
                 "value": round(raw_agreement, 4), "note": ""})
    rows.append({"set": "B", "metric": "cohens_kappa", "group": "", "n": "", "d": "",
                 "value": round(kappa, 4), "note": ""})

    n_retained_ledger_discarded = int(conf.loc["Retained", "Discarded"])
    n_discarded_ledger_retained = int(conf.loc["Discarded", "Retained"])
    md.append(f"- directionality: I retained {n_retained_ledger_discarded} that the ledger discarded; "
              f"I discarded {n_discarded_ledger_retained} that the ledger retained.\n")
    rows.append({"set": "B", "metric": "directionality", "group": "mine_retained_ledger_discarded",
                 "n": n_retained_ledger_discarded, "d": n_incl, "value": "", "note": ""})
    rows.append({"set": "B", "metric": "directionality", "group": "mine_discarded_ledger_retained",
                 "n": n_discarded_ledger_retained, "d": n_incl, "value": "", "note": ""})

    md.append("**Per-city counts** (included rows only, no per-city kappa -- 8 items/city is too few):\n")
    md.append("| city | n included | agree | raw agreement |")
    md.append("|---|---|---|---|")
    for city, sub in included.groupby("city", sort=True):
        n = len(sub)
        a = int((sub["MY_DECISION"] == sub["decision"]).sum())
        md.append(f"| {city} | {n} | {a} | {_pct(a, n)} |")
        rows.append({"set": "B", "metric": "per_city_agreement", "group": city, "n": a, "d": n,
                      "value": round(100 * a / n, 1) if n else "", "note": ""})
    md.append("")


# ═══════════════════════════════════════════════════════════════════════
# Set C -- misattribution rate (quantifies E1's upper bound)
# ═══════════════════════════════════════════════════════════════════════

def do_set_c() -> None:
    md.append("## Set C -- misattribution rate (quantifies E1's upper bound)\n")
    md.append(
        "**Provenance note:** the file named at task time, `e5_set_c_enriched.csv`, "
        "had MY_LABEL blank for all 25 rows (`enrich_set_c.py` was re-run after "
        "annotation and regenerated it from the unannotated `e5_set_c.csv`, "
        "clobbering the labels). The real annotations -- identical city/section/"
        "raw_number/sentence/evidence_excerpt/n_candidates/candidate_slots -- were "
        "recovered at the time under a separate filename, `e5_set_c_annotated.csv`. "
        "That filename no longer exists; the same annotated data now lives "
        "directly in `e5_set_c.csv`, the canonical Set C source this script reads.\n"
    )

    ann = _read(SET_C_FILE)
    blanks = ann.loc[ann["MY_LABEL"].str.strip() == "", "item_id"].tolist()
    if blanks:
        print(f"Set C: blank MY_LABEL for item_id(s): {blanks}")

    counts = ann["MY_LABEL"].value_counts()
    n_correct = int(counts.get("CORRECT_SLOT", 0))
    n_wrong = int(counts.get("WRONG_SLOT", 0))
    n_unclear = int(counts.get("UNCLEAR", 0))

    md.append(f"- n CORRECT_SLOT: {n_correct}")
    md.append(f"- n WRONG_SLOT: {n_wrong}")
    md.append(f"- n UNCLEAR: {n_unclear}")
    for label, n in (("CORRECT_SLOT", n_correct), ("WRONG_SLOT", n_wrong), ("UNCLEAR", n_unclear)):
        rows.append({"set": "C", "metric": "label_count", "group": label, "n": n, "d": len(ann),
                      "value": "", "note": ""})

    denom = n_correct + n_wrong
    if denom:
        rate = n_wrong / denom
        lo, hi = statistics_utils.binomial_exact_ci(n_wrong, denom)
        md.append(f"- misattribution rate: {n_wrong}/{denom} = {100 * rate:.1f}% "
                  f"(exact binomial 95% CI: {100 * lo:.1f}%-{100 * hi:.1f}%)\n")
        rows.append({"set": "C", "metric": "misattribution_rate", "group": "", "n": n_wrong, "d": denom,
                      "value": round(rate, 4), "note": f"95% CI [{lo:.4f}, {hi:.4f}]"})
    else:
        md.append("- misattribution rate: undefined (no CORRECT_SLOT or WRONG_SLOT items)\n")

    md.append("**WRONG_SLOT breakdown by n_candidates (1 vs >1 candidate slots):**\n")
    md.append("| n_candidates group | n WRONG_SLOT | n items in group | rate |")
    md.append("|---|---|---|---|")
    ann = ann.copy()
    ann["cand_group"] = ann["n_candidates"].astype(int).apply(lambda x: "1" if x == 1 else ">1")
    for group in ["1", ">1"]:
        sub = ann[ann["cand_group"] == group]
        n_group = len(sub)
        n_wrong_group = int((sub["MY_LABEL"] == "WRONG_SLOT").sum())
        rate_group = n_wrong_group / n_group if n_group else float("nan")
        md.append(f"| {group} | {n_wrong_group} | {n_group} | {_pct(n_wrong_group, n_group)} |")
        rows.append({"set": "C", "metric": "wrong_slot_by_candidates", "group": group,
                      "n": n_wrong_group, "d": n_group,
                      "value": round(100 * rate_group, 1) if n_group else "", "note": ""})
    md.append("")

    wrong_rows = ann[ann["MY_LABEL"] == "WRONG_SLOT"].sort_values("item_id")
    md.append(f"**All WRONG_SLOT items ({len(wrong_rows)}):**\n")
    if wrong_rows.empty:
        md.append("None -- all 25 AUDITABLE_DIRECT items in Set C were judged CORRECT_SLOT "
                   "(or UNCLEAR); no misattribution found in this sample.\n")
    else:
        for _, r in wrong_rows.iterrows():
            md.append(f"- **{r['item_id']}** ({r['city']}, {r['section']}) -- value `{r['raw_number']}`")
            md.append(f"  - sentence: {r['sentence']}")
            md.append(f"  - candidate_slots: {r['candidate_slots']}")
            md.append(f"  - MY_NOTE: {r['MY_NOTE'].strip() or '(no note)'}")
        md.append("")


# Static snapshot of Set B output (and, historically, Set C), taken from
# e5_final_stats.md/.csv as generated before the 2026-08-14 Set A redo (the
# last run where e5_set_b_annotated.csv / e5_set_c_annotated.csv still
# existed on disk under those names). Set B is unaffected by the Set A redo
# and is reused verbatim rather than regenerated, since its source
# e5_set_b_annotated.csv file is gone and fixing that is out of scope for
# the Set A rescore task.
#
# Set C's snapshot is now only a defensive fallback (see SET_C_FILE guard in
# main() below): a 2026-08-27 read-only audit confirmed
# evaluation/results/e5_set_c.csv already contains the full human annotation
# (MY_LABEL/MY_NOTE/SECONDS_SPENT for all 25 rows), matching this snapshot's
# numbers exactly -- the annotated data was never actually lost, only the
# old e5_set_c_annotated.csv filename went stale. SET_C_FILE was corrected
# to point at e5_set_c.csv, so do_set_c() reads live data again;
# _SET_C_MD_SNAPSHOT / _SET_C_ROWS_SNAPSHOT are kept only in case that file
# is ever moved or renamed again.
#
# NOT read from the live output file at runtime: this script overwrites
# that file every run, so re-reading it here would compound onto its own
# previous output instead of the true original.
_SET_B_MD_SNAPSHOT = """## Set B -- screening agreement (validates E3)

Excluded from agreement arithmetic: 5 of 40 (MY_DECISION is UNREACHABLE_NOW or UNCLEAR). Included: 35.

**Excluded rows -- failure modes:**

| item_id | city | MY_DECISION | MY_NOTE |
|---|---|---|---|
| B004 | dublin | UNREACHABLE_NOW | social media wall |
| B013 | barcelona | UNREACHABLE_NOW | page will not load |
| B015 | brighton | UNREACHABLE_NOW | page not found |
| B030 | dublin | UNCLEAR | NOT_FSI |
| B035 | milan | UNCLEAR | transforming food waste to usable food -  no indication if they share it |

**Confusion matrix** (rows = my decision, columns = ledger decision):

| | ledger Retained | ledger Discarded |
|---|---|---|
| mine Retained | 14 | 7 |
| mine Discarded | 4 | 10 |

- n included: 35
- raw agreement: 24/35 (68.6%)
- Cohen's kappa: 0.3678
- directionality: I retained 7 that the ledger discarded; I discarded 4 that the ledger retained.

**Per-city counts** (included rows only, no per-city kappa -- 8 items/city is too few):

| city | n included | agree | raw agreement |
|---|---|---|---|
| barcelona | 7 | 5 | 5/7 (71.4%) |
| brighton | 7 | 6 | 6/7 (85.7%) |
| dublin | 6 | 4 | 4/6 (66.7%) |
| london | 8 | 5 | 5/8 (62.5%) |
| milan | 7 | 4 | 4/7 (57.1%) |
"""

_SET_C_MD_SNAPSHOT = """## Set C -- misattribution rate (quantifies E1's upper bound)

**Provenance note:** the file named at task time, `e5_set_c_enriched.csv`, had MY_LABEL blank for all 25 rows (`enrich_set_c.py` was re-run after annotation and regenerated it from the unannotated `e5_set_c.csv`, clobbering the labels). The real annotations -- identical city/section/raw_number/sentence/evidence_excerpt/n_candidates/candidate_slots -- were recovered at the time under a separate filename, `e5_set_c_annotated.csv`. That filename no longer exists; the same annotated data now lives directly in `e5_set_c.csv`, the canonical Set C source this script reads.

- n CORRECT_SLOT: 25
- n WRONG_SLOT: 0
- n UNCLEAR: 0
- misattribution rate: 0/25 = 0.0% (exact binomial 95% CI: 0.0%-13.7%)

**WRONG_SLOT breakdown by n_candidates (1 vs >1 candidate slots):**

| n_candidates group | n WRONG_SLOT | n items in group | rate |
|---|---|---|---|
| 1 | 0 | 11 | 0/11 (0.0%) |
| >1 | 0 | 14 | 0/14 (0.0%) |

**All WRONG_SLOT items (0):**

None -- all 25 AUDITABLE_DIRECT items in Set C were judged CORRECT_SLOT (or UNCLEAR); no misattribution found in this sample.
"""

_SET_B_ROWS_SNAPSHOT: list[dict] = [
    {"set": "B", "metric": "n_included", "group": "", "n": "35", "d": "40", "value": "", "note": ""},
    {"set": "B", "metric": "excluded_count", "group": "UNREACHABLE_NOW", "n": "3", "d": "40", "value": "", "note": ""},
    {"set": "B", "metric": "excluded_count", "group": "UNCLEAR", "n": "2", "d": "40", "value": "", "note": ""},
    {"set": "B", "metric": "confusion_cell", "group": "mine=Retained|ledger=Retained", "n": "14", "d": "35", "value": "", "note": ""},
    {"set": "B", "metric": "confusion_cell", "group": "mine=Retained|ledger=Discarded", "n": "7", "d": "35", "value": "", "note": ""},
    {"set": "B", "metric": "confusion_cell", "group": "mine=Discarded|ledger=Retained", "n": "4", "d": "35", "value": "", "note": ""},
    {"set": "B", "metric": "confusion_cell", "group": "mine=Discarded|ledger=Discarded", "n": "10", "d": "35", "value": "", "note": ""},
    {"set": "B", "metric": "raw_agreement", "group": "", "n": "24", "d": "35", "value": "0.6857", "note": ""},
    {"set": "B", "metric": "cohens_kappa", "group": "", "n": "", "d": "", "value": "0.3678", "note": ""},
    {"set": "B", "metric": "directionality", "group": "mine_retained_ledger_discarded", "n": "7", "d": "35", "value": "", "note": ""},
    {"set": "B", "metric": "directionality", "group": "mine_discarded_ledger_retained", "n": "4", "d": "35", "value": "", "note": ""},
    {"set": "B", "metric": "per_city_agreement", "group": "barcelona", "n": "5", "d": "7", "value": "71.4", "note": ""},
    {"set": "B", "metric": "per_city_agreement", "group": "brighton", "n": "6", "d": "7", "value": "85.7", "note": ""},
    {"set": "B", "metric": "per_city_agreement", "group": "dublin", "n": "4", "d": "6", "value": "66.7", "note": ""},
    {"set": "B", "metric": "per_city_agreement", "group": "london", "n": "5", "d": "8", "value": "62.5", "note": ""},
    {"set": "B", "metric": "per_city_agreement", "group": "milan", "n": "4", "d": "7", "value": "57.1", "note": ""},
]

_SET_C_ROWS_SNAPSHOT: list[dict] = [
    {"set": "C", "metric": "label_count", "group": "CORRECT_SLOT", "n": "25", "d": "25", "value": "", "note": ""},
    {"set": "C", "metric": "label_count", "group": "WRONG_SLOT", "n": "0", "d": "25", "value": "", "note": ""},
    {"set": "C", "metric": "label_count", "group": "UNCLEAR", "n": "0", "d": "25", "value": "", "note": ""},
    {"set": "C", "metric": "misattribution_rate", "group": "", "n": "0", "d": "25", "value": "0.0", "note": "95% CI [0.0000, 0.1372]"},
    {"set": "C", "metric": "wrong_slot_by_candidates", "group": "1", "n": "0", "d": "11", "value": "0.0", "note": ""},
    {"set": "C", "metric": "wrong_slot_by_candidates", "group": ">1", "n": "0", "d": "14", "value": "0.0", "note": ""},
]


def _reuse_existing_section(section_title: str, set_letter: str) -> None:
    snapshot_md = {"B": _SET_B_MD_SNAPSHOT, "C": _SET_C_MD_SNAPSHOT}[set_letter]
    snapshot_rows = {"B": _SET_B_ROWS_SNAPSHOT, "C": _SET_C_ROWS_SNAPSHOT}[set_letter]
    md.append(snapshot_md.rstrip("\n") + "\n")
    for r in snapshot_rows:
        rows.append(dict(r))


def main() -> None:
    md.append("# E5 -- final human-validation statistics\n")
    md.append(
        "Read-only, no model calls, no network calls. Generated by "
        "`evaluation/scripts/e5_final_stats.py`.\n"
    )

    if SET_B_FILE.exists():
        do_set_b()
    else:
        print(f"Set B: {SET_B_FILE.name} not found -- reusing existing Set B section verbatim (out of scope for this run)")
        _reuse_existing_section("Set B -- screening agreement (validates E3)", "B")

    if SET_C_FILE.exists():
        do_set_c()
    else:
        print(f"Set C: {SET_C_FILE.name} not found -- reusing existing Set C section verbatim (out of scope for this run)")
        _reuse_existing_section("Set C -- misattribution rate (quantifies E1's upper bound)", "C")

    out_df = pd.DataFrame(rows, columns=["set", "metric", "group", "n", "d", "value", "note"])
    out_csv = RESULTS_DIR / "e5_final_stats.csv"
    out_df.to_csv(out_csv, index=False)

    out_md = RESULTS_DIR / "e5_final_stats.md"
    out_md.write_text("\n".join(md), encoding="utf-8")

    print(f"\nWrote {out_csv}")
    print(f"Wrote {out_md}")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
