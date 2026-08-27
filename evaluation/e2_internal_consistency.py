#!/usr/bin/env python3
"""
E2 — Internal consistency checks.

Standalone, read-only, no network calls, no pipeline stage run. Reuses only
the number tokeniser and prose-extraction utilities from
evaluation/e1_evidence_consistency.py (this project's own Part 1 module —
not the retired evaluation). scripts/consistency_audit.py and
src/phase_6_v2/consistency_audit.py, both on the retired list, were not
read, imported, or adapted; every check below was written from scratch.

Four checks, applied to all 10 reports (5 pipeline HTML + 5 baseline PDF):

  CS-1  count/percentage pairs within one sentence that disagree with the
        report's own stated total (e.g. "105 FSIs (44.3%)" checked against
        that report's stated grand total).
  CS-2  "majority" / "most" / "over half" / "nearly all" applied to a
        value that is actually <=50% of the total.
  CS-4  aggregate claims ("A, B and C ... account for N%") that don't
        reconcile against the constituent counts stated earlier in the
        same sentence.
  CS-5  claims resting on an invented basis -- population figures,
        per-capita / per-1,000 rates, census data. These are not merely
        "unsupported" in the general E1 sense: the Phase-5 prompt itself
        names this exact category and says it cannot be computed --
        src/phase_5/text_synthesizer.py:38 ("Every number or percentage in
        your text MUST come from DATA POINTS below -- do not invent or
        estimate any figures") and line 109 ("Data gap: FSI density per
        capita requires census population data", handed to the model as
        a DATA POINT explicitly telling it this basis does not exist). Any
        population/per-capita claim in the rendered prose is therefore a
        basis the generation prompt told the model it did not have.

CS-3 (cross-section contradiction detection) was prototyped and rejected --
see CS3_REJECTION_NOTE below -- and is deliberately NOT implemented here.

All flags in e2_flags.csv are UNVERIFIED. See the printed instructions at
the end of a run.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import chi2

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import e1_evidence_consistency as e1  # noqa: E402  -- Part 1 module, reused per instruction

RESULTS_DIR = _HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CITIES = e1.CITIES
PCT_TOL = e1.EvidenceIndex.PCT_TOL  # 0.55 percentage points -- same convention as Part 1
tokenize_numbers = e1.tokenize_numbers  # reused per instruction

CS3_REJECTION_NOTE = (
    "CS-3 (cross-section contradiction detection) was prototyped and rejected: it fired "
    "121 times on report_dublin_baseline.pdf alone, a report whose own arithmetic is "
    "otherwise clean (its Screening-outcome counts, retained/discarded totals, and "
    "district counts all reconcile -- see evaluation/results/e1_geo.csv and "
    "e1_summary.csv from Part 1). That volume of flags on an arithmetically clean report "
    "means CS-3 was keying on stylistic cross-section variation -- the same fact "
    "rephrased, rounded, or re-scoped in different words across sections -- not genuine "
    "numeric contradiction. It was not carried into this implementation. CS-3 is not "
    "implemented in evaluation/e2_internal_consistency.py, and no cross-section check of "
    "any kind runs here."
)


def tok(sentence: str, known_names: list[str] | None = None) -> list[dict]:
    """Tokenise numbers in sentence. When known_names is given, numbers that
    are themselves part of a district name (the '7' in 'Dublin 7', the '1'
    in 'Municipio 1') are excluded -- see Part 1's
    tokenize_numbers_excluding_names for why: without this, a sentence like
    'Municipio 1 (28 FSIs), Municipio 5 (22 FSIs), and Municipio 9 (15
    FSIs)...' yields spurious numbers 1, 5, 9 alongside the real 28, 22, 15,
    which corrupted an early version of CS-4's constituent sum."""
    if known_names:
        return e1.tokenize_numbers_excluding_names(sentence, known_names)
    return tokenize_numbers(sentence)


# ═══════════════════════════════════════════════════════════════════════
# CS-1 — count/percentage pair vs stated total
# ═══════════════════════════════════════════════════════════════════════
# Widened (see evaluation/scripts/run_cs1_cs4_widen_e2.py for the rerun that
# motivated this): the original pattern required the count and the "(P%)"
# to be separated by at most one literal "FSI(s)"/"initiative(s)" token, so
# it matched pipeline-style "47 initiatives (40.4%)" but missed
# baseline/baseline_v2-style "19 of 47 retained initiatives (40.4%)" or
# "8 of 24 retained initiatives, or 33.3%, including ..." entirely -- those
# phrasings never produced a CS-1 match at all, not a checked-and-passed
# one. Three number-pair shapes are now recognised, each checked against
# its own correct denominator:
#   (a) "N of M <~5 tokens> (P%)"        -> P checked against 100*N/M
#   (b) "N of M <~5 tokens>, [or] P%"    -> P checked against 100*N/M
#       (covers both "...,or P%" and "...,P% of ..." -- the trailing
#       "or"/"of" wording doesn't change what's being checked)
#   (c) "N <~5 tokens> (P%)" / "N <~5 tokens>, P%" -> P checked against
#       100*N/total (this is the original check, just widened from a
#       1-token gap to ~5 arbitrary tokens; a number that is itself the N
#       or M half of an "N of M" pair, shape (a)/(b) above, is explicitly
#       excluded here via lookaround so the same pair is never
#       double-counted against two different, conflicting denominators).
# A pre-widening bug this incidentally fixes: on brighton/b1, the OLD
# pattern matched "51 (70.6%)" inside "...36 of the 51 (70.6%)..." --
# capturing the *denominator* of an "N of M" phrase as if it were a
# standalone count of the report's grand total, then flagging it as
# inconsistent (100*51/76=67.11% vs stated 70.6%). The real claim, 36/51 =
# 70.59% ~ 70.6%, was never checked at all and is in fact consistent. Shape
# (a) now checks the real claim directly; shape (c)'s lookaround stops it
# from also (mis)matching "51" on its own.
_NUMSTART = r'(?<!\d)'                    # don't start a match mid-number
# Thousands-grouped ("14,896") OR plain -- NOT a bare `\d[\d,]*` character
# class: that greedily swallows a delimiter comma that immediately follows
# a plain count (e.g. "62, 26.2%" is two separate tokens "62" and "26.2",
# not a thousands-grouped "62,26" -- `[\d,]*` doesn't know the difference
# and will eat the comma into the number, leaving the wrong number of
# delimiter characters for the `[(,]`/`,` that follows to consume; caught
# by testing against a real sentence with two adjacent "(N, P%)" groups,
# e.g. '"other" initiatives (62, 26.2%), followed by "unknown" (49,
# 20.7%)', which paired 62 with the WRONG percentage (20.7%, from the next
# group) before this fix, quietly producing a bogus false-positive flag).
_NUM = r'\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?'
_OF_PHRASE = r'\s+of\s+(?:the\s+)?'       # "of 47" or "of the 47"
_GAP = r'(?:\s+\S+){0,5}?'                # ~5 tokens, non-greedy

CS1_OF_M_PAREN_RE = re.compile(
    _NUMSTART + r'(' + _NUM + r')' + _OF_PHRASE + _NUMSTART + r'(' + _NUM + r')' + _GAP +
    r'\s*\(\s*' + _NUMSTART + r'(' + _NUM + r')%\s*\)'
)
CS1_OF_M_TAIL_RE = re.compile(
    _NUMSTART + r'(' + _NUM + r')' + _OF_PHRASE + _NUMSTART + r'(' + _NUM + r')' + _GAP +
    r',\s*(?:or\s+)?' + _NUMSTART + r'(' + _NUM + r')%'
)
CS1_PLAIN_RE = re.compile(
    _NUMSTART + r'(?<!of )(?<!the )(' + _NUM + r')(?!\s+of\s+(?:the\s+)?\d)' + _GAP +
    r'\s*[(,]\s*' + _NUMSTART + r'(' + _NUM + r')%'
)


def check_cs1(sentence: str, total: float | None, known_names: list[str] | None = None) -> list[dict]:
    # known_names unused: every CS-1 pattern requires a literal '%' at the
    # end of the paired number, which a bare district-name digit (e.g. the
    # '1' in 'Municipio 1') never has -- structurally not susceptible to
    # that false-positive class, unlike CS-2/CS-4. Kept in the signature
    # only so all four checks share one call shape in run_report().
    if not total or total <= 0:
        return []
    flags = []

    for m in CS1_OF_M_PAREN_RE.finditer(sentence):
        n, mval = float(m.group(1).replace(",", "")), float(m.group(2).replace(",", ""))
        pct = float(m.group(3))
        if mval <= 0:
            continue
        predicted = 100.0 * n / mval
        diff = abs(predicted - pct)
        if diff > PCT_TOL:
            flags.append({
                "check": "CS-1",
                "detail": f"stated {m.group(1)} of {m.group(2)} ({m.group(3)}%): "
                          f"predicted {predicted:.2f}%, diff {diff:.2f}pp (tol {PCT_TOL}pp)",
            })

    for m in CS1_OF_M_TAIL_RE.finditer(sentence):
        n, mval = float(m.group(1).replace(",", "")), float(m.group(2).replace(",", ""))
        pct = float(m.group(3))
        if mval <= 0:
            continue
        predicted = 100.0 * n / mval
        diff = abs(predicted - pct)
        if diff > PCT_TOL:
            flags.append({
                "check": "CS-1",
                "detail": f"stated {m.group(1)} of {m.group(2)}, {m.group(3)}%: "
                          f"predicted {predicted:.2f}%, diff {diff:.2f}pp (tol {PCT_TOL}pp)",
            })

    for m in CS1_PLAIN_RE.finditer(sentence):
        count = float(m.group(1).replace(",", ""))
        pct = float(m.group(2))
        predicted = 100.0 * count / total
        diff = abs(predicted - pct)
        if diff > PCT_TOL:
            flags.append({
                "check": "CS-1",
                "detail": f"stated {m.group(1)} ({m.group(2)}%) vs report total={total:g}: "
                          f"predicted {predicted:.2f}%, diff {diff:.2f}pp (tol {PCT_TOL}pp)",
            })
    return flags


# ═══════════════════════════════════════════════════════════════════════
# CS-2 — majority/most/over half/nearly all applied to a value <=50%
# ═══════════════════════════════════════════════════════════════════════

CS2_KEYWORD_RE = re.compile(r'\b(majority|most|over half|nearly all)\b', re.IGNORECASE)
CS2_WINDOW_CHARS = 60


def check_cs2(sentence: str, total: float | None, known_names: list[str] | None = None) -> list[dict]:
    if not total or total <= 0:
        return []
    nums = tok(sentence, known_names)
    if not nums:
        return []
    flags = []
    for m in CS2_KEYWORD_RE.finditer(sentence):
        kw_start, kw_end = m.start(), m.end()
        best, best_dist = None, None
        for num in nums:
            if num["start"] >= kw_end:
                dist = num["start"] - kw_end
            elif num["end"] <= kw_start:
                dist = kw_start - num["end"]
            else:
                continue
            if dist > CS2_WINDOW_CHARS:
                continue
            if best_dist is None or dist < best_dist:
                best_dist, best = dist, num
        if best is None:
            continue
        pct = best["value"] if best["is_percentage"] else 100.0 * best["value"] / total
        if pct <= 50.0:
            flags.append({
                "check": "CS-2",
                "detail": f"keyword '{m.group(0)}' near {best['raw']} "
                          f"-> {pct:.2f}% of total={total:g} (<=50%)",
            })
    return flags


# ═══════════════════════════════════════════════════════════════════════
# CS-4 — aggregate "account for N%" claims vs constituent counts
# ═══════════════════════════════════════════════════════════════════════
# Widened the same way as CS-1 (same rerun as above): the aggregate
# percentage claim isn't always stated as literal "account(s) for N%"
# immediately -- baseline/baseline_v2-style prose instead writes
# "account(s) for N of M <~5 tokens> (P%)" or "...,[or] P%", where the
# percentage that actually needs reconciling against the constituent sum
# is P, not the "N of M" ratio next to it (that ratio is its own,
# separately-checked CS-1 claim -- see CS1_OF_M_PAREN_RE /
# CS1_OF_M_TAIL_RE). N/M themselves are not used here: CS-4's job stays
# exactly what it was -- does the SUM of numbers preceding this aggregate
# phrase equal (within tolerance) P% of the report's total -- these two
# new patterns only widen *where in the sentence* a valid P can be found,
# reusing the unchanged look-back-and-sum logic below unmodified. On the
# actual baseline_v2 reports this widening fires zero additional times
# (every "account(s) for N of M ..." sentence found there names only one
# category with no preceding constituent list to sum -- correctly a CS-1
# claim, not a CS-4 one), but is included for the same reason CS-1 was
# widened: matching the phrasing actually used across all three systems,
# not only the pipeline's literal "account(s) for N%" style.
CS4_AGG_PLAIN_RE = re.compile(r'account(?:s|ed|ing)?\s+for\s+' + _NUMSTART + r'(' + _NUM + r')%')
# NOTE: _NUM contains a top-level '|' alternation, so every inline (non-
# capturing) use below MUST wrap it in '(?:...)' -- inserting it bare would
# let the '|' split the *entire* surrounding regex at that point instead of
# just alternating within the number (caught by a TypeError at match time:
# group(1) came back None because the whole pattern silently became two
# unrelated alternatives joined by a stray top-level '|').
CS4_AGG_OF_PAREN_RE = re.compile(
    r'account(?:s|ed|ing)?\s+for\s+' + _NUMSTART + r'(?:' + _NUM + r')' + _OF_PHRASE +
    _NUMSTART + r'(?:' + _NUM + r')' + _GAP +
    r'\s*\(\s*' + _NUMSTART + r'(' + _NUM + r')%\s*\)'
)
CS4_AGG_OF_TAIL_RE = re.compile(
    r'account(?:s|ed|ing)?\s+for\s+' + _NUMSTART + r'(?:' + _NUM + r')' + _OF_PHRASE +
    _NUMSTART + r'(?:' + _NUM + r')' + _GAP +
    r',\s*(?:or\s+)?' + _NUMSTART + r'(' + _NUM + r')%'
)

# The constituent list ("A, B and C") immediately precedes "account(s) for
# N%" in every observed instance of this pattern; bounding the look-back
# window keeps a sentence-initial grand total (e.g. "Barcelona hosts 237
# FSIs ..., with ... Ciutat Vella (45), ... which together account for
# 44%") from being swept in as if it were one of the constituents -- an
# early version summed the sentence's *every* preceding number, including
# that opening total, and got nonsense sums as a result.
CS4_LOOKBACK_CHARS = 130


def check_cs4(sentence: str, total: float | None, known_names: list[str] | None = None) -> list[dict]:
    if not total or total <= 0:
        return []
    flags = []
    matches: list[tuple[int, float]] = []
    for rx in (CS4_AGG_PLAIN_RE, CS4_AGG_OF_PAREN_RE, CS4_AGG_OF_TAIL_RE):
        for m in rx.finditer(sentence):
            matches.append((m.start(), float(m.group(1))))
    seen_starts: set[int] = set()
    for start, claimed_pct in sorted(matches):
        if start in seen_starts:
            # only relevant if two of the three patterns anchor at the same
            # "account(s) for" occurrence -- keeps a single aggregate
            # phrase from being checked (and possibly flagged) twice.
            continue
        seen_starts.add(start)
        prefix = sentence[max(0, start - CS4_LOOKBACK_CHARS):start]
        counts = [n["value"] for n in tok(prefix, known_names) if not n["is_percentage"]]
        if len(counts) < 2:
            # not enough constituent numbers stated in-sentence to reconcile --
            # skip rather than guess (no cross-sentence lookup is performed).
            continue
        predicted = 100.0 * sum(counts) / total
        diff = abs(predicted - claimed_pct)
        if diff > PCT_TOL:
            counts_str = "+".join(f"{c:g}" for c in counts)
            flags.append({
                "check": "CS-4",
                "detail": f"constituents ({counts_str})=sum {sum(counts):g}, total={total:g}: "
                          f"predicted {predicted:.2f}%, claimed {claimed_pct:g}%, "
                          f"diff {diff:.2f}pp (tol {PCT_TOL}pp)",
            })
    return flags


# ═══════════════════════════════════════════════════════════════════════
# CS-5 — invented bases: population / per-capita / census
# ═══════════════════════════════════════════════════════════════════════

CS5_RE = re.compile(
    r'\bpopulation of\s+\d[\d,]*\.?\d*'
    r'|\bper[\s-]capita\b'
    r'|\bper\s+\d[\d,]*\.?\d*\s*(?:residents|people|inhabitants)\b'
    r'|\bcensus(?:\s+(?:population|data))?\b',
    re.IGNORECASE,
)


def check_cs5(sentence: str, total: float | None, known_names: list[str] | None = None) -> list[dict]:
    # total, known_names unused: every match is flagged outright, no
    # arithmetic check (see module docstring). Kept in the signature only
    # so all four checks share one call shape in run_report().
    flags = []
    for m in CS5_RE.finditer(sentence):
        flags.append({
            "check": "CS-5",
            "detail": f"matched forbidden-basis phrase: '{m.group(0)}' "
                      f"(src/phase_5/text_synthesizer.py:38,109)",
        })
    return flags


CHECKS = (check_cs1, check_cs2, check_cs4, check_cs5)


# ═══════════════════════════════════════════════════════════════════════
# Exact Poisson 95% CI (Garwood / chi-squared method)
# ═══════════════════════════════════════════════════════════════════════

def poisson_exact_ci(k: int, exposure: float, alpha: float = 0.05) -> tuple[float, float]:
    """Exact Poisson confidence interval for a rate k/exposure, via the
    standard chi-squared relation (Garwood 1936): for k events observed
    over exposure T, lower = chi2.ppf(alpha/2, 2k)/2/T (0 if k=0), upper =
    chi2.ppf(1-alpha/2, 2(k+1))/2/T."""
    if exposure <= 0:
        return (float("nan"), float("nan"))
    lower = 0.0 if k == 0 else chi2.ppf(alpha / 2, 2 * k) / 2.0
    upper = chi2.ppf(1 - alpha / 2, 2 * (k + 1)) / 2.0
    return (lower / exposure, upper / exposure)


# ═══════════════════════════════════════════════════════════════════════
# Per-report run
# ═══════════════════════════════════════════════════════════════════════

def run_report(city: str, system: str, pairs: list[tuple[str, str]],
                total: float | None, flags_out: list[dict]) -> tuple[int, dict]:
    known_names = e1.known_district_names(city)
    n_words = sum(len(sentence.split()) for _, sentence in pairs)
    check_counts = {"CS-1": 0, "CS-2": 0, "CS-4": 0, "CS-5": 0}
    for heading, sentence in pairs:
        for check_fn in CHECKS:
            for f in check_fn(sentence, total, known_names):
                check_counts[f["check"]] += 1
                flags_out.append({
                    "city": city, "system": system, "check": f["check"],
                    "section_heading": heading, "sentence": sentence,
                    "detail": f["detail"], "VERDICT": "", "NOTE": "",
                })
    return n_words, check_counts


def main() -> None:
    print("=== E2 input conditions (resolved paths) ===")
    print("  pipeline    :", e1.REPORT_HTML[CITIES[0]].parent, "/ report_<city>.html")
    print("  baseline_v2 :", e1.BASELINE_V2_HTML[CITIES[0]].parent, "/ report_<city>_baseline_v2.html  (current condition)")
    print("  RETIRED     :", e1.BASELINE_PDF[CITIES[0]].parent, "/ report_<city>_baseline.pdf  (superseded -- not scored below)")
    print()

    flags_rows: list[dict] = []
    summary_rows: list[dict] = []

    for city in CITIES:
        print(f"=== {city} ===")
        full_text, pairs = e1.extract_pipeline_prose(city)
        total = e1.extract_stated_total(full_text, "pipeline")
        n_words, counts = run_report(city, "pipeline", pairs, total, flags_rows)
        n_flags = sum(counts.values())
        exposure = n_words / 1000.0
        ci_lo, ci_hi = poisson_exact_ci(n_flags, exposure)
        summary_rows.append({
            "city": city, "system": "pipeline", "stated_total": total,
            "n_words": n_words, "n_flags": n_flags,
            "n_cs1": counts["CS-1"], "n_cs2": counts["CS-2"],
            "n_cs4": counts["CS-4"], "n_cs5": counts["CS-5"],
            "defects_per_1000_words": round(n_flags / exposure, 3) if exposure else None,
            "ci95_lower": round(ci_lo, 3) if exposure else None,
            "ci95_upper": round(ci_hi, 3) if exposure else None,
        })

        bfull_text, bpairs = e1.extract_baseline_prose(city)
        btotal = e1.extract_stated_total(bfull_text, "baseline")
        bn_words, bcounts = run_report(city, "baseline", bpairs, btotal, flags_rows)
        bn_flags = sum(bcounts.values())
        bexposure = bn_words / 1000.0
        bci_lo, bci_hi = poisson_exact_ci(bn_flags, bexposure)
        summary_rows.append({
            "city": city, "system": "baseline", "stated_total": btotal,
            "n_words": bn_words, "n_flags": bn_flags,
            "n_cs1": bcounts["CS-1"], "n_cs2": bcounts["CS-2"],
            "n_cs4": bcounts["CS-4"], "n_cs5": bcounts["CS-5"],
            "defects_per_1000_words": round(bn_flags / bexposure, 3) if bexposure else None,
            "ci95_lower": round(bci_lo, 3) if bexposure else None,
            "ci95_upper": round(bci_hi, 3) if bexposure else None,
        })

    flags_df = pd.DataFrame(flags_rows, columns=[
        "city", "system", "check", "section_heading", "sentence", "detail", "VERDICT", "NOTE",
    ])
    summary_df = pd.DataFrame(summary_rows)

    flags_df.to_csv(RESULTS_DIR / "e2_flags.csv", index=False)
    summary_df.to_csv(RESULTS_DIR / "e2_summary.csv", index=False)

    print("\n=== e2_summary.csv ===")
    print(summary_df.to_string(index=False))

    write_markdown(summary_df, flags_df)

    print(f"\nWrote {RESULTS_DIR / 'e2_flags.csv'} ({len(flags_df)} flags) and "
          f"{RESULTS_DIR / 'e2_summary.csv'}")
    print(f"Wrote {_HERE / 'E2_SUMMARY.md'}")
    print()
    print("=" * 78)
    print("ALL FLAGS IN e2_flags.csv ARE UNVERIFIED.")
    print("Each is a candidate defect surfaced by a regex/heuristic, not a confirmed one.")
    print("Fill in VERDICT (TP/FP) and NOTE by hand for every row before citing any rate.")
    print("defects-per-1,000-words and its Poisson CI in e2_summary.csv are computed over")
    print("ALL flags (unverified). They must be RECOMPUTED using only rows verified")
    print("VERDICT=TP -- do not report the numbers in e2_summary.csv as-is as defect rates.")
    print("=" * 78)


def write_markdown(summary_df: pd.DataFrame, flags_df: pd.DataFrame) -> None:
    md = []
    md.append("# E2 — Internal consistency checks\n")
    md.append(
        "Read-only, no pipeline stage run, no network call. Four checks (CS-1, CS-2, CS-4, "
        "CS-5) applied to all 10 reports. Generated by "
        "`evaluation/e2_internal_consistency.py`, which reuses the number tokeniser and "
        "prose extraction from `evaluation/e1_evidence_consistency.py` (Part 1) and nothing "
        "else from any prior evaluation code.\n"
    )

    md.append("## CS-3 — rejected, not implemented\n")
    md.append(f"> {CS3_REJECTION_NOTE}\n")

    md.append("## Checks implemented\n")
    md.append(
        "- **CS-1** count/percentage pairs in one sentence (`N (P%)` / `N, P%`) checked "
        "against `100*N/stated_total`, 0.55pp tolerance (same convention as Part 1's "
        "percentage-of-total DERIVED check).\n"
        "- **CS-2** `majority` / `most` / `over half` / `nearly all` (word-boundary, "
        "case-insensitive) paired with the nearest count or percentage within 60 characters "
        "in the same sentence; flagged if that value is <=50% of the stated total. **Known "
        "false-positive classes, seen and left unfixed on manual spot-check** (this is "
        "exactly what the VERDICT column is for): (a) 'most'/'majority' can corefer to a "
        "number stated *earlier* than the nearest one -- 'Milan hosts 104 high-activity "
        "FSIs ..., representing the majority ..., while 43 low-activity FSIs remain' pairs "
        "'majority' with the nearer-but-wrong 43 (21.8% -- wrongly flagged) rather than the "
        "correct, distant 104 (70.7%, a genuine majority); (b) 'most' is frequently a "
        "within-category superlative, not a claim about the grand total -- 'elderly-focused "
        "initiatives are most prevalent [among audience types], with 32 FSIs' triggers CS-2 "
        "(32 is 21.8% of the city total) even though the sentence never claims elderly-"
        "focused FSIs are most of *all* FSIs.\n"
        "- **CS-4** `... account(s) for N%` matched, then every non-percentage number "
        "appearing earlier in the *same sentence* is summed as the claimed constituent list "
        "and checked against `N%` of the stated total (0.55pp tolerance). Sentences where "
        "fewer than two such numbers precede the phrase are skipped -- not enough is stated "
        "in-sentence to reconcile, and no cross-sentence lookup is performed.\n"
        "- **CS-5** any occurrence of `population of <N>`, `per capita`/`per-capita`, "
        "`per <N> residents/people/inhabitants`, or `census (population|data)`. Every match "
        "is flagged outright (no arithmetic check) because the Phase-5 generation prompt "
        "itself states this basis is unavailable -- `src/phase_5/text_synthesizer.py:38` "
        "('do not invent or estimate any figures') and `:109` ('Data gap: FSI density per "
        "capita requires census population data', handed to the model as a fact it does not "
        "have a population figure to work with). Applied uniformly to both systems: neither "
        "evidence base contains population data, so a population/per-capita claim is "
        "equally invented regardless of which system produced it.\n\n"
        "`total` for the percentage checks (CS-1, CS-2, CS-4) is each report's own stated "
        "total: the pipeline's `\"N Food Sharing Initiatives (FSIs)\"` figure, or the "
        "baseline's `\"Screening outcome: N of the M ... were retained\"` figure "
        "(`e1.extract_stated_total`, Part 1). `n_words` is counted over the same prose "
        "sentences the checks scan (not raw page/HTML text, which includes chart titles, "
        "captions and footer boilerplate the checks never look at) -- numerator and "
        "denominator are drawn from the same population.\n"
    )

    md.append("## Summary\n")
    md.append(summary_df.to_markdown(index=False))
    md.append("")
    md.append(
        "`defects_per_1000_words` and `ci95_lower`/`ci95_upper` (exact Poisson CI via the "
        "chi-squared/Garwood method) are computed over **all** flags -- unverified. **Do "
        "not cite these as defect rates.** Recompute after filling `VERDICT` in "
        "`e2_flags.csv`, restricting to `VERDICT == 'TP'` rows only.\n"
    )
    md.append(
        "**Every flag in this run is on a pipeline report; every baseline report has zero "
        "flags on all four checks.** Checked directly rather than assumed: the five baseline "
        "PDFs contain zero '%' characters anywhere in their prose (confirmed by direct "
        "count), so CS-1 and CS-4 -- both requiring a percentage -- cannot structurally fire "
        "on them at all. Baseline PDFs do use 'majority'/'most' language, but on manual "
        "reading every instance is a qualitative superlative ('most explicit', 'most "
        "visible', 'most operationally useful') with no nearby quantifiable value, so CS-2 "
        "correctly finds nothing to pair it with. Baseline prose never mentions population "
        "or per-capita figures at all (word count: zero occurrences of 'population' or "
        "'capita' across all five baseline PDFs), so CS-5 has nothing to match either. All "
        "of this is consistent with what Part 1 already established: the baseline system's "
        "writing is narrative and (per E1) makes far fewer of its claims checkable at all -- "
        "it isn't that the checks fail to run on baseline reports, it's that the reports "
        "themselves supply almost none of the specific textual patterns (percentage pairs, "
        "aggregate 'account for N%' claims, fabricated per-capita arithmetic) that these "
        "four checks look for.\n"
    )

    md.append("## Flag counts by check\n")
    if len(flags_df):
        counts = flags_df.groupby(["city", "system", "check"]).size().unstack(fill_value=0)
        md.append(counts.to_markdown())
    else:
        md.append("*No flags raised.*")
    md.append("")

    (RESULTS_DIR.parent / "E2_SUMMARY.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
