#!/usr/bin/env python3
"""
Human screening reproducibility check, rejoined against baseline_v2.

Read-only. No model calls, no network calls, no scraping, no report
generation, no GraphRAG. Recomputes -- does not re-annotate -- the existing
E5 Set B human screening decisions against baseline_v2's own stored
Screening Ledger, instead of the retired pre-baseline_v2 PDF ledger they
were originally scored against.

── Why this script exists ──────────────────────────────────────────────────
E5 Set B (evaluation/e5_sample_for_annotation.py) drew 40 URLs (8 per city,
stratified 4 Retained / 4 Discarded) from the RETIRED baseline's PDF ledger
(report_<city>_baseline.pdf, via e1_evidence_consistency.parse_ledger_pdf)
and had a human re-judge each blind, independent of that ledger's own call.
That scoring (evaluation/scripts/e5_final_stats.py::do_set_b(), reported in
evaluation/results/e5_final_stats.md's Set B section) measures reproducibility of the RETIRED
baseline's screening decisions -- per the authoritative rule that only
baseline_v2 is valid as the final thesis's commercial baseline, that result
does not, and must not be read to, describe baseline_v2.

The 40 sampled URLs and the human's MY_DECISION labels are unaffected by
which ledger they are compared against -- a human decision about a URL does
not change depending on what the (retired or current) commercial pipeline
decided. This script keeps the human labels exactly as annotated and rejoins
them, by URL, against baseline_v2's ledger instead -- so this is not a new
interpretation of the baseline_v2 ledger, just the existing one applied to
an existing sample.

── What this measures, and what it does not ────────────────────────────────
This measures whether a blinded human, given only a URL, reproduces
baseline_v2's own semantic screening decision for that URL. It is:
  - NOT a validation of Phase 0 (Phase 0 is not compared here at all).
  - NOT classifier accuracy (there is no classifier under test; both sides
    are human-or-agent screening judgements, and neither is ground truth).
  - NOT evidence that the retired old-baseline ledger, or its own human
    validation, remains valid -- that result is superseded and out of scope.

── Reused vs ported vs new ──────────────────────────────────────────────────
Reused verbatim by partial AST exec (same technique
evaluation/scripts/run_baseline_v2_e3.py::load_url_helpers() and
evaluation/e1_evidence_consistency.py::load_parse_ledger_pdf() already use
in this repository to reuse one function without running a whole module):
  - normalize_url() straight out of analysis/corpus_analysis.py -- byte-
    identical to what run_baseline_v2_e3.py's own load_url_helpers() pulls.

Ported line-for-line (same algorithm, same td indices, same decision/bucket
logic -- NOT a reinterpretation), because this execution environment has no
`pandas` installed and this step is run with no network access, so the
pandas-dependent modules below cannot be live-imported here:
  - parse_ledger_html_v2_pure() below is
    evaluation/scripts/run_baseline_v2_e3.py::parse_ledger_html() (lines
    176-203 there), with the single change that it returns a list[dict]
    instead of a pandas.DataFrame(rows) at the end. Every BeautifulSoup
    call, td index, and decision/bucket rule is unchanged.
  - cohens_kappa_multiclass() is imported from evaluation/statistics_utils.py
    (the generic stats helpers extracted from the now-removed
    evaluation/e4_pairwise_judging.py pairwise-judging module -- see that
    module's own docstring). statistics_utils.py has no pandas/ollama
    dependency, so it can be imported directly in this environment.

New in this script: the URL join between e5_set_b.csv (human decisions) and
baseline_v2's 5 per-city ledgers, and the fail-fast checks that every one of
the 40 sampled URLs matches baseline_v2's ledger exactly once.

── Sampling-balance caveat (important, printed and written out below) ──────
The 40-row sample was stratified 4 Retained / 4 Discarded PER CITY against
the RETIRED ledger that existed at sampling time. That stratification does
NOT carry over to baseline_v2: baseline_v2 is a different agent run with its
own, independently-arrived-at Retained/Discarded calls, so the same 40 URLs
land on a different, generally not 4/4, Retained/Discarded split once rejoined
against baseline_v2's ledger. This script reports that actual baseline_v2
split (both pooled and per city) rather than assuming it is still 4/4.

── Verification tripwire ────────────────────────────────────────────────────
A prior read-only recomputation (analysis history / scratchpad, not itself a
repository artefact) found: 40/40 URLs matched, 35 decidable, 25/35 (71.4%)
raw agreement, Cohen's kappa ~0.432, confusion matrix (human rows x
baseline_v2 columns) [[14, 7], [3, 11]], and a baseline_v2 sample balance of
18 Retained / 22 Discarded. This script does NOT hard-code those numbers into
the computation -- it recomputes everything from the stored CSV/HTML files
below, and only asserts, at the very end, that the freshly computed values
match that prior recomputation. If they disagree, the script raises and
writes nothing, rather than silently reporting a different result.

Inputs (read-only):
  evaluation/results/e5_set_b.csv                       (human decisions)
  data/<city>/output/report_<city>_baseline_v2.html      (baseline_v2 ledger, x5)
  analysis/corpus_analysis.py                            (normalize_url, by AST exec only)

Outputs (written only after the tripwire above passes):
  evaluation/results/baseline_v2_human_screening_validation.csv
  evaluation/results/baseline_v2_human_screening_validation_summary.md

Run from the repo root:
    python3 evaluation/scripts/run_baseline_v2_human_screening_validation.py
"""
from __future__ import annotations

import ast
import csv
import sys
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup

_HERE = Path(__file__).resolve().parent
_EVAL_DIR = _HERE.parent
ROOT = _EVAL_DIR.parent
DATA = ROOT / "data"
RESULTS_DIR = _EVAL_DIR / "results"

sys.path.insert(0, str(_EVAL_DIR))
from statistics_utils import cohens_kappa_multiclass  # noqa: E402 -- reused, see module docstring

CITIES = ["barcelona", "brighton", "dublin", "london", "milan"]
BASELINE_V2_HTML = {c: DATA / c / "output" / f"report_{c}_baseline_v2.html" for c in CITIES}

SET_B_FILE = RESULTS_DIR / "e5_set_b.csv"
SET_B_EXCLUDE = {"UNREACHABLE_NOW", "UNCLEAR"}   # identical to e5_final_stats.py
SET_B_CLASSES = ["Retained", "Discarded"]        # identical to e5_final_stats.py

OUT_CSV = RESULTS_DIR / "baseline_v2_human_screening_validation.csv"
OUT_MD = RESULTS_DIR / "baseline_v2_human_screening_validation_summary.md"

# Prior read-only recomputation this script's own output must reproduce.
# See module docstring "Verification tripwire". Not used anywhere in the
# computation itself -- only compared against it at the very end.
EXPECTED_N_SAMPLED = 40
EXPECTED_N_DECIDABLE = 35
EXPECTED_N_AGREE = 25
EXPECTED_RAW_AGREEMENT_PCT = 71.4
EXPECTED_KAPPA = 0.432
EXPECTED_KAPPA_TOL = 0.001
EXPECTED_CONFUSION = {
    ("Retained", "Retained"): 14,
    ("Retained", "Discarded"): 7,
    ("Discarded", "Retained"): 3,
    ("Discarded", "Discarded"): 11,
}
EXPECTED_BALANCE = {"Retained": 18, "Discarded": 22}


# ═══════════════════════════════════════════════════════════════════════
# Reused verbatim: normalize_url(), pulled by partial AST exec straight out
# of analysis/corpus_analysis.py -- the same reuse technique
# run_baseline_v2_e3.py::load_url_helpers() and
# e1_evidence_consistency.py::load_parse_ledger_pdf() already use elsewhere
# in this repository. Only this one self-contained function is executed;
# the rest of analysis/corpus_analysis.py (and its own pandas dependency)
# is never touched.
# ═══════════════════════════════════════════════════════════════════════

def load_normalize_url():
    src_path = ROOT / "analysis" / "corpus_analysis.py"
    source = src_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(src_path))
    wanted = {"normalize_url"}
    nodes = [n for n in tree.body if getattr(n, "name", None) in wanted]
    found = {n.name for n in nodes}
    missing = wanted - found
    if missing:
        raise RuntimeError(f"analysis/corpus_analysis.py: could not locate {missing} for reuse")
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    from urllib.parse import urlparse
    ns = {"urlparse": urlparse}
    exec(compile(module, filename=str(src_path), mode="exec"), ns)
    return ns["normalize_url"]


normalize_url = load_normalize_url()


# ═══════════════════════════════════════════════════════════════════════
# Ported line-for-line from
# evaluation/scripts/run_baseline_v2_e3.py::parse_ledger_html() (its lines
# 176-203), pandas.DataFrame(rows) swapped for a plain list[dict] because
# pandas is not installed in this execution environment. Every
# BeautifulSoup call, td index, and decision/bucket rule below is
# unchanged from that source function -- see this repository's own
# run_baseline_v2_e3.py for the original.
# ═══════════════════════════════════════════════════════════════════════

def parse_ledger_html_v2_pure(city: str) -> list[dict]:
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
    return rows


# ═══════════════════════════════════════════════════════════════════════
# New: load the 40 human decisions, load baseline_v2's 5 ledgers, join.
# ═══════════════════════════════════════════════════════════════════════

def load_human_decisions() -> list[dict]:
    with open(SET_B_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    expected_cols = {"item_id", "url", "city", "MY_DECISION", "MY_NOTE", "SECONDS_SPENT"}
    if rows and not expected_cols.issubset(rows[0].keys()):
        raise RuntimeError(f"{SET_B_FILE}: missing expected column(s) {expected_cols - set(rows[0].keys())}")
    if len(rows) != EXPECTED_N_SAMPLED:
        raise RuntimeError(
            f"{SET_B_FILE}: expected {EXPECTED_N_SAMPLED} rows, found {len(rows)} -- "
            "the human annotation artefact should not have changed size."
        )
    blanks = [r["item_id"] for r in rows if not r["MY_DECISION"].strip()]
    if blanks:
        raise RuntimeError(f"{SET_B_FILE}: blank MY_DECISION for item_id(s): {blanks}")
    return rows


def load_baseline_v2_ledgers() -> dict[str, list[dict]]:
    ledgers = {}
    for city in CITIES:
        if not BASELINE_V2_HTML[city].exists():
            raise FileNotFoundError(f"baseline_v2 report missing for {city}: {BASELINE_V2_HTML[city]}")
        rows = parse_ledger_html_v2_pure(city)
        for r in rows:
            r["url_norm"] = normalize_url(r["url"]) if r["url"] else None
        ledgers[city] = rows
    return ledgers


def match_url(url: str, ledger: list[dict]) -> tuple[str, str]:
    """Exact match first, normalised-URL match as fallback. Returns
    (baseline_v2_decision, match_type) and raises if the match is missing or
    ambiguous (>1 row) -- every one of the 40 sampled URLs must match
    baseline_v2's ledger exactly/unambiguously."""
    exact = [r for r in ledger if r["url"] == url]
    if len(exact) == 1:
        return exact[0]["decision"], "exact"
    if len(exact) > 1:
        raise RuntimeError(f"Ambiguous exact URL match in baseline_v2 ledger for {url!r} ({len(exact)} rows)")

    norm = normalize_url(url)
    norm_match = [r for r in ledger if r["url_norm"] == norm]
    if len(norm_match) == 1:
        return norm_match[0]["decision"], "normalized"
    if len(norm_match) > 1:
        raise RuntimeError(f"Ambiguous normalised URL match in baseline_v2 ledger for {url!r} ({len(norm_match)} rows)")

    raise RuntimeError(f"No baseline_v2 ledger match (exact or normalised) for {url!r}")


def build_joined(human_rows: list[dict], ledgers: dict[str, list[dict]]) -> list[dict]:
    joined = []
    for r in human_rows:
        city = r["city"]
        if city not in ledgers:
            raise RuntimeError(f"No baseline_v2 ledger loaded for city {city!r}")
        bv2_decision, match_type = match_url(r["url"], ledgers[city])
        joined.append({
            "item_id": r["item_id"],
            "city": city,
            "url": r["url"],
            "MY_DECISION": r["MY_DECISION"],
            "MY_NOTE": r["MY_NOTE"],
            "baseline_v2_decision": bv2_decision,
            "match_type": match_type,
        })
    if len(joined) != EXPECTED_N_SAMPLED:
        raise RuntimeError(f"Expected {EXPECTED_N_SAMPLED} joined rows, got {len(joined)}")
    bad = [j["item_id"] for j in joined if j["baseline_v2_decision"] not in ("Retained", "Discarded")]
    if bad:
        raise RuntimeError(f"baseline_v2 ledger decision is not Retained/Discarded for item_id(s) {bad}")
    return joined


def compute_metrics(joined: list[dict]) -> dict:
    for j in joined:
        j["included_in_agreement"] = j["MY_DECISION"] not in SET_B_EXCLUDE

    included = [j for j in joined if j["included_in_agreement"]]
    excluded = [j for j in joined if not j["included_in_agreement"]]

    confusion = {(a, b): 0 for a in SET_B_CLASSES for b in SET_B_CLASSES}
    for j in included:
        confusion[(j["MY_DECISION"], j["baseline_v2_decision"])] += 1

    n_incl = len(included)
    n_agree = sum(1 for j in included if j["MY_DECISION"] == j["baseline_v2_decision"])
    raw_agreement = n_agree / n_incl if n_incl else float("nan")
    kappa = cohens_kappa_multiclass(
        [j["MY_DECISION"] for j in included],
        [j["baseline_v2_decision"] for j in included],
    )

    per_city = []
    for city in CITIES:
        sub = [j for j in included if j["city"] == city]
        n = len(sub)
        agree = sum(1 for j in sub if j["MY_DECISION"] == j["baseline_v2_decision"])
        per_city.append({"city": city, "n_included": n, "n_agree": agree,
                          "raw_agreement_pct": round(100 * agree / n, 1) if n else None})

    balance_pooled = Counter(j["baseline_v2_decision"] for j in joined)
    balance_per_city = {
        city: Counter(j["baseline_v2_decision"] for j in joined if j["city"] == city)
        for city in CITIES
    }

    return {
        "joined": joined,
        "included": included,
        "excluded": excluded,
        "confusion": confusion,
        "n_sampled": len(joined),
        "n_included": n_incl,
        "n_excluded": len(excluded),
        "n_agree": n_agree,
        "raw_agreement": raw_agreement,
        "kappa": kappa,
        "per_city": per_city,
        "balance_pooled": balance_pooled,
        "balance_per_city": balance_per_city,
    }


def verify_against_prior_recomputation(m: dict) -> None:
    """Fails loudly (raises, writes nothing) if the freshly computed values
    disagree with the prior read-only recomputation this script exists to
    turn into a repository artefact. See module docstring."""
    problems = []

    if m["n_sampled"] != EXPECTED_N_SAMPLED:
        problems.append(f"n_sampled: got {m['n_sampled']}, expected {EXPECTED_N_SAMPLED}")
    if m["n_included"] != EXPECTED_N_DECIDABLE:
        problems.append(f"n_included (decidable): got {m['n_included']}, expected {EXPECTED_N_DECIDABLE}")
    if m["n_agree"] != EXPECTED_N_AGREE:
        problems.append(f"n_agree: got {m['n_agree']}, expected {EXPECTED_N_AGREE}")

    got_pct = round(100 * m["raw_agreement"], 1)
    if abs(got_pct - EXPECTED_RAW_AGREEMENT_PCT) > 0.05:
        problems.append(f"raw_agreement: got {got_pct}%, expected {EXPECTED_RAW_AGREEMENT_PCT}%")

    if abs(m["kappa"] - EXPECTED_KAPPA) > EXPECTED_KAPPA_TOL:
        problems.append(f"cohens_kappa: got {m['kappa']:.4f}, expected {EXPECTED_KAPPA:.4f} (+/-{EXPECTED_KAPPA_TOL})")

    for (my_cls, bv2_cls), expected_n in EXPECTED_CONFUSION.items():
        got_n = m["confusion"][(my_cls, bv2_cls)]
        if got_n != expected_n:
            problems.append(f"confusion[{my_cls}][{bv2_cls}]: got {got_n}, expected {expected_n}")

    for cls, expected_n in EXPECTED_BALANCE.items():
        got_n = m["balance_pooled"].get(cls, 0)
        if got_n != expected_n:
            problems.append(f"baseline_v2 sample balance[{cls}]: got {got_n}, expected {expected_n}")

    if problems:
        raise RuntimeError(
            "Recomputed baseline_v2 human-screening-validation values disagree with "
            "the prior read-only recomputation this script is meant to reproduce. "
            "Refusing to write output files.\n  - " + "\n  - ".join(problems)
        )


def write_outputs(m: dict) -> None:
    fieldnames = ["item_id", "city", "url", "MY_DECISION", "MY_NOTE",
                  "baseline_v2_decision", "match_type", "included_in_agreement"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for j in m["joined"]:
            writer.writerow({k: j[k] for k in fieldnames})

    confusion = m["confusion"]
    balance = m["balance_pooled"]
    balance_city = m["balance_per_city"]

    lines = []
    lines.append("# Human screening reproducibility -- baseline_v2\n")
    lines.append(
        "Read-only recomputation. No model calls, no network calls, no scraping, "
        "no GraphRAG, no report generation. Generated by "
        "`evaluation/scripts/run_baseline_v2_human_screening_validation.py`.\n"
    )
    lines.append(
        "**What this is:** the 40 human screening decisions from E5 Set B "
        "(`evaluation/results/e5_set_b.csv`) -- unchanged, not re-annotated -- "
        "rejoined by exact/normalised URL match against **baseline_v2's own "
        "stored Screening Ledger** (`report_<city>_baseline_v2.html`, parsed "
        "with the same ledger-table logic "
        "`evaluation/scripts/run_baseline_v2_e3.py::parse_ledger_html()` uses). "
        "The human labels were created before baseline_v2 existed, blind to "
        "any ledger's decision; this script rejoins them against baseline_v2 "
        "now that baseline_v2 is the final thesis's commercial baseline "
        "condition. This measures **reproducibility of baseline_v2's own "
        "semantic screening decisions** by an independent human working from "
        "the same URL.\n"
    )
    lines.append(
        "**What this is not:** it does NOT validate Phase 0 (Phase 0 is not "
        "compared here). It is NOT classifier accuracy (neither side is "
        "ground truth). It does NOT imply the retired old-baseline ledger, or "
        "its own human-validation result (`evaluation/results/e5_final_stats.md`'s "
        "Set B section), remains valid -- that prior result describes a "
        "different, retired commercial condition (`report_<city>_baseline.pdf`) "
        "and is out of scope here.\n"
    )
    lines.append(
        "**Sampling-balance caveat:** the 40-row sample was drawn stratified "
        "4 Retained / 4 Discarded per city against the *retired* ledger at "
        "sampling time. That 4/4 split does not carry over to baseline_v2 -- "
        "baseline_v2 is a different, independent agent run with its own "
        "Retained/Discarded calls for the same URLs. The baseline_v2 "
        "Retained/Discarded balance for these same 40 URLs is reported below "
        "as actually observed, not assumed to still be 4/4.\n"
    )

    lines.append(f"- n sampled: {m['n_sampled']}")
    lines.append(f"- n excluded (MY_DECISION is UNREACHABLE_NOW or UNCLEAR): {m['n_excluded']}")
    lines.append(f"- n decidable (included in agreement arithmetic): {m['n_included']}")
    lines.append("")

    lines.append("**Excluded rows:**\n")
    lines.append("| item_id | city | MY_DECISION | MY_NOTE |")
    lines.append("|---|---|---|---|")
    for j in sorted(m["excluded"], key=lambda r: r["item_id"]):
        note = j["MY_NOTE"].strip() or "(no note)"
        lines.append(f"| {j['item_id']} | {j['city']} | {j['MY_DECISION']} | {note} |")
    lines.append("")

    lines.append("**Confusion matrix** (rows = human decision, columns = baseline_v2 ledger decision):\n")
    lines.append("| | baseline_v2 Retained | baseline_v2 Discarded |")
    lines.append("|---|---|---|")
    lines.append(f"| human Retained | {confusion[('Retained', 'Retained')]} | {confusion[('Retained', 'Discarded')]} |")
    lines.append(f"| human Discarded | {confusion[('Discarded', 'Retained')]} | {confusion[('Discarded', 'Discarded')]} |")
    lines.append("")

    lines.append(f"- raw agreement: {m['n_agree']}/{m['n_included']} ({100 * m['raw_agreement']:.1f}%)")
    lines.append(f"- Cohen's kappa: {m['kappa']:.4f}")
    n_ret_disc = confusion[("Retained", "Discarded")]
    n_disc_ret = confusion[("Discarded", "Retained")]
    lines.append(
        f"- directionality: the human retained {n_ret_disc} that baseline_v2 discarded; "
        f"the human discarded {n_disc_ret} that baseline_v2 retained.\n"
    )

    lines.append("**Per-city agreement** (decidable rows only):\n")
    lines.append("| city | n included | agree | raw agreement |")
    lines.append("|---|---|---|---|")
    for row in m["per_city"]:
        lines.append(f"| {row['city']} | {row['n_included']} | {row['n_agree']} | {row['raw_agreement_pct']}% |")
    lines.append("")

    lines.append(
        "**baseline_v2 Retained/Discarded balance in the 40-row sample** "
        "(all 40 sampled URLs, including the 5 excluded from agreement "
        "arithmetic -- this is baseline_v2's own decision for each URL, not "
        "the human's):\n"
    )
    lines.append(f"- pooled: {balance.get('Retained', 0)} Retained / {balance.get('Discarded', 0)} Discarded")
    lines.append("")
    lines.append("| city | baseline_v2 Retained | baseline_v2 Discarded |")
    lines.append("|---|---|---|")
    for city in CITIES:
        c = balance_city[city]
        lines.append(f"| {city} | {c.get('Retained', 0)} | {c.get('Discarded', 0)} |")
    lines.append("")

    lines.append(
        "**Verification:** this run's recomputed values were checked against a "
        "prior read-only recomputation and matched exactly (see module "
        "docstring's \"Verification tripwire\"); the script would have raised "
        "and written nothing otherwise.\n"
    )

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    human_rows = load_human_decisions()
    ledgers = load_baseline_v2_ledgers()
    joined = build_joined(human_rows, ledgers)
    metrics = compute_metrics(joined)
    verify_against_prior_recomputation(metrics)
    write_outputs(metrics)

    print(f"n sampled:     {metrics['n_sampled']}")
    print(f"n decidable:   {metrics['n_included']}")
    print(f"n agree:       {metrics['n_agree']}")
    print(f"raw agreement: {100 * metrics['raw_agreement']:.1f}%")
    print(f"cohen's kappa: {metrics['kappa']:.4f}")
    print("confusion matrix (rows=human, cols=baseline_v2):")
    for my_cls in SET_B_CLASSES:
        print(f"  {my_cls:10s} " + "  ".join(
            f"{bv2_cls}={metrics['confusion'][(my_cls, bv2_cls)]}" for bv2_cls in SET_B_CLASSES
        ))
    print("baseline_v2 sample balance:", dict(metrics["balance_pooled"]))
    print()
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
