"""
Replication/stability analysis of the Phase-5 Full/B1 replication experiment
(evaluation/run_b1_replication.py's output).

Purpose: determine whether the original one-shot Full-vs-B1 numeric-claim
overlap (71.6%, pooled, evaluation/results/b1_claim_overlap.csv) is
distinguishable from ordinary run-to-run Phase-5 generation variability, by
comparing it against the observed distribution of Full-vs-Full, B1-vs-B1, and
Full-vs-B1 overlaps across 3 new replicates per condition per city. This is a
DESCRIPTIVE STABILITY ANALYSIS, not a significance test -- see
analysis_summary.md's INTERPRETATION LIMITS section, always written, for what
n=3 per condition does and does not support.

READ-ONLY against:
  - evaluation/results/b1_replication/<city>/{full,b1}_rep{1,2,3}.json
    (never opened for writing -- see _assert_safe_output_path)
  - evaluation/results/b1_replication/manifest.json (validated before any
    analysis runs -- see validate_manifest(): integrity_check, changed-file
    list, model/digest/temperature/num_ctx, replicate/condition/city counts,
    created_files, and per-replicate agreement all checked; any mismatch
    aborts before analysis starts)
  - data/<city>/output/fsi_enriched.jsonl, data/<city>/districts.geojson
    (read via e1_evidence_consistency's own functions, exactly as the
    existing E1/E2 instruments already read them -- required to build the
    same evidence base / district-name list those instruments use; nothing
    under data/ is ever opened in a write mode anywhere in this script)

WRITES ONLY under evaluation/results/b1_replication/analysis/ -- enforced by
_assert_safe_output_path(), called before every single write in this file,
which rejects any path that is not a direct child of that directory or is
not one of the explicitly allowed output filenames. Never imports ollama,
never imports anything from src/phase_3 (GraphRAG), never calls
synthesize_all() or any other generation function -- this script only reads
already-generated JSON and existing evaluation code.

── Which numeric-claim tokenizer this script uses, and why (do not change
   without re-reading this) ──────────────────────────────────────────────
Two different number-token regexes exist in this codebase's B1-comparison
work:
  1. runs/b1_diff.py's NUM_RE = r"\\b\\d[\\d,]*(?:\\.\\d+)?%?\\b" -- textually
     the same shape as e1_evidence_consistency.py's own documented-buggy
     NUMBER_RE_OLD: the trailing \\b after the optional %? only holds when
     % is the very last character in the whole string, so in real prose
     ("44.3%," / "44.3%)") it almost never holds and the regex silently
     backtracks to matching the bare number without the %. b1_diff.py never
     writes a CSV or computes an overlap percentage itself (it only prints a
     JSON diff blob for manual inspection) -- it did not produce 71.6%.
  2. e1_evidence_consistency.py's tokenize_numbers()/NUMBER_RE_NEW -- the
     corrected tokenizer (thousands-grouped alternation, negative lookahead
     instead of trailing \\b). This is what actually produced
     evaluation/results/e1_claims.csv's raw_number column, which
     evaluation/scripts/run_b1_claim_overlap.py then consumed (matching
     claims by the exact (section_heading, raw_number) pair, per that
     script's own docstring) to compute the 71.6% pooled figure this
     analysis compares against.
Using b1_diff.py's regex here would make every new overlap figure NOT
comparable to 71.6% (different, buggier tokenization), silently invalidating
the one comparison this whole experiment exists to make. This script
therefore uses e1_evidence_consistency.tokenize_numbers() (imported, not
reimplemented) for ALL numeric-claim extraction, and the
(section, raw_number)-keyed set-difference/overlap_pct formula from
run_b1_claim_overlap.py (100 * |both| / |union|, reproduced here as
overlap_stats() -- same formula, not reimplemented differently). What IS
reused unmodified from runs/b1_diff.py: its SECTIONS title mapping, and its
difflib.SequenceMatcher(None, a, b).ratio() character-similarity method (see
char_similarity()) -- neither of those has the tokenizer bug.

Repeated numbers within one section collapse to one set member (matching
both b1_diff.py's numbers_in() and run_b1_claim_overlap.py's set(zip(...)));
across sections the same literal token string is kept as two distinct
(section, token) claims, never collapsed. Percentages keep their trailing
'%' as part of the token string and are never treated as equal to the same
digits without '%'. Integers/decimals are compared as literal strings (no
float-parsing equality) -- "34" and "34.0" are different tokens, "10,000"
and "10000" are different tokens -- exactly as e1_claims.csv/
b1_claim_overlap.csv already treat them; no new normalisation is introduced.

Usage:
    python evaluation/analyze_b1_replication.py --dry-run
    python evaluation/analyze_b1_replication.py
    python evaluation/analyze_b1_replication.py --overwrite
"""
from __future__ import annotations

import argparse
import csv
import difflib
import itertools
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "evaluation"
RUNS_DIR = ROOT / "runs"

sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(RUNS_DIR))

import e1_evidence_consistency as e1  # noqa: E402 -- reused unmodified
import e2_internal_consistency as e2  # noqa: E402 -- reused unmodified
import b1_diff  # noqa: E402 -- SECTIONS list + similarity pattern reused

CITIES = ["barcelona", "brighton", "dublin", "london", "milan"]
CONDITIONS = ["full", "b1"]
N_REPLICATES = 3
SECTIONS = b1_diff.SECTIONS  # [(key, title), ...] -- reused, not redefined
SECTION_KEYS = [k for k, _ in SECTIONS]
SECTION_TITLES = dict(SECTIONS)

INPUT_ROOT = (ROOT / "evaluation" / "results" / "b1_replication").resolve()
OUTPUT_ROOT = (INPUT_ROOT / "analysis").resolve()
MANIFEST_PATH = INPUT_ROOT / "manifest.json"
DATA_ROOT = (ROOT / "data").resolve()

# Expected generation-environment values, checked against
# evaluation/results/b1_replication/manifest.json before any analysis runs
# (see validate_manifest()). These describe the completed generation run,
# not an analysis parameter -- do not confuse with anything in the
# numeric/E1/E2 analysis logic below, none of which this constant touches.
EXPECTED_MODEL = "qwen3:14b"
EXPECTED_MODEL_DIGEST = "bdbd181c33f2"
EXPECTED_TEMPERATURE = 0
EXPECTED_NUM_CTX = 16384

ALLOWED_OUTPUT_NAMES = {
    "numeric_pairwise.csv", "numeric_summary.csv", "section_summary.csv",
    "digital_control.csv", "claim_stability.csv", "text_similarity.csv",
    "e1_replication.csv", "e2_candidate_flags.csv", "analysis_summary.md",
}

# ── Module-load-time safety assertions ─────────────────────────────────────
assert DATA_ROOT not in OUTPUT_ROOT.parents and OUTPUT_ROOT != DATA_ROOT, \
    "OUTPUT_ROOT must not resolve inside data/"
assert INPUT_ROOT in OUTPUT_ROOT.parents, \
    "OUTPUT_ROOT must be a subdirectory of the replication input root"

# The original one-shot pooled overlap figure, read verbatim (not recomputed)
# from evaluation/results/b1_claim_overlap.csv, row_type=summary,
# city=ALL_POOLED, overlap_pct column. Kept as a literal constant with its
# exact provenance documented, per the instruction not to mix the original
# reports into the new replicate dataset.
ORIGINAL_ONE_SHOT_OVERLAP_PCT = 71.6
ORIGINAL_ONE_SHOT_SOURCE = "evaluation/results/b1_claim_overlap.csv (row_type=summary, city=ALL_POOLED)"


def _assert_safe_output_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == DATA_ROOT or DATA_ROOT in resolved.parents:
        raise RuntimeError(f"SAFETY ABORT: refusing to write under data/: {resolved}")
    if resolved.parent != OUTPUT_ROOT:
        raise RuntimeError(f"SAFETY ABORT: output path must be a direct child of {OUTPUT_ROOT}: {resolved}")
    if resolved.name not in ALLOWED_OUTPUT_NAMES:
        raise RuntimeError(f"SAFETY ABORT: filename not in the allowed output list: {resolved.name}")
    return resolved


# ═══════════════════════════════════════════════════════════════════════
# Loading + validating replicates (read-only)
# ═══════════════════════════════════════════════════════════════════════

def _replicate_path(city: str, condition: str, rep: int) -> Path:
    return INPUT_ROOT / city / f"{condition}_rep{rep}.json"


def discover_and_validate() -> tuple[dict[tuple[str, str, int], dict], list[str]]:
    """Reads every expected replicate JSON (read-only) and checks: exactly
    30 files present; each prose dict has exactly the 6 expected section
    keys; city/condition/replicate metadata inside the file matches what its
    path implies; model/digest/temperature/num_ctx identical across all 30.
    Returns (records, errors) -- if errors is non-empty the caller must stop
    before running any analysis (see main())."""
    errors: list[str] = []
    records: dict[tuple[str, str, int], dict] = {}
    expected_section_keys = set(SECTION_KEYS)

    for city in CITIES:
        for condition in CONDITIONS:
            for rep in range(1, N_REPLICATES + 1):
                path = _replicate_path(city, condition, rep)
                if not path.exists():
                    errors.append(f"MISSING replicate file: {path}")
                    continue
                try:
                    rec = json.loads(path.read_text(encoding="utf-8"))
                except Exception as exc:
                    errors.append(f"UNREADABLE replicate file {path}: {exc}")
                    continue

                if rec.get("city") != city:
                    errors.append(f"{path}: recorded city {rec.get('city')!r} != expected {city!r}")
                if rec.get("condition") != condition:
                    errors.append(f"{path}: recorded condition {rec.get('condition')!r} != expected {condition!r}")
                if rec.get("replicate") != rep:
                    errors.append(f"{path}: recorded replicate {rec.get('replicate')!r} != expected {rep}")

                prose = rec.get("prose", {})
                prose_keys = set(prose.keys())
                if prose_keys != expected_section_keys:
                    errors.append(
                        f"{path}: prose section keys {sorted(prose_keys)} != "
                        f"expected {sorted(expected_section_keys)}"
                    )

                records[(city, condition, rep)] = rec

    if len(records) != len(CITIES) * len(CONDITIONS) * N_REPLICATES:
        errors.append(
            f"expected {len(CITIES) * len(CONDITIONS) * N_REPLICATES} replicate files, "
            f"found {len(records)} readable+located"
        )

    triple_counts = Counter(
        (r.get("city"), r.get("condition"), r.get("replicate")) for r in records.values()
    )
    dupes = [k for k, v in triple_counts.items() if v > 1]
    if dupes:
        errors.append(f"duplicate (city, condition, replicate) identifiers found: {dupes}")

    configs = set()
    for r in records.values():
        configs.add((r.get("model"), r.get("model_digest"), r.get("temperature"), r.get("num_ctx")))
    if len(configs) > 1:
        errors.append(f"inconsistent model/digest/temperature/num_ctx across replicates: {sorted(configs)}")

    return records, errors


def validate_manifest(records: dict) -> tuple[list[str], dict | None]:
    """Reads evaluation/results/b1_replication/manifest.json (read-only) and
    checks it before any analysis runs. Returns (errors, manifest_dict) --
    manifest_dict is None if the file is missing/unreadable. Does not alter
    any analysis method, metric, tokenizer, E1 logic, E2 logic, output
    filename, or comparison formula -- this function only gates whether
    main() proceeds to call those, exactly like discover_and_validate()
    already does for the replicate files themselves."""
    if not MANIFEST_PATH.exists():
        return [f"MISSING manifest file: {MANIFEST_PATH}"], None
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"UNREADABLE manifest file {MANIFEST_PATH}: {exc}"], None

    errors: list[str] = []

    # 1. integrity_check == "OK"
    if manifest.get("integrity_check") != "OK":
        errors.append(f"manifest integrity_check != 'OK': {manifest.get('integrity_check')!r}")

    # 2. integrity_changed_files empty
    changed = manifest.get("integrity_changed_files")
    if changed:
        errors.append(f"manifest integrity_changed_files is non-empty: {changed}")

    # 3-6. generation-environment values
    if manifest.get("model") != EXPECTED_MODEL:
        errors.append(f"manifest model != {EXPECTED_MODEL!r}: {manifest.get('model')!r}")
    if manifest.get("model_digest") != EXPECTED_MODEL_DIGEST:
        errors.append(f"manifest model_digest != {EXPECTED_MODEL_DIGEST!r}: {manifest.get('model_digest')!r}")
    if manifest.get("temperature") != EXPECTED_TEMPERATURE:
        errors.append(f"manifest temperature != {EXPECTED_TEMPERATURE!r}: {manifest.get('temperature')!r}")
    if manifest.get("num_ctx") != EXPECTED_NUM_CTX:
        errors.append(f"manifest num_ctx != {EXPECTED_NUM_CTX!r}: {manifest.get('num_ctx')!r}")

    # 7. n_replicates_per_condition == 3
    if manifest.get("n_replicates_per_condition") != N_REPLICATES:
        errors.append(
            f"manifest n_replicates_per_condition != {N_REPLICATES}: "
            f"{manifest.get('n_replicates_per_condition')!r}"
        )

    # 8. conditions == {"full", "b1"}
    manifest_conditions = set(manifest.get("conditions") or [])
    if manifest_conditions != set(CONDITIONS):
        errors.append(f"manifest conditions {manifest_conditions} != expected {set(CONDITIONS)}")

    # 9. cities == the 5 expected cities
    manifest_cities = set(manifest.get("cities") or [])
    if manifest_cities != set(CITIES):
        errors.append(f"manifest cities {manifest_cities} != expected {set(CITIES)}")

    # 10. created_files contains exactly the 30 expected replicate paths
    expected_created = {
        str(_replicate_path(city, condition, rep).relative_to(ROOT))
        for city in CITIES for condition in CONDITIONS for rep in range(1, N_REPLICATES + 1)
    }
    manifest_created = set(manifest.get("created_files") or [])
    if manifest_created != expected_created:
        missing = expected_created - manifest_created
        extra = manifest_created - expected_created
        detail = []
        if missing:
            detail.append(f"missing from manifest.created_files: {sorted(missing)}")
        if extra:
            detail.append(f"unexpected entries in manifest.created_files: {sorted(extra)}")
        errors.append(
            "manifest created_files does not exactly match the 30 expected replicate files; "
            + "; ".join(detail)
        )

    # 11. every already-loaded replicate JSON agrees with the manifest for
    # model / model_digest (if stored) / temperature / num_ctx / city /
    # condition / replicate number. The city/condition/replicate agreement
    # is also checked, independently, inside discover_and_validate() against
    # each file's own path -- repeated here, against the manifest's implied
    # identity for that same path, per the explicit ask, not a logic change.
    for (city, condition, rep), rec in records.items():
        path = _replicate_path(city, condition, rep)
        if rec.get("model") != manifest.get("model"):
            errors.append(f"{path}: model {rec.get('model')!r} != manifest model {manifest.get('model')!r}")
        if "model_digest" in rec and rec.get("model_digest") != manifest.get("model_digest"):
            errors.append(
                f"{path}: model_digest {rec.get('model_digest')!r} != "
                f"manifest model_digest {manifest.get('model_digest')!r}"
            )
        if rec.get("temperature") != manifest.get("temperature"):
            errors.append(
                f"{path}: temperature {rec.get('temperature')!r} != "
                f"manifest temperature {manifest.get('temperature')!r}"
            )
        if rec.get("num_ctx") != manifest.get("num_ctx"):
            errors.append(f"{path}: num_ctx {rec.get('num_ctx')!r} != manifest num_ctx {manifest.get('num_ctx')!r}")
        if rec.get("city") != city:
            errors.append(f"{path}: city {rec.get('city')!r} != manifest-implied {city!r}")
        if rec.get("condition") != condition:
            errors.append(f"{path}: condition {rec.get('condition')!r} != manifest-implied {condition!r}")
        if rec.get("replicate") != rep:
            errors.append(f"{path}: replicate {rec.get('replicate')!r} != manifest-implied {rep}")

    return errors, manifest


# ═══════════════════════════════════════════════════════════════════════
# Numeric claim extraction (see module docstring for tokenizer choice)
# ═══════════════════════════════════════════════════════════════════════

def section_number_sets(record: dict) -> dict[str, set[str]]:
    """{section_key: set(raw_number_token, ...)}, one token set per section,
    using e1_evidence_consistency.tokenize_numbers() (NUMBER_RE_NEW).
    Repeated occurrences within a section collapse via set()."""
    out = {}
    for key in SECTION_KEYS:
        text = record["prose"].get(key, "")
        out[key] = {n["raw"] for n in e1.tokenize_numbers(text)}
    return out


def whole_report_claims(record: dict) -> set[tuple[str, str]]:
    """(section_key, raw_number) pairs -- the exact claim-identity key
    evaluation/scripts/run_b1_claim_overlap.py uses, so a whole-report
    'claim' here means the same thing it means in the 71.6% figure."""
    claims: set[tuple[str, str]] = set()
    for key, toks in section_number_sets(record).items():
        for t in toks:
            claims.add((key, t))
    return claims


def overlap_stats(a: set, b: set) -> tuple[int, int, int, int, float | None]:
    """(n_only_a, n_only_b, n_both, n_union, overlap_pct). overlap_pct =
    100 * |both| / |union| -- the exact formula from
    evaluation/scripts/run_b1_claim_overlap.py, reproduced not reinvented."""
    only_a = a - b
    only_b = b - a
    both = a & b
    union = a | b
    pct = round(100 * len(both) / len(union), 1) if union else None
    return len(only_a), len(only_b), len(both), len(union), pct


def char_similarity(a: str, b: str) -> float:
    """difflib.SequenceMatcher(None, a, b).ratio() -- the same method
    runs/b1_diff.py uses for its per-section char_similarity_ratio,
    factored out here since b1_diff.py does not itself export it as a
    standalone function."""
    return round(difflib.SequenceMatcher(None, a, b).ratio(), 4)


def build_pairs(record: dict) -> list[tuple[str, str]]:
    """(section_title, sentence) pairs from replicate prose, using
    e1_evidence_consistency.split_sentences() -- the same sentence-splitter
    extract_pipeline_prose() calls internally on HTML-derived section text.
    Applied here directly to the replicate's own per-section prose (which is
    already correctly section-separated, since synthesize_all() returns one
    string per section) instead of re-deriving section boundaries from <h2>
    tags, which the JSON-only replicate has none of."""
    pairs = []
    for key, title in SECTIONS:
        text = record["prose"].get(key, "")
        for sent in e1.split_sentences(text):
            pairs.append((title, sent))
    return pairs


def whole_text(record: dict) -> str:
    return " ".join(record["prose"].get(key, "") for key in SECTION_KEYS)


# ═══════════════════════════════════════════════════════════════════════
# 2/3/4 — pairwise numeric stability, summaries, section-level
# ═══════════════════════════════════════════════════════════════════════

def replicate_pairs_for_city(records: dict, city: str):
    """Yields (comparison_type, rep_a, rep_b, record_a, record_b) for all
    3 Full-Full + 3 B1-B1 + 9 Full-B1 pairs for one city."""
    full_reps = [records[(city, "full", r)] for r in range(1, N_REPLICATES + 1)]
    b1_reps = [records[(city, "b1", r)] for r in range(1, N_REPLICATES + 1)]

    for (ra, rec_a), (rb, rec_b) in itertools.combinations(enumerate(full_reps, 1), 2):
        yield "full_full", ra, rb, rec_a, rec_b
    for (ra, rec_a), (rb, rec_b) in itertools.combinations(enumerate(b1_reps, 1), 2):
        yield "b1_b1", ra, rb, rec_a, rec_b
    for ra, rec_a in enumerate(full_reps, 1):
        for rb, rec_b in enumerate(b1_reps, 1):
            yield "full_b1", ra, rb, rec_a, rec_b


def compute_numeric_pairwise(records: dict) -> list[dict]:
    rows = []
    for city in CITIES:
        for comparison_type, ra, rb, rec_a, rec_b in replicate_pairs_for_city(records, city):
            # whole-report level
            claims_a = whole_report_claims(rec_a)
            claims_b = whole_report_claims(rec_b)
            n_only_a, n_only_b, n_both, n_union, pct = overlap_stats(claims_a, claims_b)
            rows.append({
                "city": city, "comparison_type": comparison_type,
                "rep_a": ra, "rep_b": rb, "level": "whole_report",
                "n_only_a": n_only_a, "n_only_b": n_only_b, "n_both": n_both,
                "n_union": n_union, "overlap_pct": pct,
            })
            # per-section level
            sets_a = section_number_sets(rec_a)
            sets_b = section_number_sets(rec_b)
            for key in SECTION_KEYS:
                n_only_a, n_only_b, n_both, n_union, pct = overlap_stats(sets_a[key], sets_b[key])
                rows.append({
                    "city": city, "comparison_type": comparison_type,
                    "rep_a": ra, "rep_b": rb, "level": f"section:{key}",
                    "n_only_a": n_only_a, "n_only_b": n_only_b, "n_both": n_both,
                    "n_union": n_union, "overlap_pct": pct,
                })
    return rows


def _dist_stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


def compute_numeric_summary(pairwise_rows: list[dict]) -> list[dict]:
    """Overall (n, mean, median, min, max) per comparison_type, pooled
    across cities, plus per-city mean/range -- whole_report level only.
    Also the descriptive contrast: mean(full_b1) - mean(pooled full_full +
    b1_b1 raw pairwise values combined into one list, per the instruction
    that pooled-within-condition COMBINES the pairwise values, not the
    average of two group means). Also carries the original 71.6% figure as
    a labelled reference row for direct visual comparison."""
    whole = [r for r in pairwise_rows if r["level"] == "whole_report" and r["overlap_pct"] is not None]
    rows = []

    for comparison_type in ("full_full", "b1_b1", "full_b1"):
        vals = [r["overlap_pct"] for r in whole if r["comparison_type"] == comparison_type]
        stats = _dist_stats(vals)
        rows.append({
            "scope": "pooled_all_cities", "comparison_type": comparison_type,
            "city": "", **stats,
        })
        for city in CITIES:
            cvals = [r["overlap_pct"] for r in whole
                     if r["comparison_type"] == comparison_type and r["city"] == city]
            cstats = _dist_stats(cvals)
            rows.append({
                "scope": "per_city", "comparison_type": comparison_type,
                "city": city, **cstats,
            })

    within_pooled = (
        [r["overlap_pct"] for r in whole if r["comparison_type"] == "full_full"] +
        [r["overlap_pct"] for r in whole if r["comparison_type"] == "b1_b1"]
    )
    between_vals = [r["overlap_pct"] for r in whole if r["comparison_type"] == "full_b1"]
    contrast = None
    if within_pooled and between_vals:
        contrast = round(statistics.mean(between_vals) - statistics.mean(within_pooled), 2)
    rows.append({
        "scope": "descriptive_contrast", "comparison_type": "full_b1_minus_pooled_within",
        "city": "", "n": len(between_vals) + len(within_pooled),
        "mean": contrast, "median": None, "min": None, "max": None,
    })

    rows.append({
        "scope": "original_one_shot_reference", "comparison_type": "pipeline_vs_b1_original_report",
        "city": "ALL_POOLED", "n": 1,
        "mean": ORIGINAL_ONE_SHOT_OVERLAP_PCT, "median": ORIGINAL_ONE_SHOT_OVERLAP_PCT,
        "min": ORIGINAL_ONE_SHOT_OVERLAP_PCT, "max": ORIGINAL_ONE_SHOT_OVERLAP_PCT,
    })
    return rows


def compute_section_summary(pairwise_rows: list[dict]) -> list[dict]:
    rows = []
    for key in SECTION_KEYS:
        level = f"section:{key}"
        for comparison_type in ("full_full", "b1_b1", "full_b1"):
            vals = [r["overlap_pct"] for r in pairwise_rows
                    if r["level"] == level and r["comparison_type"] == comparison_type
                    and r["overlap_pct"] is not None]
            stats = _dist_stats(vals)
            rows.append({
                "section_key": key, "section_title": SECTION_TITLES[key],
                "comparison_type": comparison_type, **stats,
            })
    return rows


# ═══════════════════════════════════════════════════════════════════════
# 5 — digital same-prompt control
# ═══════════════════════════════════════════════════════════════════════

def compute_digital_control(records: dict) -> tuple[list[dict], list[dict]]:
    """Returns (pairwise_rows, summary_rows). The six digital-section
    outputs per city (3 full + 3 b1) share an identical effective prompt
    (instruction='' in both conditions -- see text_synthesizer.py's digital
    section and B1_report.md's 'Determinism check'), so all 15 pairwise
    comparisons among them are a same-prompt, temperature=0 reproducibility
    estimate, not an ablation comparison."""
    pairwise_rows = []
    summary_rows = []

    for city in CITIES:
        labels_and_text = []
        for condition in CONDITIONS:
            for rep in range(1, N_REPLICATES + 1):
                rec = records[(city, condition, rep)]
                labels_and_text.append((f"{condition}_rep{rep}", rec["prose"]["digital"]))

        identical_flags = []
        overlaps = []
        similarities = []
        for (label_a, text_a), (label_b, text_b) in itertools.combinations(labels_and_text, 2):
            nums_a = {n["raw"] for n in e1.tokenize_numbers(text_a)}
            nums_b = {n["raw"] for n in e1.tokenize_numbers(text_b)}
            _, _, _, _, pct = overlap_stats(nums_a, nums_b)
            sim = char_similarity(text_a, text_b)
            identical = text_a.strip() == text_b.strip()

            identical_flags.append(identical)
            if pct is not None:
                overlaps.append(pct)
            similarities.append(sim)

            pairwise_rows.append({
                "city": city, "output_a": label_a, "output_b": label_b,
                "identical_text": identical, "numeric_overlap_pct": pct,
                "char_similarity": sim,
            })

        unique_texts = {t.strip() for _, t in labels_and_text}
        # frozenset of raw tokens per output, deduplicated across the six outputs
        unique_number_sets = {frozenset(n["raw"] for n in e1.tokenize_numbers(t)) for _, t in labels_and_text}

        summary_rows.append({
            "city": city,
            "n_pairs": len(identical_flags),
            "exact_identity_rate": round(sum(identical_flags) / len(identical_flags), 4) if identical_flags else None,
            "numeric_overlap_mean": round(statistics.mean(overlaps), 2) if overlaps else None,
            "numeric_overlap_median": round(statistics.median(overlaps), 2) if overlaps else None,
            "numeric_overlap_min": round(min(overlaps), 2) if overlaps else None,
            "numeric_overlap_max": round(max(overlaps), 2) if overlaps else None,
            "char_similarity_mean": round(statistics.mean(similarities), 4) if similarities else None,
            "char_similarity_median": round(statistics.median(similarities), 4) if similarities else None,
            "char_similarity_min": round(min(similarities), 4) if similarities else None,
            "char_similarity_max": round(max(similarities), 4) if similarities else None,
            "n_unique_texts_among_6": len(unique_texts),
            "all_6_identical_numeric_claim_sets": len(unique_number_sets) == 1,
        })

    return pairwise_rows, summary_rows


# ═══════════════════════════════════════════════════════════════════════
# 6 — text similarity (Full-Full / B1-B1 / Full-B1, report + section level)
# ═══════════════════════════════════════════════════════════════════════

def compute_text_similarity(records: dict) -> list[dict]:
    rows = []
    for city in CITIES:
        for comparison_type, ra, rb, rec_a, rec_b in replicate_pairs_for_city(records, city):
            sim = char_similarity(whole_text(rec_a), whole_text(rec_b))
            rows.append({
                "city": city, "comparison_type": comparison_type,
                "rep_a": ra, "rep_b": rb, "level": "whole_report",
                "char_similarity": sim,
            })
            for key in SECTION_KEYS:
                sim = char_similarity(rec_a["prose"].get(key, ""), rec_b["prose"].get(key, ""))
                rows.append({
                    "city": city, "comparison_type": comparison_type,
                    "rep_a": ra, "rep_b": rb, "level": f"section:{key}",
                    "char_similarity": sim,
                })
    return rows


# ═══════════════════════════════════════════════════════════════════════
# 7 — claim presence stability
# ═══════════════════════════════════════════════════════════════════════

def compute_claim_stability(records: dict) -> list[dict]:
    rows = []
    for city in CITIES:
        per_condition_claims: dict[str, list[set]] = {}
        for condition in CONDITIONS:
            per_condition_claims[condition] = [
                whole_report_claims(records[(city, condition, r)]) for r in range(1, N_REPLICATES + 1)
            ]

        stable_full = set.intersection(*per_condition_claims["full"])
        stable_b1 = set.intersection(*per_condition_claims["b1"])
        any_full = set.union(*per_condition_claims["full"])
        any_b1 = set.union(*per_condition_claims["b1"])

        for condition in CONDITIONS:
            sets3 = per_condition_claims[condition]
            all_claims = set.union(*sets3)
            presence_count = Counter()
            for claim in all_claims:
                presence_count[claim] = sum(1 for s in sets3 if claim in s)

            for (section_key, raw_number), count in presence_count.items():
                if count == 3:
                    presence_category = "in_all_3"
                elif count == 2:
                    presence_category = "in_exactly_2_of_3"
                else:
                    presence_category = "in_exactly_1_of_3"

                cross_condition = "n/a"
                if condition == "full" and (section_key, raw_number) in stable_full:
                    if (section_key, raw_number) not in any_b1:
                        cross_condition = "stable_full_absent_from_all_b1"
                    elif (section_key, raw_number) in stable_b1:
                        cross_condition = "stable_in_both"
                if condition == "b1" and (section_key, raw_number) in stable_b1:
                    if (section_key, raw_number) not in any_full:
                        cross_condition = "stable_b1_absent_from_all_full"
                    elif (section_key, raw_number) in stable_full:
                        cross_condition = "stable_in_both"

                rows.append({
                    "city": city, "condition": condition,
                    "section_key": section_key, "section_title": SECTION_TITLES[section_key],
                    "raw_number": raw_number, "presence_count": count,
                    "presence_category": presence_category,
                    "cross_condition_note": cross_condition,
                })
    return rows


# ═══════════════════════════════════════════════════════════════════════
# 9 — E1 replication (AUDITABLE_DIRECT / DERIVED / NOT_MACHINE_AUDITABLE)
# ═══════════════════════════════════════════════════════════════════════

def build_city_evidence(city: str):
    """Reused unmodified from e1_evidence_consistency.py's own run_city():
    load fsi_enriched.jsonl (read-only) -> extract_facts() -> flatten ->
    EvidenceIndex, plus known_district_names() for the same digit-exclusion
    tok() uses in Part 1/Part 2. Built once per city, shared by all 6
    replicates of that city (Full and B1 draw on the identical facts base --
    only graph_answers differs between conditions, per text_synthesizer.py)."""
    pipeline_records = e1.load_pipeline_records(city)
    facts = e1.extract_facts(pipeline_records)
    flat, groups = e1._flatten_facts(facts)
    index = e1.EvidenceIndex(flat, groups, totals=[float(facts["total"])])
    known_names = e1.known_district_names(city)
    return index, known_names


def compute_e1_replication(records: dict) -> list[dict]:
    rows = []
    for city in CITIES:
        index, known_names = build_city_evidence(city)

        def tok(sentence: str):
            return e1.tokenize_numbers_excluding_names(sentence, known_names) if known_names \
                else e1.tokenize_numbers(sentence)

        for condition in CONDITIONS:
            for rep in range(1, N_REPLICATES + 1):
                rec = records[(city, condition, rep)]
                pairs = build_pairs(rec)
                n_direct = n_derived = n_not = 0
                for _heading, sentence in pairs:
                    for num in tok(sentence):
                        label, _justification = index.label(num["value"], num["is_percentage"])
                        n_direct += label == "AUDITABLE_DIRECT"
                        n_derived += label == "AUDITABLE_DERIVED"
                        n_not += label == "NOT_MACHINE_AUDITABLE"
                n_claims = n_direct + n_derived + n_not
                rows.append({
                    "city": city, "condition": condition, "replicate": rep,
                    "n_claims": n_claims, "n_auditable_direct": n_direct,
                    "n_auditable_derived": n_derived, "n_not_machine_auditable": n_not,
                    "auditability_rate": round((n_direct + n_derived) / n_claims, 4) if n_claims else None,
                })
    return rows


# ═══════════════════════════════════════════════════════════════════════
# 10 — E2 candidate flags (CS-1, CS-2, CS-4, CS-5) -- UNVERIFIED
# ═══════════════════════════════════════════════════════════════════════

def compute_e2_candidate_flags(records: dict) -> list[dict]:
    """Reuses e2_internal_consistency.run_report() unmodified -- the exact
    same function the existing E2 instrument calls per (city, system).
    'system' here is f'{condition}_rep{rep}' so flags stay traceable to
    their exact replicate without touching run_report()'s signature. Output
    rows are explicitly labelled CANDIDATE -- verification status is left
    blank for manual review, and no VERDICT from the original e2_flags.csv
    is reused or implied for this new text."""
    rows = []
    for city in CITIES:
        for condition in CONDITIONS:
            for rep in range(1, N_REPLICATES + 1):
                rec = records[(city, condition, rep)]
                pairs = build_pairs(rec)
                full = whole_text(rec)
                total = e1.extract_stated_total(full, "pipeline")

                flags_out: list[dict] = []
                system_label = f"{condition}_rep{rep}"
                e2.run_report(city, system_label, pairs, total, flags_out)

                for f in flags_out:
                    rows.append({
                        "city": city, "condition": condition, "replicate": rep,
                        "section": f["section_heading"], "check_type": f["check"],
                        "matched_sentence": f["sentence"], "detail": f["detail"],
                        "label": "CANDIDATE", "verification_status": "UNVERIFIED",
                    })
    return rows


# ═══════════════════════════════════════════════════════════════════════
# CSV writing
# ═══════════════════════════════════════════════════════════════════════

def write_csv(name: str, rows: list[dict], fieldnames: list[str], overwrite: bool):
    path = _assert_safe_output_path(OUTPUT_ROOT / name)
    if path.exists() and not overwrite:
        raise RuntimeError(
            f"REFUSING TO OVERWRITE existing analysis output (pass --overwrite to replace): {path}"
        )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def write_markdown(name: str, text: str, overwrite: bool):
    path = _assert_safe_output_path(OUTPUT_ROOT / name)
    if path.exists() and not overwrite:
        raise RuntimeError(
            f"REFUSING TO OVERWRITE existing analysis output (pass --overwrite to replace): {path}"
        )
    path.write_text(text, encoding="utf-8")
    return path


# ═══════════════════════════════════════════════════════════════════════
# analysis_summary.md
# ═══════════════════════════════════════════════════════════════════════

def build_summary_markdown(
    validation_errors: list[str],
    numeric_summary: list[dict],
    section_summary: list[dict],
    digital_summary: list[dict],
    claim_stability: list[dict],
    e1_rows: list[dict],
    e2_rows: list[dict],
) -> str:
    md = []
    md.append("# Phase-5 Full/B1 Replication -- Stability Analysis\n")
    md.append(
        "Descriptive replication/stability analysis, NOT a significance test. "
        "3 new Full and 3 new B1 Phase-5 prose generations per city, generated "
        "by `evaluation/run_b1_replication.py`, analysed by "
        "`evaluation/analyze_b1_replication.py`. The original one-shot Full/B1 "
        "reports were NOT included in this replicate set.\n"
    )

    md.append("## DATA INTEGRITY\n")
    if validation_errors:
        md.append("**VALIDATION FAILED -- analysis below, if present, was computed "
                   "despite the following issues; treat with caution:**\n")
        for e in validation_errors:
            md.append(f"- {e}")
        md.append("")
    else:
        md.append(
            "All 30 replicate files present (5 cities x 2 conditions x 3 replicates), "
            "each with exactly the 6 expected prose sections, consistent "
            "city/condition/replicate metadata, and identical "
            "model/model_digest/temperature/num_ctx recorded across all 30 files.\n"
        )

    md.append("## SAME-PROMPT NONDETERMINISM\n")
    md.append(
        "The `digital` section has an empty GraphRAG instruction in both Full and "
        "B1 (see `src/phase_5/text_synthesizer.py`'s digital section, hardcoded "
        "`instruction=\"\"`), so its prompt is identical across all 6 outputs per "
        "city (3 Full + 3 B1) -- this is a same-prompt, temperature=0 "
        "reproducibility estimate, not an ablation effect. See `digital_control.csv` "
        "for full pairwise detail (15 pairs/city).\n"
    )
    for row in digital_summary:
        md.append(
            f"- **{row['city']}**: exact-identity rate "
            f"{row['exact_identity_rate']*100:.1f}% of 15 pairs "
            f"({row['n_unique_texts_among_6']} unique texts among 6 outputs); "
            f"numeric overlap {row['numeric_overlap_mean']}% mean "
            f"(range {row['numeric_overlap_min']}-{row['numeric_overlap_max']}%); "
            f"char similarity {row['char_similarity_mean']} mean "
            f"(range {row['char_similarity_min']}-{row['char_similarity_max']}); "
            f"all 6 identical numeric-claim sets: {row['all_6_identical_numeric_claim_sets']}"
            if row["exact_identity_rate"] is not None else f"- **{row['city']}**: no pairs computed"
        )
    md.append("")

    md.append("## WITHIN-CONDITION STABILITY\n")
    for comparison_type, label in (("full_full", "Full-vs-Full"), ("b1_b1", "B1-vs-B1")):
        pooled = next((r for r in numeric_summary
                        if r["scope"] == "pooled_all_cities" and r["comparison_type"] == comparison_type), None)
        if pooled:
            md.append(
                f"- **{label}** (n={pooled['n']} pairs pooled across cities): "
                f"mean {pooled['mean']}%, median {pooled['median']}%, "
                f"range {pooled['min']}-{pooled['max']}%."
            )
    md.append("")

    md.append("## FULL-VS-B1 STABILITY\n")
    pooled_fb1 = next((r for r in numeric_summary
                        if r["scope"] == "pooled_all_cities" and r["comparison_type"] == "full_b1"), None)
    if pooled_fb1:
        md.append(
            f"- **Full-vs-B1** (n={pooled_fb1['n']} pairs pooled across cities): "
            f"mean {pooled_fb1['mean']}%, median {pooled_fb1['median']}%, "
            f"range {pooled_fb1['min']}-{pooled_fb1['max']}%."
        )
    contrast = next((r for r in numeric_summary if r["scope"] == "descriptive_contrast"), None)
    if contrast:
        md.append(
            f"\n**Descriptive contrast** (mean Full-vs-B1 overlap minus pooled "
            f"within-condition overlap, where pooled-within combines every "
            f"Full-Full and B1-B1 pairwise value into one list before "
            f"averaging): **{contrast['mean']} percentage points**.\n\n"
            f"A substantially negative value (between-condition overlap much lower "
            f"than within-condition overlap) would support a systematic "
            f"GraphRAG-context effect distinguishable from ordinary generation "
            f"variability. A value close to zero, or overlapping the within-condition "
            f"range, means the original one-shot Full/B1 difference is not "
            f"distinguishable from ordinary Phase-5 run-to-run variability on this "
            f"evidence alone. No arbitrary significance threshold is applied to this "
            f"number -- read it against the ranges above, not against a cutoff.\n"
        )

    md.append("## SECTION-LEVEL RESULTS\n")
    md.append("| section | comparison | n | mean | median | min | max |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    for row in section_summary:
        md.append(
            f"| {row['section_title']} | {row['comparison_type']} | {row['n']} | "
            f"{row['mean']} | {row['median']} | {row['min']} | {row['max']} |"
        )
    md.append("")

    md.append("## STABLE CONDITION-SENSITIVE CLAIMS\n")
    stable_full_only = [r for r in claim_stability if r["cross_condition_note"] == "stable_full_absent_from_all_b1"]
    stable_b1_only = [r for r in claim_stability if r["cross_condition_note"] == "stable_b1_absent_from_all_full"]
    stable_both = [r for r in claim_stability if r["cross_condition_note"] == "stable_in_both"]
    md.append(
        f"Claims stable (3/3 replicates) in Full but absent from all 3 B1 replicates: "
        f"**{len(stable_full_only)}**. Stable in B1 but absent from all 3 Full replicates: "
        f"**{len(stable_b1_only)}**. Stable in both conditions: **{len(stable_both)}**. "
        f"Full detail (city, section, raw_number) in `claim_stability.csv`. These are "
        f"the claims most useful for distinguishing a systematic, condition-linked "
        f"effect from a one-off stochastic claim: a claim appearing in all 3 replicates "
        f"of one condition and none of the other is a stronger signal than a claim "
        f"appearing once in each.\n"
    )

    md.append("## ORIGINAL 71.6% RESULT IN CONTEXT\n")
    orig_row = next((r for r in numeric_summary if r["scope"] == "original_one_shot_reference"), None)
    ff = next((r for r in numeric_summary if r["scope"] == "pooled_all_cities" and r["comparison_type"] == "full_full"), None)
    bb = next((r for r in numeric_summary if r["scope"] == "pooled_all_cities" and r["comparison_type"] == "b1_b1"), None)
    fb = next((r for r in numeric_summary if r["scope"] == "pooled_all_cities" and r["comparison_type"] == "full_b1"), None)
    md.append(
        f"Original one-shot pooled overlap (read verbatim from "
        f"`{ORIGINAL_ONE_SHOT_SOURCE}`, not recomputed): **{ORIGINAL_ONE_SHOT_OVERLAP_PCT}%**.\n\n"
        f"Observed new-replicate ranges (pooled across cities, whole-report level): "
        f"Full-vs-Full {ff['min'] if ff else '?'}-{ff['max'] if ff else '?'}% "
        f"(mean {ff['mean'] if ff else '?'}%); "
        f"B1-vs-B1 {bb['min'] if bb else '?'}-{bb['max'] if bb else '?'}% "
        f"(mean {bb['mean'] if bb else '?'}%); "
        f"Full-vs-B1 {fb['min'] if fb else '?'}-{fb['max'] if fb else '?'}% "
        f"(mean {fb['mean'] if fb else '?'}%).\n\n"
        f"Where 71.6% falls relative to these three ranges (inside the "
        f"within-condition range, inside the new Full-vs-B1 range, or outside all "
        f"three) should be read directly off the numbers above once computed -- "
        f"no threshold is imposed here.\n"
    )

    md.append("## E1 REPLICATION\n")
    if e1_rows:
        md.append(
            "E1's full AUDITABLE_DIRECT/AUDITABLE_DERIVED/NOT_MACHINE_AUDITABLE "
            "taxonomy was reproduced by importing "
            "`evaluation.e1_evidence_consistency`'s own `tokenize_numbers`, "
            "`tokenize_numbers_excluding_names`, `split_sentences`, `_flatten_facts`, "
            "`EvidenceIndex`, and `known_district_names` unmodified -- the only "
            "difference from the original instrument is how (section, sentence) "
            "pairs are built (from the replicate's own per-section prose dict via "
            "`split_sentences()`, instead of from `<h2>`/`<p>` HTML tags via "
            "`extract_pipeline_prose()`, since replicate JSONs have no HTML "
            "structure to parse). Per-replicate results are in `e1_replication.csv`.\n"
        )
        rates = [r["auditability_rate"] for r in e1_rows if r["auditability_rate"] is not None]
        if rates:
            md.append(
                f"Auditability rate across all 30 replicates: mean "
                f"{round(statistics.mean(rates), 4)}, range "
                f"{round(min(rates), 4)}-{round(max(rates), 4)}.\n"
            )
    else:
        md.append(
            "E1 replication not computed because exact reuse of the existing "
            "instrument was not possible without changing the instrument.\n"
        )

    md.append("## E2 CANDIDATE FLAGS\n")
    if e2_rows:
        md.append(
            f"E2's four automated checks (CS-1, CS-2, CS-4, CS-5) were reused "
            f"unmodified via `e2_internal_consistency.run_report()`, called once per "
            f"replicate. **{len(e2_rows)} candidate flags** raised across all 30 "
            f"replicates -- **every flag is UNVERIFIED**; none of the original "
            f"`e2_flags.csv` VERDICT labels are reused or implied for this new text. "
            f"Full detail in `e2_candidate_flags.csv`; verify manually (VERDICT "
            f"TP/FP) before citing any rate.\n"
        )
        by_check = Counter(r["check_type"] for r in e2_rows)
        for check, n in sorted(by_check.items()):
            md.append(f"- {check}: {n} candidate flags")
        md.append("")
    else:
        md.append(
            "E2 replication not computed because exact reuse of the existing "
            "instrument was not possible without changing the instrument.\n"
        )

    md.append("## INTERPRETATION LIMITS\n")
    md.append(
        "- n=3 per condition per city is a **stability/replication analysis**, "
        "not a powered significance test.\n"
        "- **No p-values or causal-effect estimates are justified by this data** "
        "and none are computed anywhere in this script.\n"
        "- A lower Full-vs-B1 overlap than Full-vs-Full/B1-vs-B1 overlap (see the "
        "descriptive contrast above) would be evidence **consistent with** a "
        "systematic GraphRAG-context effect -- it does not, on its own, prove one.\n"
        "- Overlap of the three distributions would mean ordinary stochastic "
        "Phase-5 variation remains a plausible explanation for at least some of "
        "the differences observed in the original one-shot Full/B1 comparison.\n"
    )

    return "\n".join(md)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def _print_dry_run(records: dict, discover_errors: list[str],
                    manifest: dict | None, manifest_errors: list[str]):
    total_expected = len(CITIES) * len(CONDITIONS) * N_REPLICATES
    expected_section_keys = set(SECTION_KEYS)
    n_sections_ok = sum(
        1 for r in records.values()
        if set(r.get("prose", {}).keys()) == expected_section_keys
    )

    print("=" * 70)
    print("DRY RUN — no files will be written")
    print("=" * 70)
    print(f"\nINPUT_ROOT  = {INPUT_ROOT}")
    print(f"OUTPUT_ROOT = {OUTPUT_ROOT}")

    print("\n--- Manifest integrity status ---")
    if manifest is None:
        print(f"  manifest NOT LOADED ({MANIFEST_PATH})")
    else:
        print(f"  integrity_check: {manifest.get('integrity_check')!r}")
        print(f"  integrity_changed_files: {manifest.get('integrity_changed_files')!r}")
        print("\n--- Model / digest / config (from manifest) ---")
        print(f"  model: {manifest.get('model')!r}")
        print(f"  model_digest: {manifest.get('model_digest')!r}")
        print(f"  temperature: {manifest.get('temperature')!r}")
        print(f"  num_ctx: {manifest.get('num_ctx')!r}")
        print(f"  n_replicates_per_condition: {manifest.get('n_replicates_per_condition')!r}")
        print(f"  conditions: {manifest.get('conditions')!r}")
        print(f"  cities: {manifest.get('cities')!r}")

    print(f"\n{len(records)}/{total_expected} replicate files validated")
    print(f"{n_sections_ok}/{total_expected} replicates with 6/6 expected sections")

    all_errors = discover_errors + manifest_errors
    if all_errors:
        print("\nVALIDATION ERRORS:")
        for e in all_errors:
            print(f"  - {e}")
    else:
        print("\nValidation: OK (replicate files + manifest)")

    print("\nOutputs that would be created:")
    for name in sorted(ALLOWED_OUTPUT_NAMES):
        print(f"  {OUTPUT_ROOT / name}")

    print("\nNo output has been written. Dry run complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Replication/stability analysis of the Phase-5 Full/B1 replication experiment."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs and print the plan; write nothing.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow overwriting existing analysis outputs. Off by default.")
    args = parser.parse_args()

    records, discover_errors = discover_and_validate()
    manifest_errors, manifest = validate_manifest(records)
    errors = discover_errors + manifest_errors

    if args.dry_run:
        _print_dry_run(records, discover_errors, manifest, manifest_errors)
        return

    if errors:
        print("VALIDATION FAILED -- aborting before any analysis:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    numeric_pairwise = compute_numeric_pairwise(records)
    numeric_summary = compute_numeric_summary(numeric_pairwise)
    section_summary = compute_section_summary(numeric_pairwise)
    digital_pairwise, digital_summary = compute_digital_control(records)
    text_sim_rows = compute_text_similarity(records)
    claim_stability_rows = compute_claim_stability(records)
    e1_rows = compute_e1_replication(records)
    e2_rows = compute_e2_candidate_flags(records)

    write_csv("numeric_pairwise.csv", numeric_pairwise,
              ["city", "comparison_type", "rep_a", "rep_b", "level",
               "n_only_a", "n_only_b", "n_both", "n_union", "overlap_pct"], args.overwrite)
    write_csv("numeric_summary.csv", numeric_summary,
              ["scope", "comparison_type", "city", "n", "mean", "median", "min", "max"], args.overwrite)
    write_csv("section_summary.csv", section_summary,
              ["section_key", "section_title", "comparison_type", "n", "mean", "median", "min", "max"], args.overwrite)
    write_csv("digital_control.csv", digital_summary,
              ["city", "n_pairs", "exact_identity_rate", "numeric_overlap_mean",
               "numeric_overlap_median", "numeric_overlap_min", "numeric_overlap_max",
               "char_similarity_mean", "char_similarity_median", "char_similarity_min",
               "char_similarity_max", "n_unique_texts_among_6", "all_6_identical_numeric_claim_sets"],
              args.overwrite)
    write_csv("claim_stability.csv", claim_stability_rows,
              ["city", "condition", "section_key", "section_title", "raw_number",
               "presence_count", "presence_category", "cross_condition_note"], args.overwrite)
    write_csv("text_similarity.csv", text_sim_rows,
              ["city", "comparison_type", "rep_a", "rep_b", "level", "char_similarity"], args.overwrite)
    write_csv("e1_replication.csv", e1_rows,
              ["city", "condition", "replicate", "n_claims", "n_auditable_direct",
               "n_auditable_derived", "n_not_machine_auditable", "auditability_rate"], args.overwrite)
    write_csv("e2_candidate_flags.csv", e2_rows,
              ["city", "condition", "replicate", "section", "check_type",
               "matched_sentence", "detail", "label", "verification_status"], args.overwrite)

    summary_md = build_summary_markdown(
        errors, numeric_summary, section_summary, digital_summary,
        claim_stability_rows, e1_rows, e2_rows,
    )
    write_markdown("analysis_summary.md", summary_md, args.overwrite)

    print(f"Analysis complete. Outputs written under {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
