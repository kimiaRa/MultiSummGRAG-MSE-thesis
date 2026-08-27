"""
Phase-2 classification human-validation analysis — additive, read-only.

Analyses the completed, frozen human-annotation study
(evaluation/results/phase2_validation/sample_blinded_ANNOTATED_FROZEN.csv)
against the hidden key (sample_key.csv) and the generation manifest
(sample_manifest.json). Descriptive master's-thesis validation of the
Phase-2 fsi_type / operational_level labels -- NOT a powered population
accuracy study (see analysis_summary.md's INTERPRETATION LIMITS, always
written).

READ-ONLY against:
  - evaluation/results/phase2_validation/sample_blinded_ANNOTATED_FROZEN.csv
  - evaluation/results/phase2_validation/sample_key.csv
  - evaluation/results/phase2_validation/sample_manifest.json
  Joined ONLY by sample_id. None of the three is ever opened in a write
  mode anywhere in this script -- see _assert_safe_output_path(), called
  before every write, which additionally rejects any path under
  evaluation/results/phase2_validation/ that is not inside its own
  analysis/ subdirectory (so the frozen CSV, the key, and the manifest
  cannot be targeted even by a path-construction mistake).

WRITES ONLY under evaluation/results/phase2_validation/analysis/. Never
imports ollama, GraphRAG, or anything from src/phase_0-6 -- there is no
code path here that could rerun Phase 2 or any other pipeline stage, and no
network call is made anywhere in this file.

── CONSTRUCT RULE (see module docstring of the task, and INTERPRETATION
   LIMITS in the generated summary) ─────────────────────────────────────
Phase 2 has no model output equivalent to genuine_fsi / not_an_fsi /
insufficient_evidence. This script therefore never computes FSI-status
accuracy, precision/recall/F1, a status confusion matrix, or a status
kappa -- human_fsi_status is used only to (a) describe what the CORE sample
actually contains and (b) determine which rows are evaluable for
fsi_type/operational_level agreement (only human_fsi_status == genuine_fsi
rows ever enter the type/op agreement calculations). not_an_fsi and
insufficient_evidence rows are reported as separate DIAGNOSTIC label
distributions (§7/§8 of the task), never as classification errors.

CORE (50, 10/city, uniformly random, city-balanced) is the primary
validation sample. STRESS (25, 5/city, deliberately edge-case) is analysed
separately and is never pooled with CORE into one accuracy/agreement
number, anywhere in this script.

Usage:
    python evaluation/analyze_phase2_validation.py --dry-run
    python evaluation/analyze_phase2_validation.py
    python evaluation/analyze_phase2_validation.py --overwrite
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VALIDATION_DIR = (ROOT / "evaluation" / "results" / "phase2_validation").resolve()
OUTPUT_ROOT = (VALIDATION_DIR / "analysis").resolve()
DATA_ROOT = (ROOT / "data").resolve()

FROZEN_PATH = VALIDATION_DIR / "sample_blinded_ANNOTATED_FROZEN.csv"
KEY_PATH = VALIDATION_DIR / "sample_key.csv"
MANIFEST_PATH = VALIDATION_DIR / "sample_manifest.json"

CITIES = ["barcelona", "brighton", "dublin", "london", "milan"]
FSI_STATUSES = ["genuine_fsi", "not_an_fsi", "insufficient_evidence"]
CONFIDENCE_LEVELS = ["high", "medium", "low"]
N_CORE_PER_CITY = 10
N_STRESS_PER_CITY = 5
EXPECTED_TOTAL_RECORDS = 635
EXPECTED_SEED = 42

# The Phase-2 classifier's real enums (see src/phase_2/classifier.py's
# SYSTEM_PROMPT) -- "unknown" is a separate sentinel, not a ninth/sixth
# class, for both fields.
FSI_TYPE_ENUM = {
    "food_sharing", "food_swapping", "food_gifting", "community_garden",
    "food_bank", "meals_service", "food_education", "other",
}
OP_LEVEL_ENUM = {
    "government_funded", "council_supported", "ngo_led", "community_led", "commercial",
}
FSI_TYPE_ALLOWED = FSI_TYPE_ENUM | {"unknown"}
OP_LEVEL_ALLOWED = OP_LEVEL_ENUM | {"unknown"}

ALLOWED_OUTPUT_NAMES = {
    "core_summary.csv", "core_type_confusion.csv", "core_operational_confusion.csv",
    "core_disagreements.csv", "core_status_diagnostics.csv", "core_confidence_summary.csv",
    "stress_summary.csv", "stress_diagnostics.csv", "sentinel_result.csv", "analysis_summary.md",
}

# ── Module-load-time safety assertions ─────────────────────────────────────
assert DATA_ROOT not in OUTPUT_ROOT.parents and OUTPUT_ROOT != DATA_ROOT, \
    "OUTPUT_ROOT must not resolve inside data/"
assert OUTPUT_ROOT != VALIDATION_DIR and VALIDATION_DIR in OUTPUT_ROOT.parents, \
    "OUTPUT_ROOT must be the analysis/ subdirectory of phase2_validation/, not that directory itself"


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
# Loading (read-only)
# ═══════════════════════════════════════════════════════════════════════

def load_frozen() -> list[dict]:
    with open(FROZEN_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_key() -> list[dict]:
    with open(KEY_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["source_record_index"] = int(r["source_record_index"])
    return rows


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


# ═══════════════════════════════════════════════════════════════════════
# §1 — Validation (abort before analysis if anything fails)
# ═══════════════════════════════════════════════════════════════════════

def validate(frozen_rows: list[dict] | None, key_rows: list[dict] | None,
             manifest: dict | None) -> list[str]:
    errors: list[str] = []

    if not FROZEN_PATH.exists():
        errors.append(f"MISSING frozen annotation file: {FROZEN_PATH}")
    if not KEY_PATH.exists():
        errors.append(f"MISSING key file: {KEY_PATH}")
    if not MANIFEST_PATH.exists():
        errors.append(f"MISSING manifest file: {MANIFEST_PATH}")
    if errors:
        return errors  # can't proceed to row-level checks without all three

    frozen_rows = frozen_rows or []
    key_rows = key_rows or []

    if len(frozen_rows) != 75:
        errors.append(f"frozen file has {len(frozen_rows)} rows, expected 75")
    if len(key_rows) != 75:
        errors.append(f"key file has {len(key_rows)} rows, expected 75")

    frozen_ids = [r["sample_id"] for r in frozen_rows]
    key_ids = [r["sample_id"] for r in key_rows]
    if len(set(frozen_ids)) != len(frozen_ids):
        dupes = [i for i, c in Counter(frozen_ids).items() if c > 1]
        errors.append(f"duplicate sample_id in frozen file: {dupes}")
    if len(set(key_ids)) != len(key_ids):
        dupes = [i for i, c in Counter(key_ids).items() if c > 1]
        errors.append(f"duplicate sample_id in key file: {dupes}")
    if set(frozen_ids) != set(key_ids):
        errors.append(
            f"sample_id sets differ between frozen and key: "
            f"frozen-only={sorted(set(frozen_ids) - set(key_ids))}, "
            f"key-only={sorted(set(key_ids) - set(frozen_ids))}"
        )

    # Group/city consistency between the two files (join sanity, beyond the
    # literal checklist -- a silent group/city mismatch would corrupt every
    # downstream breakdown, so it is caught here rather than assumed).
    key_by_id = {r["sample_id"]: r for r in key_rows}
    for r in frozen_rows:
        k = key_by_id.get(r["sample_id"])
        if k is None:
            continue
        if r.get("sample_group") != k.get("sample_group"):
            errors.append(f"{r['sample_id']}: sample_group differs frozen={r.get('sample_group')!r} vs key={k.get('sample_group')!r}")
        if r.get("city") != k.get("city"):
            errors.append(f"{r['sample_id']}: city differs frozen={r.get('city')!r} vs key={k.get('city')!r}")

    core_count = sum(1 for r in key_rows if r.get("sample_group") == "core")
    stress_count = sum(1 for r in key_rows if r.get("sample_group") == "stress")
    if core_count != 50:
        errors.append(f"expected 50 core rows, key file has {core_count}")
    if stress_count != 25:
        errors.append(f"expected 25 stress rows, key file has {stress_count}")

    for city in CITIES:
        c_core = sum(1 for r in key_rows if r.get("city") == city and r.get("sample_group") == "core")
        c_stress = sum(1 for r in key_rows if r.get("city") == city and r.get("sample_group") == "stress")
        if c_core != N_CORE_PER_CITY:
            errors.append(f"{city}: expected {N_CORE_PER_CITY} core rows, found {c_core}")
        if c_stress != N_STRESS_PER_CITY:
            errors.append(f"{city}: expected {N_STRESS_PER_CITY} stress rows, found {c_stress}")

    for r in frozen_rows:
        sid = r.get("sample_id", "?")
        status = (r.get("human_fsi_status") or "").strip()
        confidence = (r.get("human_confidence") or "").strip()
        ftype = (r.get("human_fsi_type") or "").strip()
        oplevel = (r.get("human_operational_level") or "").strip()

        if not status:
            errors.append(f"{sid}: missing human_fsi_status")
        elif status not in FSI_STATUSES:
            errors.append(f"{sid}: human_fsi_status {status!r} not one of {FSI_STATUSES}")

        if not confidence:
            errors.append(f"{sid}: missing human_confidence")

        if status == "genuine_fsi":
            if not ftype:
                errors.append(f"{sid}: human_fsi_status=genuine_fsi but human_fsi_type is blank")
            if not oplevel:
                errors.append(f"{sid}: human_fsi_status=genuine_fsi but human_operational_level is blank")
        elif status in ("not_an_fsi", "insufficient_evidence"):
            if ftype:
                errors.append(f"{sid}: human_fsi_status={status} but human_fsi_type is non-blank ({ftype!r})")
            if oplevel:
                errors.append(f"{sid}: human_fsi_status={status} but human_operational_level is non-blank ({oplevel!r})")

    # Model labels must come only from sample_key.csv -- the frozen file
    # must carry no model-label column at all.
    if frozen_rows:
        frozen_fields = set(frozen_rows[0].keys())
        forbidden = {"model_fsi_type", "model_operational_level", "fsi_type", "operational_level"}
        present = forbidden & frozen_fields
        if present:
            errors.append(f"frozen file contains forbidden model-label column(s): {sorted(present)}")

    if manifest is not None:
        if manifest.get("actual_total_records") != EXPECTED_TOTAL_RECORDS:
            errors.append(
                f"manifest actual_total_records {manifest.get('actual_total_records')!r} "
                f"!= expected {EXPECTED_TOTAL_RECORDS}"
            )
        if manifest.get("random_seed") != EXPECTED_SEED:
            errors.append(f"manifest random_seed {manifest.get('random_seed')!r} != expected {EXPECTED_SEED}")

    return errors


# ═══════════════════════════════════════════════════════════════════════
# Join
# ═══════════════════════════════════════════════════════════════════════

def build_joined(frozen_rows: list[dict], key_rows: list[dict]) -> list[dict]:
    """Joined ONLY by sample_id. city/url/sample_group taken from the
    frozen file (what the human annotator actually saw); model labels and
    stress_reason taken from the key (never present in the frozen file --
    validated above)."""
    key_by_id = {r["sample_id"]: r for r in key_rows}
    joined = []
    for r in frozen_rows:
        k = key_by_id[r["sample_id"]]
        joined.append({
            "sample_id": r["sample_id"],
            "sample_group": r["sample_group"],
            "city": r["city"],
            "url": r["url"],
            "human_fsi_status": (r.get("human_fsi_status") or "").strip(),
            "human_fsi_type": (r.get("human_fsi_type") or "").strip(),
            "human_operational_level": (r.get("human_operational_level") or "").strip(),
            "human_confidence": (r.get("human_confidence") or "").strip(),
            "human_notes": r.get("human_notes") or "",
            "model_fsi_type": k.get("model_fsi_type", ""),
            "model_operational_level": k.get("model_operational_level", ""),
            "stored_text_length": k.get("stored_text_length", ""),
            "stress_reason": k.get("stress_reason", ""),
            "source_record_index": k.get("source_record_index"),
        })
    return joined


def is_evaluable(row: dict) -> bool:
    """The ONLY definition of 'evaluable' used anywhere in this script for
    fsi_type/operational_level agreement: a CORE row whose human annotator
    judged it a genuine FSI. STRESS rows are never evaluated by this
    function (stress agreement is computed separately, see §10)."""
    return row["sample_group"] == "core" and row["human_fsi_status"] == "genuine_fsi"


# ═══════════════════════════════════════════════════════════════════════
# Label-distribution diagnostics (§7 / §8 / stress non-fsi diagnostics)
# ═══════════════════════════════════════════════════════════════════════

def label_diagnostic(rows: list[dict], field: str, enum: set[str]) -> dict:
    """field is 'model_fsi_type' or 'model_operational_level'. Returns
    distribution + specific/unknown/out-of-schema counts, per the literal
    definition: 'specific' = non-empty and != 'unknown'; 'out_of_schema' =
    not in (enum | {'unknown'})."""
    n = len(rows)
    dist = Counter(r[field] for r in rows)
    allowed = enum | {"unknown"}
    n_specific = sum(1 for r in rows if r[field] not in ("", "unknown"))
    n_unknown = sum(1 for r in rows if r[field] == "unknown")
    out_of_schema_vals = [r[field] for r in rows if r[field] not in allowed]
    n_oos = len(out_of_schema_vals)
    return {
        "n": n,
        "distribution": dict(dist),
        "n_specific": n_specific,
        "pct_specific": round(100 * n_specific / n, 1) if n else None,
        "n_unknown": n_unknown,
        "pct_unknown": round(100 * n_unknown / n, 1) if n else None,
        "n_out_of_schema": n_oos,
        "pct_out_of_schema": round(100 * n_oos / n, 1) if n else None,
        "out_of_schema_values": sorted(set(out_of_schema_vals)),
    }


# ═══════════════════════════════════════════════════════════════════════
# §4/§5/§6 — type / operational-level / joint agreement (CORE, evaluable only)
# ═══════════════════════════════════════════════════════════════════════

def agreement_stats(rows: list[dict], human_field: str, model_field: str) -> dict:
    n = len(rows)
    matches = [r for r in rows if r[human_field] == r[model_field]]
    n_match = len(matches)
    return {
        "n": n, "n_match": n_match, "n_mismatch": n - n_match,
        "pct_match": round(100 * n_match / n, 1) if n else None,
    }


def confusion_pairs(rows: list[dict], human_field: str, model_field: str) -> list[dict]:
    """Long-format confusion 'matrix' (human_label, model_label, n) --
    every value used exactly as stored, including 'unknown' and any
    out-of-schema model value; nothing is remapped or collapsed."""
    counts = Counter((r[human_field], r[model_field]) for r in rows)
    return [
        {"human_label": h, "model_label": m, "n": n}
        for (h, m), n in sorted(counts.items())
    ]


def joint_agreement(rows: list[dict]) -> dict:
    both = sum(1 for r in rows if r["human_fsi_type"] == r["model_fsi_type"]
               and r["human_operational_level"] == r["model_operational_level"])
    type_only = sum(1 for r in rows if r["human_fsi_type"] == r["model_fsi_type"]
                     and r["human_operational_level"] != r["model_operational_level"])
    op_only = sum(1 for r in rows if r["human_fsi_type"] != r["model_fsi_type"]
                   and r["human_operational_level"] == r["model_operational_level"])
    neither = sum(1 for r in rows if r["human_fsi_type"] != r["model_fsi_type"]
                   and r["human_operational_level"] != r["model_operational_level"])
    n = len(rows)
    return {
        "n": n,
        "both_match": both, "pct_both_match": round(100 * both / n, 1) if n else None,
        "type_only_match": type_only, "pct_type_only_match": round(100 * type_only / n, 1) if n else None,
        "op_only_match": op_only, "pct_op_only_match": round(100 * op_only / n, 1) if n else None,
        "neither_match": neither, "pct_neither_match": round(100 * neither / n, 1) if n else None,
    }


def by_group(rows: list[dict], key_fn) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for r in rows:
        out[key_fn(r)].append(r)
    return dict(out)


# ═══════════════════════════════════════════════════════════════════════
# §9 — row-level tables
# ═══════════════════════════════════════════════════════════════════════

def build_core_disagreements(evaluable: list[dict]) -> list[dict]:
    rows = []
    for r in evaluable:
        type_match = r["human_fsi_type"] == r["model_fsi_type"]
        op_match = r["human_operational_level"] == r["model_operational_level"]
        if type_match and op_match:
            continue
        rows.append({
            "sample_id": r["sample_id"], "city": r["city"], "url": r["url"],
            "human_confidence": r["human_confidence"],
            "human_fsi_type": r["human_fsi_type"], "model_fsi_type": r["model_fsi_type"],
            "fsi_type_match": type_match,
            "human_operational_level": r["human_operational_level"],
            "model_operational_level": r["model_operational_level"],
            "operational_level_match": op_match,
        })
    return rows


def build_core_status_diagnostics(core_rows: list[dict]) -> list[dict]:
    rows = []
    for r in core_rows:
        if r["human_fsi_status"] not in ("not_an_fsi", "insufficient_evidence"):
            continue
        rows.append({
            "sample_id": r["sample_id"], "city": r["city"], "url": r["url"],
            "human_fsi_status": r["human_fsi_status"], "human_confidence": r["human_confidence"],
            "model_fsi_type": r["model_fsi_type"], "model_operational_level": r["model_operational_level"],
            "fsi_type_specific": r["model_fsi_type"] not in ("", "unknown"),
            "fsi_type_out_of_schema": r["model_fsi_type"] not in FSI_TYPE_ALLOWED,
            "operational_level_specific": r["model_operational_level"] not in ("", "unknown"),
            "operational_level_out_of_schema": r["model_operational_level"] not in OP_LEVEL_ALLOWED,
        })
    return rows


def build_stress_diagnostics(stress_rows: list[dict]) -> list[dict]:
    rows = []
    for r in stress_rows:
        is_genuine = r["human_fsi_status"] == "genuine_fsi"
        rows.append({
            "sample_id": r["sample_id"], "city": r["city"], "stress_reason": r["stress_reason"],
            "human_fsi_status": r["human_fsi_status"], "human_confidence": r["human_confidence"],
            "human_fsi_type": r["human_fsi_type"] if is_genuine else "",
            "model_fsi_type": r["model_fsi_type"],
            "fsi_type_match": (r["human_fsi_type"] == r["model_fsi_type"]) if is_genuine else "",
            "human_operational_level": r["human_operational_level"] if is_genuine else "",
            "model_operational_level": r["model_operational_level"],
            "operational_level_match": (r["human_operational_level"] == r["model_operational_level"]) if is_genuine else "",
        })
    return rows


def build_stress_summary(stress_rows: list[dict]) -> list[dict]:
    """One row per (grouping, group_value) -- grouping is 'stress_reason' or
    'city'. Diagnostic only; never combined with core anywhere."""
    def summarize(rows: list[dict]) -> dict:
        n = len(rows)
        status_counts = Counter(r["human_fsi_status"] for r in rows)
        genuine = [r for r in rows if r["human_fsi_status"] == "genuine_fsi"]
        not_fsi = [r for r in rows if r["human_fsi_status"] == "not_an_fsi"]
        insuff = [r for r in rows if r["human_fsi_status"] == "insufficient_evidence"]
        type_stats = agreement_stats(genuine, "human_fsi_type", "model_fsi_type")
        op_stats = agreement_stats(genuine, "human_operational_level", "model_operational_level")
        n_not_fsi_specific = sum(1 for r in not_fsi if r["model_fsi_type"] not in ("", "unknown"))
        n_insuff_specific = sum(1 for r in insuff if r["model_fsi_type"] not in ("", "unknown"))
        return {
            "n_total": n,
            "n_genuine_fsi": status_counts.get("genuine_fsi", 0),
            "n_not_an_fsi": status_counts.get("not_an_fsi", 0),
            "n_insufficient_evidence": status_counts.get("insufficient_evidence", 0),
            "n_type_evaluable": type_stats["n"], "n_type_match": type_stats["n_match"],
            "pct_type_match": type_stats["pct_match"],
            "n_op_evaluable": op_stats["n"], "n_op_match": op_stats["n_match"],
            "pct_op_match": op_stats["pct_match"],
            "n_not_an_fsi_specific_model_type": n_not_fsi_specific,
            "n_insufficient_evidence_specific_model_type": n_insuff_specific,
        }

    rows = []
    for reason, grp in sorted(by_group(stress_rows, lambda r: r["stress_reason"] or "(none)").items()):
        rows.append({"grouping": "stress_reason", "group_value": reason, **summarize(grp)})
    for city, grp in sorted(by_group(stress_rows, lambda r: r["city"]).items()):
        rows.append({"grouping": "city", "group_value": city, **summarize(grp)})
    return rows


# ═══════════════════════════════════════════════════════════════════════
# §11 — Dublin out-of-schema sentinel
# ═══════════════════════════════════════════════════════════════════════

def find_sentinel(joined: list[dict]) -> list[dict]:
    hits = [r for r in joined if r["model_fsi_type"] == "food_service"]
    rows = []
    for r in hits:
        rows.append({
            "sample_id": r["sample_id"], "sample_group": r["sample_group"], "url": r["url"],
            "human_fsi_status": r["human_fsi_status"], "human_fsi_type": r["human_fsi_type"],
            "human_operational_level": r["human_operational_level"],
            "human_confidence": r["human_confidence"],
            "model_fsi_type": r["model_fsi_type"], "model_operational_level": r["model_operational_level"],
            "fsi_type_match": (r["human_fsi_type"] == r["model_fsi_type"]) if r["human_fsi_status"] == "genuine_fsi" else "",
            "operational_level_match": (r["human_operational_level"] == r["model_operational_level"]) if r["human_fsi_status"] == "genuine_fsi" else "",
        })
    return rows


# ═══════════════════════════════════════════════════════════════════════
# Writing
# ═══════════════════════════════════════════════════════════════════════

def write_csv(name: str, rows: list[dict], fieldnames: list[str], overwrite: bool) -> Path:
    path = _assert_safe_output_path(OUTPUT_ROOT / name)
    if path.exists() and not overwrite:
        raise RuntimeError(f"REFUSING TO OVERWRITE existing output (pass --overwrite to replace): {path}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def write_text(name: str, text: str, overwrite: bool) -> Path:
    path = _assert_safe_output_path(OUTPUT_ROOT / name)
    if path.exists() and not overwrite:
        raise RuntimeError(f"REFUSING TO OVERWRITE existing output (pass --overwrite to replace): {path}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ═══════════════════════════════════════════════════════════════════════
# analysis_summary.md
# ═══════════════════════════════════════════════════════════════════════

def build_summary_markdown(core_rows, stress_rows, evaluable, type_stats, op_stats,
                            joint_stats, not_fsi_type_diag, not_fsi_op_diag,
                            insuff_type_diag, insuff_op_diag, confidence_rows,
                            stress_summary_rows, sentinel_rows) -> str:
    md = []
    md.append("# Phase-2 Classification Human Validation — Analysis\n")

    md.append("## DATA INTEGRITY\n")
    md.append(
        "All validation checks in §1 passed before this analysis ran (75/75 rows in "
        "both files, matching sample_id sets, 50 core/25 stress overall and "
        "10 core/5 stress per city, no missing human_fsi_status/human_confidence, "
        "genuine_fsi rows have both type/op labels and non-genuine rows have neither, "
        "no model-label column present in the frozen file, manifest "
        "actual_total_records=635 and random_seed=42) -- see the printed validation "
        "log for the exact checks run.\n"
    )

    md.append("## CORE HUMAN STATUS DISTRIBUTION\n")
    md.append(
        "CORE is 10 uniformly random records per city (city-balanced), NOT a simple "
        "random sample of all 635 records -- pooled percentages below describe this "
        "50-row sample, not a precise population estimate.\n"
    )
    total_status = Counter(r["human_fsi_status"] for r in core_rows)
    md.append("| status | n (of 50) | % |")
    md.append("|---|---:|---:|")
    for s in FSI_STATUSES:
        n = total_status.get(s, 0)
        md.append(f"| {s} | {n} | {round(100*n/50,1)} |")
    md.append("")
    md.append("Per city:\n")
    md.append("| city | genuine_fsi | not_an_fsi | insufficient_evidence |")
    md.append("|---|---:|---:|---:|")
    for city in CITIES:
        c = [r for r in core_rows if r["city"] == city]
        cc = Counter(r["human_fsi_status"] for r in c)
        md.append(f"| {city} | {cc.get('genuine_fsi',0)} | {cc.get('not_an_fsi',0)} | {cc.get('insufficient_evidence',0)} |")
    md.append("")

    md.append("## FSI-TYPE AGREEMENT\n")
    md.append(
        f"Among {type_stats['n']} CORE genuine_fsi rows (evaluable = CORE + "
        f"human_fsi_status==genuine_fsi): **{type_stats['n_match']}/{type_stats['n']} exact "
        f"matches ({type_stats['pct_match']}%)**, {type_stats['n_mismatch']} mismatches. "
        "Full confusion matrix (long format, every value used as stored, including "
        "'unknown' and any out-of-schema model value) in `core_type_confusion.csv`. "
        "Macro-F1 not computed: this repository does not already use a classification-"
        "metrics library elsewhere, so per the instruction to avoid it unless trivially "
        "available, exact agreement and the confusion matrix are reported instead.\n"
    )

    md.append("## OPERATIONAL-LEVEL AGREEMENT\n")
    md.append(
        f"Among {op_stats['n']} CORE genuine_fsi rows: **{op_stats['n_match']}/{op_stats['n']} "
        f"exact matches ({op_stats['pct_match']}%)**, {op_stats['n_mismatch']} mismatches. "
        "human='unknown' vs model='unknown' counts as an exact match under the same "
        "plain-equality rule as every other label pair -- no special-casing was needed "
        "or applied. Full confusion matrix in `core_operational_confusion.csv`.\n"
    )

    md.append("## JOINT AGREEMENT\n")
    md.append(
        f"Among {joint_stats['n']} CORE genuine_fsi rows: both fields match "
        f"{joint_stats['both_match']} ({joint_stats['pct_both_match']}%); type only "
        f"{joint_stats['type_only_match']} ({joint_stats['pct_type_only_match']}%); "
        f"operational level only {joint_stats['op_only_match']} "
        f"({joint_stats['pct_op_only_match']}%); neither {joint_stats['neither_match']} "
        f"({joint_stats['pct_neither_match']}%).\n"
    )

    md.append("## NON-FSI CLASSIFICATION DIAGNOSTIC\n")
    md.append(
        "CORE rows where the human judged `not_an_fsi` — Phase 2 has no such output "
        "class, so these are reported as **non-FSI records receiving downstream "
        "classification labels**, not as classification accuracy.\n"
    )
    md.append(
        f"n={not_fsi_type_diag['n']}. model_fsi_type: {not_fsi_type_diag['n_specific']} "
        f"({not_fsi_type_diag['pct_specific']}%) given a specific ordinary type, "
        f"{not_fsi_type_diag['n_unknown']} ({not_fsi_type_diag['pct_unknown']}%) "
        f"'unknown', {not_fsi_type_diag['n_out_of_schema']} out-of-schema "
        f"({not_fsi_type_diag['out_of_schema_values']}). model_operational_level: "
        f"{not_fsi_op_diag['n_specific']} ({not_fsi_op_diag['pct_specific']}%) specific, "
        f"{not_fsi_op_diag['n_unknown']} ({not_fsi_op_diag['pct_unknown']}%) 'unknown', "
        f"{not_fsi_op_diag['n_out_of_schema']} out-of-schema "
        f"({not_fsi_op_diag['out_of_schema_values']}). Row-level detail in "
        f"`core_status_diagnostics.csv`.\n"
    )

    md.append("## INSUFFICIENT-EVIDENCE DIAGNOSTIC\n")
    md.append(
        "CORE rows where the human judged the stored evidence `insufficient_evidence`. "
        "Specific model labels here are reported as **model specificity where the human "
        "annotator judged stored evidence insufficient** — not automatically an error — "
        "especially relevant since the Phase-2 prompt explicitly instructs best-guessing "
        "over 'unknown', and a keyword fallback can independently replace 'unknown'.\n"
    )
    md.append(
        f"n={insuff_type_diag['n']}. model_fsi_type: {insuff_type_diag['n_specific']} "
        f"({insuff_type_diag['pct_specific']}%) specific, {insuff_type_diag['n_unknown']} "
        f"({insuff_type_diag['pct_unknown']}%) 'unknown', {insuff_type_diag['n_out_of_schema']} "
        f"out-of-schema. model_operational_level: {insuff_op_diag['n_specific']} "
        f"({insuff_op_diag['pct_specific']}%) specific, {insuff_op_diag['n_unknown']} "
        f"({insuff_op_diag['pct_unknown']}%) 'unknown', {insuff_op_diag['n_out_of_schema']} "
        f"out-of-schema. Row-level detail in `core_status_diagnostics.csv`.\n"
    )

    md.append("## CONFIDENCE SENSITIVITY\n")
    md.append(
        "Sensitivity/interpretation aid only -- low-confidence human disagreement is "
        "not automatically interpreted as model error.\n"
    )
    md.append("| confidence | n | type agreement % | op agreement % |")
    md.append("|---|---:|---:|---:|")
    for row in confidence_rows:
        md.append(f"| {row['human_confidence']} | {row['n']} | {row['pct_type_match']} | {row['pct_op_match']} |")
    md.append("")

    md.append("## STRESS-SAMPLE DIAGNOSTICS\n")
    md.append(
        "STRESS (25 rows, 5/city) was deliberately selected for classifier edge cases "
        "(rare types, other, unknown, short-text-but-specific, type/op-unknown "
        "mismatches, out-of-schema values). **Every result in this section is "
        "diagnostic, not representative, and is never pooled with CORE into one "
        "accuracy/agreement number anywhere in this script.** Full grouped breakdown "
        "(by stress_reason and by city) in `stress_summary.csv`; full row-level detail "
        "in `stress_diagnostics.csv`.\n"
    )
    stress_status = Counter(r["human_fsi_status"] for r in stress_rows)
    md.append(
        f"Overall stress human-status counts: genuine_fsi={stress_status.get('genuine_fsi',0)}, "
        f"not_an_fsi={stress_status.get('not_an_fsi',0)}, "
        f"insufficient_evidence={stress_status.get('insufficient_evidence',0)}.\n"
    )

    md.append("## OUT-OF-SCHEMA SENTINEL\n")
    md.append(
        "`\"food_service\"` is outside the active Phase-2 fsi_type schema "
        "(`src/phase_2/classifier.py`'s SYSTEM_PROMPT enum) and demonstrates the "
        "absence of output-enum validation in the classifier, regardless of what the "
        "human annotator judged this specific record to be.\n"
    )
    if sentinel_rows:
        for r in sentinel_rows:
            md.append(
                f"- `{r['sample_id']}` ({r['sample_group']}): human_fsi_status="
                f"{r['human_fsi_status']!r}, human_fsi_type={r['human_fsi_type']!r}, "
                f"model_fsi_type={r['model_fsi_type']!r}, "
                f"model_operational_level={r['model_operational_level']!r}. "
                f"Full detail in `sentinel_result.csv`."
            )
    else:
        md.append(
            "**NOT FOUND** in this sample's model_fsi_type values -- if this sample was "
            "regenerated since the sentinel was last confirmed present, re-check "
            "`sample_key.csv` directly; this script does not treat its absence as a "
            "validation-abort condition, only reports it here.\n"
        )
    md.append("")

    md.append("## INTERPRETATION LIMITS\n")
    md.append(
        "- CORE is the primary validation sample; STRESS is diagnostic only.\n"
        "- CORE is 10 uniformly random rows per city -- city-balanced, not proportional "
        "to the 635-record corpus (barcelona and london are under-weighted relative to "
        "their true share, brighton/dublin/milan over-weighted); pooled CORE percentages "
        "describe this 50-row sample, not a precise, unweighted population estimate.\n"
        "- CORE and STRESS are never pooled into a single accuracy/agreement number "
        "anywhere in this script.\n"
        "- There is one human annotator: these labels are a reference annotation, not "
        "unquestionable ground truth.\n"
        "- No inter-annotator agreement was measured (single annotator, no second pass).\n"
        "- Several fsi_type/operational_level classes have very small counts within this "
        "50-row sample; per-class conclusions are descriptive, not inferentially powered.\n"
        "- No FSI-status accuracy, precision/recall/F1, confusion matrix, or kappa was "
        "computed anywhere in this script -- Phase 2 has no not_an_fsi/insufficient_"
        "evidence output class to compare against, so no such comparison is meaningful.\n"
    )

    md.append("## THESIS IMPLICATION\n")
    md.append(
        f"**A.** Does CORE support treating model-derived `fsi_type` labels as broadly "
        f"reliable descriptive annotations? Based on {type_stats['n']} evaluable CORE "
        f"rows at {type_stats['pct_match']}% exact agreement -- read this figure "
        f"together with the confusion matrix and city/confidence breakdowns above "
        f"before drawing a conclusion; do not overclaim from N={type_stats['n']}.\n\n"
        f"**B.** Does CORE support treating `operational_level` labels as broadly "
        f"reliable? Based on {op_stats['n']} evaluable CORE rows at "
        f"{op_stats['pct_match']}% exact agreement (including unknown-unknown matches) "
        f"-- same caveat: read alongside the confusion matrix, not as a standalone "
        f"headline figure.\n\n"
        f"**C.** How serious is the absence of a `not_an_fsi` class in practice? See "
        f"the NON-FSI CLASSIFICATION DIAGNOSTIC above: "
        f"{total_status.get('not_an_fsi',0)}/50 CORE rows were judged not_an_fsi by the "
        f"human annotator, and every one of them nonetheless received an ordinary "
        f"Phase-2 type/operational-level label with no exclusion mechanism available.\n\n"
        f"**D.** Which existing report-level conclusions require qualification? Any "
        f"city-comparison or cross-city narrative built on `type_counts`/`type_pct` or "
        f"operational-level breakdowns should be read alongside this validation's "
        f"agreement rates and confusion matrix, not as unqualified fact.\n\n"
        f"**E.** Is rerunning or correcting all 635 classifications necessary for this "
        f"thesis, or is transparent limitation + validation sufficient? This is a "
        f"judgement call for the thesis author to make from the figures above, not a "
        f"conclusion this script draws automatically.\n"
    )

    return "\n".join(md)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def _print_validation_log(errors: list[str]):
    if errors:
        print("VALIDATION ERRORS:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("Validation: OK (all §1 checks passed)")


def _print_dry_run(frozen_rows, key_rows, manifest, errors):
    print("=" * 70)
    print("DRY RUN — no files will be written")
    print("=" * 70)

    print(f"\nFrozen file: {FROZEN_PATH} ({'found' if FROZEN_PATH.exists() else 'MISSING'})")
    print(f"Key file:    {KEY_PATH} ({'found' if KEY_PATH.exists() else 'MISSING'})")
    print(f"Manifest:    {MANIFEST_PATH} ({'found' if MANIFEST_PATH.exists() else 'MISSING'})")

    n_frozen = len(frozen_rows) if frozen_rows is not None else 0
    n_key = len(key_rows) if key_rows is not None else 0
    frozen_ids = {r["sample_id"] for r in frozen_rows} if frozen_rows else set()
    key_ids = {r["sample_id"] for r in key_rows} if key_rows else set()
    n_joined = len(frozen_ids & key_ids)
    print(f"\nFrozen rows: {n_frozen}/75   Key rows: {n_key}/75   Joined by sample_id: {n_joined}/75")

    if key_rows:
        core_n = sum(1 for r in key_rows if r.get("sample_group") == "core")
        stress_n = sum(1 for r in key_rows if r.get("sample_group") == "stress")
        print(f"Core: {core_n}/50   Stress: {stress_n}/25")
        for city in CITIES:
            c_core = sum(1 for r in key_rows if r.get("city") == city and r.get("sample_group") == "core")
            c_stress = sum(1 for r in key_rows if r.get("city") == city and r.get("sample_group") == "stress")
            print(f"  {city}: core={c_core}/10 stress={c_stress}/5")

    if frozen_rows:
        status_counts = Counter((r.get("human_fsi_status") or "").strip() for r in frozen_rows)
        print("\nHuman status counts (all 75 rows):")
        for s in FSI_STATUSES:
            print(f"  {s}: {status_counts.get(s, 0)}")

    print()
    _print_validation_log(errors)

    print("\nPlanned output paths (none created in --dry-run):")
    for name in sorted(ALLOWED_OUTPUT_NAMES):
        print(f"  {OUTPUT_ROOT / name}")

    print("\nNo output has been written. Dry run complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Analyse the completed Phase-2 human-validation study (read-only)."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs and print the plan; write nothing.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow overwriting existing analysis outputs. Off by default.")
    args = parser.parse_args()

    frozen_rows = load_frozen() if FROZEN_PATH.exists() else None
    key_rows = load_key() if KEY_PATH.exists() else None
    manifest = load_manifest() if MANIFEST_PATH.exists() else None

    errors = validate(frozen_rows, key_rows, manifest)

    if args.dry_run:
        _print_dry_run(frozen_rows, key_rows, manifest, errors)
        if errors:
            sys.exit(1)
        return

    if errors:
        print("VALIDATION FAILED — aborting before any analysis:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    joined = build_joined(frozen_rows, key_rows)
    core_rows = [r for r in joined if r["sample_group"] == "core"]
    stress_rows = [r for r in joined if r["sample_group"] == "stress"]
    evaluable = [r for r in core_rows if is_evaluable(r)]

    type_stats = agreement_stats(evaluable, "human_fsi_type", "model_fsi_type")
    op_stats = agreement_stats(evaluable, "human_operational_level", "model_operational_level")
    joint_stats = joint_agreement(evaluable)

    core_summary_rows = []
    total_status = Counter(r["human_fsi_status"] for r in core_rows)
    core_summary_rows.append({"scope": "pooled", "city": "", **{s: total_status.get(s, 0) for s in FSI_STATUSES},
                               "n": len(core_rows)})
    for city in CITIES:
        c = [r for r in core_rows if r["city"] == city]
        cc = Counter(r["human_fsi_status"] for r in c)
        core_summary_rows.append({"scope": "per_city", "city": city,
                                   **{s: cc.get(s, 0) for s in FSI_STATUSES}, "n": len(c)})
    core_summary_rows.append({"scope": "type_agreement_pooled", "city": "", **type_stats,
                               "genuine_fsi": "", "not_an_fsi": "", "insufficient_evidence": ""})
    for city in CITIES:
        c_eval = [r for r in evaluable if r["city"] == city]
        stats = agreement_stats(c_eval, "human_fsi_type", "model_fsi_type")
        core_summary_rows.append({"scope": "type_agreement_per_city", "city": city, **stats,
                                   "genuine_fsi": "", "not_an_fsi": "", "insufficient_evidence": ""})
    core_summary_rows.append({"scope": "op_agreement_pooled", "city": "", **op_stats,
                               "genuine_fsi": "", "not_an_fsi": "", "insufficient_evidence": ""})
    for city in CITIES:
        c_eval = [r for r in evaluable if r["city"] == city]
        stats = agreement_stats(c_eval, "human_operational_level", "model_operational_level")
        core_summary_rows.append({"scope": "op_agreement_per_city", "city": city, **stats,
                                   "genuine_fsi": "", "not_an_fsi": "", "insufficient_evidence": ""})
    core_summary_rows.append({"scope": "joint_agreement_pooled", "city": "", **joint_stats,
                               "genuine_fsi": "", "not_an_fsi": "", "insufficient_evidence": ""})

    type_confusion = confusion_pairs(evaluable, "human_fsi_type", "model_fsi_type")
    op_confusion = confusion_pairs(evaluable, "human_operational_level", "model_operational_level")

    core_disagreements = build_core_disagreements(evaluable)
    core_status_diagnostics = build_core_status_diagnostics(core_rows)

    not_fsi_core = [r for r in core_rows if r["human_fsi_status"] == "not_an_fsi"]
    insuff_core = [r for r in core_rows if r["human_fsi_status"] == "insufficient_evidence"]
    not_fsi_type_diag = label_diagnostic(not_fsi_core, "model_fsi_type", FSI_TYPE_ENUM)
    not_fsi_op_diag = label_diagnostic(not_fsi_core, "model_operational_level", OP_LEVEL_ENUM)
    insuff_type_diag = label_diagnostic(insuff_core, "model_fsi_type", FSI_TYPE_ENUM)
    insuff_op_diag = label_diagnostic(insuff_core, "model_operational_level", OP_LEVEL_ENUM)

    confidence_rows = []
    for conf in CONFIDENCE_LEVELS:
        c_eval = [r for r in evaluable if r["human_confidence"] == conf]
        t = agreement_stats(c_eval, "human_fsi_type", "model_fsi_type")
        o = agreement_stats(c_eval, "human_operational_level", "model_operational_level")
        confidence_rows.append({
            "human_confidence": conf, "n": t["n"],
            "n_type_match": t["n_match"], "pct_type_match": t["pct_match"],
            "n_op_match": o["n_match"], "pct_op_match": o["pct_match"],
        })

    stress_summary_rows = build_stress_summary(stress_rows)
    stress_diag_rows = build_stress_diagnostics(stress_rows)
    sentinel_rows = find_sentinel(joined)

    write_csv("core_summary.csv", core_summary_rows,
              ["scope", "city", "n", "genuine_fsi", "not_an_fsi", "insufficient_evidence",
               "n_match", "n_mismatch", "pct_match",
               "both_match", "pct_both_match", "type_only_match", "pct_type_only_match",
               "op_only_match", "pct_op_only_match", "neither_match", "pct_neither_match"],
              args.overwrite)
    write_csv("core_type_confusion.csv", type_confusion, ["human_label", "model_label", "n"], args.overwrite)
    write_csv("core_operational_confusion.csv", op_confusion, ["human_label", "model_label", "n"], args.overwrite)
    write_csv("core_disagreements.csv", core_disagreements,
              ["sample_id", "city", "url", "human_confidence", "human_fsi_type", "model_fsi_type",
               "fsi_type_match", "human_operational_level", "model_operational_level",
               "operational_level_match"], args.overwrite)
    write_csv("core_status_diagnostics.csv", core_status_diagnostics,
              ["sample_id", "city", "url", "human_fsi_status", "human_confidence",
               "model_fsi_type", "model_operational_level", "fsi_type_specific",
               "fsi_type_out_of_schema", "operational_level_specific",
               "operational_level_out_of_schema"], args.overwrite)
    write_csv("core_confidence_summary.csv", confidence_rows,
              ["human_confidence", "n", "n_type_match", "pct_type_match", "n_op_match", "pct_op_match"],
              args.overwrite)
    write_csv("stress_summary.csv", stress_summary_rows,
              ["grouping", "group_value", "n_total", "n_genuine_fsi", "n_not_an_fsi",
               "n_insufficient_evidence", "n_type_evaluable", "n_type_match", "pct_type_match",
               "n_op_evaluable", "n_op_match", "pct_op_match",
               "n_not_an_fsi_specific_model_type", "n_insufficient_evidence_specific_model_type"],
              args.overwrite)
    write_csv("stress_diagnostics.csv", stress_diag_rows,
              ["sample_id", "city", "stress_reason", "human_fsi_status", "human_confidence",
               "human_fsi_type", "model_fsi_type", "fsi_type_match",
               "human_operational_level", "model_operational_level", "operational_level_match"],
              args.overwrite)
    write_csv("sentinel_result.csv", sentinel_rows,
              ["sample_id", "sample_group", "url", "human_fsi_status", "human_fsi_type",
               "human_operational_level", "human_confidence", "model_fsi_type",
               "model_operational_level", "fsi_type_match", "operational_level_match"],
              args.overwrite)

    summary_md = build_summary_markdown(
        core_rows, stress_rows, evaluable, type_stats, op_stats, joint_stats,
        not_fsi_type_diag, not_fsi_op_diag, insuff_type_diag, insuff_op_diag,
        confidence_rows, stress_summary_rows, sentinel_rows,
    )
    write_text("analysis_summary.md", summary_md, args.overwrite)

    print(f"Analysis complete. Outputs written under {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
