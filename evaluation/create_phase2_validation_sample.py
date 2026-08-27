"""
Phase-2 classification human-validation sampling — additive, read-only.

Draws a blinded human-annotation sample from the already-generated
data/<city>/output/fsi_enriched.jsonl files (all five cities), to validate
the Phase-2 classifier's fsi_type / operational_level labels against human
judgement. Descriptive master's-thesis validation, NOT a powered population
accuracy study (see annotation_rubric.md for the annotation task itself).

READ-ONLY against:
  - data/<city>/output/fsi_enriched.jsonl, for all five cities -- opened only
    in read mode, hashed before sampling, never written to. This script
    imports nothing from src/phase_2, src/phase_0, GraphRAG, or Ollama, and
    makes no network call -- there is no code path here that could rerun
    Phase 2 or any other pipeline stage.

WRITES ONLY under evaluation/results/phase2_validation/ -- enforced by
_assert_safe_output_path(), called before every write, which rejects any
path not a direct child of that directory or under data/. Refuses to
overwrite an existing output unless --overwrite is passed.

Two groups, drawn in this exact order for reproducibility (see main()):
  A. CORE (50 total, 10/city): uniform random sample without replacement,
     with NO reference to fsi_type/operational_level at all -- selection is
     over record *indices* only, before any label is read.
  B. STRESS (25 total, 5/city): does not overlap the core sample by
     construction (drawn only from indices NOT selected for core -- core is
     always drawn first, uniformly over ALL of a city's records, including
     the known Dublin out-of-schema sentinel; only after core is fixed does
     _sample_city() check whether the sentinel landed there and either
     leave it in core or force it into stress -- see _sample_city()).
     Targets 7 priority categories (rare type, other,
     unknown, short-text-but-specific, type-specific/op-unknown,
     type-unknown/op-specific, out-of-schema), round-robinned to maximise
     category variety within each city's 5 slots rather than repeating one
     category.

The human annotator sees the FULL stored title/description/text (not the
<=1000-character window the original classifier saw) and never sees the
model's own fsi_type/operational_level -- those are written only to
sample_key.csv, a separate, non-blinded file not meant to be consulted while
annotating.

Usage:
    python evaluation/create_phase2_validation_sample.py --dry-run
    python evaluation/create_phase2_validation_sample.py
    python evaluation/create_phase2_validation_sample.py --overwrite
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = (ROOT / "data").resolve()
OUTPUT_ROOT = (ROOT / "evaluation" / "results" / "phase2_validation").resolve()

CITIES = ["barcelona", "brighton", "dublin", "london", "milan"]
SEED = 42
EXPECTED_TOTAL_RECORDS = 635
N_CORE_PER_CITY = 10
N_STRESS_PER_CITY = 5

FSI_TYPES_ALLOWED = {
    "food_sharing", "food_swapping", "food_gifting", "community_garden",
    "food_bank", "meals_service", "food_education", "other", "unknown",
}
OP_LEVELS_ALLOWED = {
    "government_funded", "council_supported", "ngo_led", "community_led",
    "commercial", "unknown",
}

# Priority order for the stress round-robin -- matches the task's numbered
# list (1-7) exactly.
STRESS_CATEGORY_ORDER = [
    "rare_type",                 # 1. food_swapping / food_gifting
    "type_other",                # 2. fsi_type == other
    "type_unknown",               # 3. fsi_type == unknown
    "short_text_specific",        # 4. <50 chars stored text, specific label
    "type_specific_op_unknown",   # 5. specific type + operational_level unknown
    "type_unknown_op_specific",   # 6. type unknown + specific operational_level
    "out_of_schema",              # 7. any value outside the allowed enums
]

ALLOWED_OUTPUT_NAMES = {
    "sample_blinded.csv", "sample_key.csv", "annotation_rubric.md", "sample_manifest.json",
}

BLINDED_FIELDS = [
    "sample_id", "sample_group", "city", "url", "title", "description", "text",
    "human_fsi_status", "human_fsi_type", "human_operational_level",
    "human_confidence", "human_notes",
]
KEY_FIELDS = [
    "sample_id", "sample_group", "city", "source_record_index", "url",
    "model_fsi_type", "model_operational_level", "stored_text_length",
    "stress_reason", "selection_probability_or_design_note",
]

# ── Module-load-time safety assertions ─────────────────────────────────────
assert DATA_ROOT not in OUTPUT_ROOT.parents and OUTPUT_ROOT != DATA_ROOT, \
    "OUTPUT_ROOT must not resolve inside data/"
assert str(OUTPUT_ROOT).startswith(str(ROOT)), \
    "OUTPUT_ROOT must be inside the repository root"


def _assert_safe_output_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == DATA_ROOT or DATA_ROOT in resolved.parents:
        raise RuntimeError(f"SAFETY ABORT: refusing to write under data/: {resolved}")
    if resolved.parent != OUTPUT_ROOT:
        raise RuntimeError(f"SAFETY ABORT: output path must be a direct child of {OUTPUT_ROOT}: {resolved}")
    if resolved.name not in ALLOWED_OUTPUT_NAMES:
        raise RuntimeError(f"SAFETY ABORT: filename not in the allowed output list: {resolved.name}")
    return resolved


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=str(ROOT), timeout=15,
        )
        return out.stdout.strip() or None
    except Exception as e:
        print(f"  [warn] could not query git rev-parse HEAD: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════
# Loading (read-only)
# ═══════════════════════════════════════════════════════════════════════

def _fsi_enriched_path(city: str) -> Path:
    return DATA_ROOT / city / "output" / "fsi_enriched.jsonl"


def load_all_records() -> dict[str, list[dict]]:
    records = {}
    for city in CITIES:
        path = _fsi_enriched_path(city)
        recs = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        records[city] = recs
    return records


# ═══════════════════════════════════════════════════════════════════════
# Stress-category classification (read-only labelling of existing records
# for sampling purposes -- does not touch fsi_type/operational_level values,
# only reads them)
# ═══════════════════════════════════════════════════════════════════════

def stored_text_length(record: dict) -> int:
    return len(record.get("text") or "")


def stress_categories(record: dict) -> list[str]:
    """Every priority category this record qualifies for (a record may
    match more than one; the sampler removes a picked record from all pools
    once selected, so double-counting cannot occur in the final sample)."""
    cats = []
    ft = record.get("fsi_type", "unknown")
    ol = record.get("operational_level", "unknown")
    tlen = stored_text_length(record)

    if ft in ("food_swapping", "food_gifting"):
        cats.append("rare_type")
    if ft == "other":
        cats.append("type_other")
    if ft == "unknown":
        cats.append("type_unknown")
    if tlen < 50 and (ft != "unknown" or ol != "unknown"):
        cats.append("short_text_specific")
    if ft not in ("unknown", "other") and ol == "unknown":
        cats.append("type_specific_op_unknown")
    if ft == "unknown" and ol != "unknown":
        cats.append("type_unknown_op_specific")
    if ft not in FSI_TYPES_ALLOWED or ol not in OP_LEVELS_ALLOWED:
        cats.append("out_of_schema")
    return cats


def find_forced_dublin_index(records: list[dict]) -> int:
    """The known out-of-schema fsi_type="food_service" Dublin record. Found
    generically (any record whose fsi_type is outside FSI_TYPES_ALLOWED),
    not by hardcoded URL, so this self-verifies against the live file. Hard
    error if not found -- the task requires it be included, so a missing
    record must abort loudly, not be silently skipped."""
    forced = [i for i, r in enumerate(records) if r.get("fsi_type") not in FSI_TYPES_ALLOWED]
    if not forced:
        raise RuntimeError(
            "Expected out-of-schema fsi_type record not found in dublin's "
            "fsi_enriched.jsonl (previously observed fsi_type='food_service'). "
            "Aborting rather than silently sampling without it."
        )
    if len(forced) > 1:
        print(f"  NOTE: {len(forced)} out-of-schema records found in dublin "
              f"(expected 1); forcing the first by file order: index {forced[0]}")
    return forced[0]


# ═══════════════════════════════════════════════════════════════════════
# Per-city sampling
# ═══════════════════════════════════════════════════════════════════════

def _sample_city(city: str, records: list[dict], rng: random.Random) -> dict:
    """Returns {'core': [indices], 'stress': [indices], 'stress_reason':
    {index: [categories]}, 'stress_note': {index: str}, 'forced_idx':
    int|None, 'forced_in_core': bool}.

    Order of operations (fixed, for reproducibility):
      1. CORE FIRST, unconditionally, for every city including dublin: draw
         10 indices uniformly at random from the FULL, unfiltered set of
         this city's record indices. fsi_type, operational_level, stress
         category, and the known dublin out-of-schema sentinel are not
         inspected before or during this draw -- every record in the city,
         including the sentinel, has the same N_CORE_PER_CITY/n probability
         of being drawn. See the CORE INDEPENDENCE assertion just below.
      2. Only after core is fixed: if dublin, locate the known out-of-schema
         sentinel and check whether the core draw already selected it.
         - If NOT in core: force it into stress (its only remaining route
           into the sample), then round-robin the other 4 dublin stress
           slots as usual.
         - If already IN core: leave it in core (do not duplicate it into
           stress); select all 5 dublin stress slots from the remaining
           non-core records via the normal round-robin, and flag the core
           row itself in sample_key.csv as also being the sentinel.
      3. Build stress category pools from every record NOT in core and NOT
         already placed in stress, in ascending index order (deterministic).
      4. Round-robin through STRESS_CATEGORY_ORDER, picking one record per
         available category per pass (rng.choice on a sorted candidate
         list) until 5 slots are filled or all pools are exhausted; any
         remaining slots are filled by uniform random draw from whatever
         non-core, non-selected records remain.
    """
    n = len(records)

    # CORE INDEPENDENCE: sampled from the full, unfiltered index range --
    # no record (including the dublin sentinel) is excluded from this draw
    # for any label- or suspiciousness-based reason. No fsi_type/
    # operational_level/stress-category value is read anywhere above this
    # line for this city.
    all_indices = list(range(n))
    assert len(all_indices) == n, "core sampling frame must equal the full city record count"
    core_indices = sorted(rng.sample(all_indices, N_CORE_PER_CITY))

    stress_note: dict[int, str] = {}
    stress_indices: list[int] = []
    stress_reason_forced: dict[int, list[str]] = {}
    forced_idx = None
    forced_in_core = False

    if city == "dublin":
        forced_idx = find_forced_dublin_index(records)
        if forced_idx in core_indices:
            # Sentinel landed in core via the uniform draw -- stays there,
            # not added to stress_indices. build_sample() generates this
            # row's design note directly from forced_in_core/forced_idx
            # (see its core-row branch), so nothing further is needed here.
            forced_in_core = True
        else:
            stress_indices.append(forced_idx)
            stress_reason_forced[forced_idx] = ["forced_out_of_schema"]
            stress_note[forced_idx] = (
                "forced inclusion (out-of-schema fsi_type); sentinel was not drawn into "
                "core, so added directly to stress"
            )

    excluded = set(core_indices) | set(stress_indices)
    pools: dict[str, list[int]] = {cat: [] for cat in STRESS_CATEGORY_ORDER}
    cats_by_index: dict[int, list[str]] = {}
    for i in range(n):
        if i in excluded:
            continue
        cats = stress_categories(records[i])
        if cats:
            cats_by_index[i] = cats
            for c in cats:
                pools[c].append(i)

    stress_reason: dict[int, list[str]] = dict(stress_reason_forced)

    def remove_from_pools(idx: int):
        for c in pools:
            if idx in pools[c]:
                pools[c].remove(idx)

    for idx in stress_indices:
        remove_from_pools(idx)

    guard = 0
    while len(stress_indices) < N_STRESS_PER_CITY and guard < 100:
        guard += 1
        progressed = False
        for cat in STRESS_CATEGORY_ORDER:
            if len(stress_indices) >= N_STRESS_PER_CITY:
                break
            candidates = sorted(pools[cat])
            if not candidates:
                continue
            pick = rng.choice(candidates)
            stress_indices.append(pick)
            stress_reason[pick] = cats_by_index.get(pick, [cat])
            stress_note[pick] = (
                f"priority-category round robin (matched: {','.join(cats_by_index.get(pick, [cat]))}); "
                f"pool size at selection={len(candidates)}"
            )
            remove_from_pools(pick)
            progressed = True
        if not progressed:
            break

    if len(stress_indices) < N_STRESS_PER_CITY:
        remaining = sorted(
            i for i in range(n) if i not in core_indices and i not in stress_indices
        )
        needed = N_STRESS_PER_CITY - len(stress_indices)
        extra = rng.sample(remaining, min(needed, len(remaining))) if remaining else []
        for idx in extra:
            stress_reason[idx] = ["no_priority_category_matched"]
            stress_note[idx] = "fallback: priority pools exhausted, uniform random from remaining non-core records"
        stress_indices.extend(extra)

    return {
        "core": core_indices,
        "stress": stress_indices[:N_STRESS_PER_CITY],
        "stress_reason": stress_reason,
        "stress_note": stress_note,
        "forced_idx": forced_idx,
        "forced_in_core": forced_in_core,
    }


# ═══════════════════════════════════════════════════════════════════════
# Assembly
# ═══════════════════════════════════════════════════════════════════════

def build_sample(records_by_city: dict[str, list[dict]]) -> tuple[list[dict], list[dict]]:
    """Deterministic single RNG, advanced city-by-city in CITIES order (see
    module docstring). Returns (blinded_rows, key_rows)."""
    rng = random.Random(SEED)
    blinded_rows = []
    key_rows = []
    sample_counter = 0

    for city in CITIES:
        records = records_by_city[city]
        plan = _sample_city(city, records, rng)

        for group, indices in (("core", plan["core"]), ("stress", plan["stress"])):
            for idx in indices:
                sample_counter += 1
                sample_id = f"P2-{sample_counter:03d}"
                rec = records[idx]

                blinded_rows.append({
                    "sample_id": sample_id, "sample_group": group, "city": city,
                    "url": rec.get("url", ""), "title": rec.get("title", ""),
                    "description": rec.get("description", ""), "text": rec.get("text", ""),
                    "human_fsi_status": "", "human_fsi_type": "",
                    "human_operational_level": "", "human_confidence": "", "human_notes": "",
                })

                if group == "core":
                    if plan["forced_in_core"] and idx == plan["forced_idx"]:
                        reason = "core_and_known_out_of_schema_sentinel"
                        note = (
                            f"uniform random without replacement, {N_CORE_PER_CITY}/{len(records)}, "
                            f"seed={SEED} -- this is also the known out-of-schema sentinel "
                            f"(fsi_type='food_service'), which landed in core naturally and was "
                            f"NOT duplicated into stress"
                        )
                    else:
                        reason = ""
                        note = f"uniform random without replacement, {N_CORE_PER_CITY}/{len(records)}, seed={SEED}"
                else:
                    reason = ",".join(plan["stress_reason"].get(idx, []))
                    note = plan["stress_note"].get(idx, "")

                key_rows.append({
                    "sample_id": sample_id, "sample_group": group, "city": city,
                    "source_record_index": idx, "url": rec.get("url", ""),
                    "model_fsi_type": rec.get("fsi_type", "unknown"),
                    "model_operational_level": rec.get("operational_level", "unknown"),
                    "stored_text_length": stored_text_length(rec),
                    "stress_reason": reason, "selection_probability_or_design_note": note,
                })

    return blinded_rows, key_rows


# ═══════════════════════════════════════════════════════════════════════
# Writing
# ═══════════════════════════════════════════════════════════════════════

def write_csv(name: str, rows: list[dict], fieldnames: list[str], overwrite: bool) -> Path:
    path = _assert_safe_output_path(OUTPUT_ROOT / name)
    if path.exists() and not overwrite:
        raise RuntimeError(f"REFUSING TO OVERWRITE existing output (pass --overwrite to replace): {path}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Blinding guard: sample_blinded.csv must never contain a model-label
    # column, checked at write time regardless of caller.
    if name == "sample_blinded.csv":
        forbidden = {"fsi_type", "operational_level", "model_fsi_type", "model_operational_level"}
        if forbidden & set(fieldnames):
            raise RuntimeError(f"SAFETY ABORT: model label column(s) present in sample_blinded.csv fieldnames: {forbidden & set(fieldnames)}")
        for row in rows:
            if forbidden & set(row.keys()):
                raise RuntimeError("SAFETY ABORT: model label found in a sample_blinded.csv row")

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


RUBRIC_MD = """# Phase-2 Classification Validation — Annotation Rubric

Descriptive master's-thesis validation of the Phase-2 `fsi_type` /
`operational_level` labels. You are shown the FULL stored `title` /
`description` / `text` for each record — not the truncated (<=1000
character) excerpt the original classifier actually saw. The model's own
labels are hidden; do not attempt to guess or look them up while annotating.

## STEP 1 — FSI STATUS

`human_fsi_status` must be exactly one of:

- **genuine_fsi** — the stored evidence describes an identifiable
  initiative, programme, organisation, site or recurring activity whose
  purpose includes sharing, redistributing, providing, growing communally,
  swapping, or otherwise making food available through a
  collective/community/service mechanism.
- **not_an_fsi** — the evidence describes something related to food but is
  not itself an initiative of this type, for example: a directory/listing
  page; a policy/document; a news article about initiatives (rather than an
  initiative itself); generic council information; a commercial food
  business with no food-sharing/community-service function; an unrelated
  page.
- **insufficient_evidence** — the stored title/description/text does not
  contain enough information to decide.

## STEP 2 — FSI TYPE

Only if `human_fsi_status = genuine_fsi`. Choose exactly one primary type:

- **food_sharing** — redistributing or sharing surplus/prepared food between
  people or organisations (not a food bank, meal service, or single gifting
  event).
- **food_swapping** — organised exchange of food/produce/seeds between
  participants (swap events, seed swaps).
- **food_gifting** — food made freely available with no exchange expected
  (community fridges, "take what you need" points).
- **community_garden** — communal growing space (allotments, urban gardens,
  orchards) whose primary activity is growing food together.
- **food_bank** — collects and distributes food to people in need, typically
  via a referral or drop-in system.
- **meals_service** — prepares and serves meals directly (soup kitchens,
  community dining, social canteens).
- **food_education** — primary activity is teaching about food (cooking
  classes, nutrition workshops, food-growing training) rather than
  distributing/serving/growing food itself.
- **other** — a genuine FSI that does not fit any of the seven categories
  above. **"other" must NEVER be used for `not_an_fsi` cases** — if the page
  is not an FSI at all, use `human_fsi_status = not_an_fsi` in Step 1
  instead, and leave `human_fsi_type` blank.

If multiple types plausibly apply, select the most central/primary activity
and explain the ambiguity in `human_notes`.

## STEP 3 — OPERATIONAL LEVEL

Only if `human_fsi_status = genuine_fsi`. Choose exactly one:

- **government_funded** — explicit national/state government funding or
  operation.
- **council_supported** — explicit local council/municipal funding, grant,
  or operational support.
- **ngo_led** — run by a registered charity, non-profit, or NGO/foundation.
- **community_led** — grassroots, volunteer-run, or community-group led,
  with no evidence of government/council/NGO/commercial structure.
- **commercial** — social enterprise, CIC, or otherwise operates on a
  trading/membership/fee basis.
- **unknown** — the initiative appears genuine but the operational/funding
  model cannot be determined from the stored evidence.

Use a specific category **only** where the stored evidence directly
supports it. **Do not "best guess" from generic food-related wording alone**
— if the text never mentions funding, ownership, or organisational status,
use `unknown`. If two categories are genuinely plausible, choose the best
supported one and note the ambiguity in `human_notes`.

## STEP 4 — CONFIDENCE

`human_confidence`, exactly one of:

- **high** — explicit evidence directly supports the chosen labels.
- **medium** — reasonable inference from the evidence, but not stated
  explicitly.
- **low** — weak or ambiguous evidence; the label is a plausible best
  reading, not a confident one.

## Notes

- `human_notes` is free text — use it for any ambiguity, multiple plausible
  types/operational levels, or reasoning you want on record.
- This is a descriptive validation sample (75 records total: 50 uniformly
  random "core" records, 25 deliberately edge-case "stress" records that
  must not be treated as representative and must not be pooled with the
  core sample when computing an overall agreement statistic).
"""


def build_manifest(records_by_city: dict[str, list[dict]], input_hashes: dict[str, str],
                    blinded_rows: list[dict], key_rows: list[dict]) -> dict:
    actual_total = sum(len(v) for v in records_by_city.values())
    return {
        "random_seed": SEED,
        "git_commit": _get_git_commit(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_files_sha256": {
            city: {"path": str(_fsi_enriched_path(city).relative_to(ROOT)), "sha256": input_hashes[city]}
            for city in CITIES
        },
        "expected_total_records": EXPECTED_TOTAL_RECORDS,
        "actual_total_records": actual_total,
        "per_city_record_counts": {city: len(records_by_city[city]) for city in CITIES},
        "core_count": sum(1 for r in key_rows if r["sample_group"] == "core"),
        "stress_count": sum(1 for r in key_rows if r["sample_group"] == "stress"),
        "core_per_city": {
            city: sum(1 for r in key_rows if r["sample_group"] == "core" and r["city"] == city)
            for city in CITIES
        },
        "stress_per_city": {
            city: sum(1 for r in key_rows if r["sample_group"] == "stress" and r["city"] == city)
            for city in CITIES
        },
        "sample_ids": [r["sample_id"] for r in blinded_rows],
        "note": (
            "All five data/<city>/output/fsi_enriched.jsonl source files were opened "
            "read-only. No pipeline phase was run, no Ollama or GraphRAG call was made, "
            "and no source file was modified in the generation of this sample."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def _print_dry_run(records_by_city: dict[str, list[dict]]):
    print("=" * 70)
    print("DRY RUN — no files will be written")
    print("=" * 70)

    total = sum(len(v) for v in records_by_city.values())
    print(f"\nTotal records found: {total} (expected {EXPECTED_TOTAL_RECORDS})")
    for city in CITIES:
        print(f"  {city:10s} n={len(records_by_city[city])}")

    print(f"\nPlanned core sample: {N_CORE_PER_CITY}/city x {len(CITIES)} cities = "
          f"{N_CORE_PER_CITY * len(CITIES)}")
    print(f"Planned stress sample: {N_STRESS_PER_CITY}/city x {len(CITIES)} cities = "
          f"{N_STRESS_PER_CITY * len(CITIES)}")

    print("\n--- Stress-category availability per city (candidate counts, pre-selection) ---")
    for city in CITIES:
        records = records_by_city[city]
        counts = {cat: 0 for cat in STRESS_CATEGORY_ORDER}
        for r in records:
            for cat in stress_categories(r):
                counts[cat] += 1
        print(f"  {city}: " + ", ".join(f"{cat}={counts[cat]}" for cat in STRESS_CATEGORY_ORDER))

    print("\n--- Dublin known out-of-schema sentinel (fsi_type='food_service') ---")
    try:
        idx = find_forced_dublin_index(records_by_city["dublin"])
        rec = records_by_city["dublin"][idx]
        print(f"  found at index {idx}: fsi_type={rec.get('fsi_type')!r} url={rec.get('url')!r}")

        # Replicate build_sample()'s exact RNG sequence (one shared
        # random.Random(SEED), advanced city-by-city in CITIES order) so the
        # preview below reflects precisely what the real run would produce
        # -- computation only, no files touched, no data mutated.
        preview_rng = random.Random(SEED)
        dublin_plan = None
        for c in CITIES:
            plan = _sample_city(c, records_by_city[c], preview_rng)
            if c == "dublin":
                dublin_plan = plan
        if dublin_plan["forced_in_core"]:
            print(f"  -> would land naturally in CORE (index {idx} drawn by the uniform "
                  f"core sample); NOT duplicated into stress")
        else:
            print(f"  -> would NOT be drawn into core; added to STRESS after core sampling")
    except RuntimeError as e:
        print(f"  NOT FOUND — {e}")

    print(f"\nRandom seed: {SEED}")

    print("\nPlanned output paths (none created in --dry-run):")
    for name in sorted(ALLOWED_OUTPUT_NAMES):
        print(f"  {OUTPUT_ROOT / name}")

    print("\nNo output has been written. Dry run complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Draw a blinded human-validation sample for the Phase-2 classifier."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs and print the plan; write nothing.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow overwriting existing outputs. Off by default.")
    args = parser.parse_args()

    records_by_city = load_all_records()
    total = sum(len(v) for v in records_by_city.values())

    # RECORD COUNT CHECK -- applies identically to --dry-run and the real
    # run: do not continue (sampling OR printing a sampling plan) against a
    # source corpus that doesn't match the expected total.
    if total != EXPECTED_TOTAL_RECORDS:
        print(f"VALIDATION FAILED — expected {EXPECTED_TOTAL_RECORDS} total records, found {total}:")
        for city in CITIES:
            print(f"  {city}: {len(records_by_city[city])}")
        print("Aborting before sampling.")
        sys.exit(1)

    if args.dry_run:
        _print_dry_run(records_by_city)
        return

    input_hashes = {city: _sha256(_fsi_enriched_path(city)) for city in CITIES}

    blinded_rows, key_rows = build_sample(records_by_city)

    assert len(blinded_rows) == len(key_rows) == 75, \
        f"expected 75 sample rows, built {len(blinded_rows)} blinded / {len(key_rows)} key"
    core_n = sum(1 for r in key_rows if r["sample_group"] == "core")
    stress_n = sum(1 for r in key_rows if r["sample_group"] == "stress")
    assert core_n == 50 and stress_n == 25, f"expected 50 core / 25 stress, got {core_n} / {stress_n}"

    core_urls_by_city = {city: set() for city in CITIES}
    stress_urls_by_city = {city: set() for city in CITIES}
    for r in key_rows:
        target = core_urls_by_city if r["sample_group"] == "core" else stress_urls_by_city
        target[r["city"]].add(r["source_record_index"])
    for city in CITIES:
        overlap = core_urls_by_city[city] & stress_urls_by_city[city]
        assert not overlap, f"{city}: core/stress index overlap detected: {overlap}"
        assert len(core_urls_by_city[city]) == N_CORE_PER_CITY, city
        assert len(stress_urls_by_city[city]) == N_STRESS_PER_CITY, city

    write_csv("sample_blinded.csv", blinded_rows, BLINDED_FIELDS, args.overwrite)
    write_csv("sample_key.csv", key_rows, KEY_FIELDS, args.overwrite)
    write_text("annotation_rubric.md", RUBRIC_MD, args.overwrite)

    manifest = build_manifest(records_by_city, input_hashes, blinded_rows, key_rows)
    write_text("sample_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False), args.overwrite)

    print(f"Sample created: 50 core + 25 stress = 75 records, under {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
