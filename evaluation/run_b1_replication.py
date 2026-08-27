"""
Phase-5 replication / stability experiment — Full vs. B1.

Generates 3 NEW Full-condition and 3 NEW B1-condition Phase-5 prose
generations per city, for barcelona / brighton / dublin / london / milan,
using the pipeline's own unmodified synthesize_all(). The existing original
report_<city>.html/_b1.html and their prose are NOT treated as replicate 1 —
this script only ever produces rep1/rep2/rep3, all newly generated today,
so all three replicates per condition share one confirmed model digest
(see the provenance discussion in the replication-feasibility audit this
script implements).

Reuses cached artefacts only (fsi_enriched.jsonl, phase3_answers.json,
config_loader.load_city_config / OLLAMA) exactly as run_phase5.py and
run_phase5_b1.py do. Does NOT call run_phase3(), run_phase5(), query_all(),
build_index(), or anything under src/phase_3 — those modules are never
imported and GraphRAG is therefore structurally unreachable from this
script. Does NOT call render_html/save_html/export_pdf — only the raw
`prose` dict synthesize_all() returns is persisted, as JSON, under
evaluation/results/b1_replication/. Nothing under data/<city>/ is ever
opened for writing; every write goes through _assert_safe_output_path(),
which raises before any write that would resolve outside
evaluation/results/b1_replication/ or inside data/.

Usage:
    python evaluation/run_b1_replication.py --dry-run
    python evaluation/run_b1_replication.py
    python evaluation/run_b1_replication.py --overwrite-replicates

Do NOT add a seed, change temperature, change num_ctx, or change the prompt
template — this experiment's validity depends on being the *same* Phase-5
generation procedure the original pipeline uses, run repeatedly.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

CITIES = ["barcelona", "brighton", "dublin", "london", "milan"]
CONDITIONS = ["full", "b1"]
N_REPLICATES = 3

DATA_ROOT = (ROOT / "data").resolve()
OUTPUT_ROOT = (ROOT / "evaluation" / "results" / "b1_replication").resolve()

# ── Module-load-time safety assertions (defence-in-depth #1) ──────────────
# These run the instant this file is imported/executed, before argparse,
# before any city loop, before any import of pipeline code.
assert OUTPUT_ROOT == (ROOT / "evaluation" / "results" / "b1_replication").resolve(), \
    "OUTPUT_ROOT must be exactly evaluation/results/b1_replication/"
assert DATA_ROOT not in OUTPUT_ROOT.parents and OUTPUT_ROOT != DATA_ROOT, \
    "OUTPUT_ROOT must not resolve inside data/"
assert str(OUTPUT_ROOT).startswith(str(ROOT)), \
    "OUTPUT_ROOT must be inside the repository root"


def _assert_safe_output_path(path: Path) -> Path:
    """Defence-in-depth #2 (runtime, survives `python -O`): every single
    write in this script passes through here first. Raises RuntimeError —
    not a bare `assert` — so it cannot be stripped by -O and always aborts
    the write rather than silently allowing it."""
    resolved = path.resolve()
    if resolved == DATA_ROOT or DATA_ROOT in resolved.parents:
        raise RuntimeError(
            f"SAFETY ABORT: refusing to write under data/: {resolved}"
        )
    if resolved != OUTPUT_ROOT and OUTPUT_ROOT not in resolved.parents:
        raise RuntimeError(
            f"SAFETY ABORT: output path is not under {OUTPUT_ROOT}: {resolved}"
        )
    if resolved.suffix not in (".json",):
        raise RuntimeError(
            f"SAFETY ABORT: this script only ever writes .json files, got: {resolved}"
        )
    return resolved


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_model_digest(model_name: str) -> str | None:
    """Read-only metadata query against the local Ollama daemon (`ollama
    list`) — NOT a generation call. Never pulls or modifies a model."""
    try:
        out = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=15
        )
        for line in out.stdout.splitlines():
            if line.strip().startswith(model_name):
                parts = line.split()
                # NAME  ID  SIZE  MODIFIED...
                if len(parts) >= 2:
                    return parts[1]
        return None
    except Exception as e:
        print(f"  [warn] could not query ollama list for digest: {e}")
        return None


def _get_ollama_version() -> str | None:
    try:
        out = subprocess.run(
            ["ollama", "--version"], capture_output=True, text=True, timeout=15
        )
        return out.stdout.strip() or out.stderr.strip() or None
    except Exception as e:
        print(f"  [warn] could not query ollama --version: {e}")
        return None


def _get_git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=str(ROOT), timeout=15
        )
        return out.stdout.strip() or None
    except Exception as e:
        print(f"  [warn] could not query git rev-parse HEAD: {e}")
        return None


def _plan():
    """Builds the full read/write plan without touching any pipeline code,
    Ollama, or the filesystem beyond stat()/existence checks. Safe to call
    for both --dry-run and the real run (the real run re-derives the same
    plan, it doesn't reuse a cached one, so there's no drift risk)."""
    # Imported lazily, inside _plan(), so that --dry-run's import graph is
    # visibly identical to the real run's — no special-cased "dry run
    # imports" that could hide a real-run-only import of something unsafe.
    sys.path.insert(0, str(SRC))
    from config_loader import load_city_config, OLLAMA  # noqa: E402

    plan = {"cities": {}}
    for city in CITIES:
        cfg = load_city_config(city)
        fsi_enriched = Path(cfg["fsi_enriched"])
        phase3_answers = Path(cfg["phase3_answers"])
        if not fsi_enriched.exists():
            raise FileNotFoundError(f"[{city}] missing fsi_enriched.jsonl: {fsi_enriched}")
        if not phase3_answers.exists():
            raise FileNotFoundError(f"[{city}] missing phase3_answers.json: {phase3_answers}")

        out_dir = OUTPUT_ROOT / city
        replicate_files = {}
        for condition in CONDITIONS:
            for rep in range(1, N_REPLICATES + 1):
                key = f"{condition}_rep{rep}"
                replicate_files[key] = out_dir / f"{key}.json"

        plan["cities"][city] = {
            "cfg": cfg,
            "fsi_enriched": fsi_enriched,
            "phase3_answers": phase3_answers,
            "out_dir": out_dir,
            "replicate_files": replicate_files,
        }
    plan["manifest_path"] = OUTPUT_ROOT / "manifest.json"
    plan["ollama"] = dict(OLLAMA)
    return plan


def _print_dry_run(plan: dict):
    print("=" * 70)
    print("DRY RUN — no Ollama generation, no filesystem writes will occur")
    print("=" * 70)

    print("\n--- Model / generation config (static, from config_loader.OLLAMA) ---")
    for k, v in plan["ollama"].items():
        print(f"  {k}: {v}")
    print("  temperature: 0 (hardcoded in text_synthesizer._synthesize, unchanged)")

    print("\n--- Live environment metadata (read-only queries, no generation) ---")
    digest = _get_model_digest(plan["ollama"]["model"])
    print(f"  qwen3:14b digest (ollama list): {digest!r}")
    print(f"  ollama --version: {_get_ollama_version()!r}")
    print(f"  git rev-parse HEAD: {_get_git_commit()!r}")
    print(f"  python: {sys.version.split()[0]}")

    print("\n--- Inputs that would be READ (read-only) ---")
    for city, c in plan["cities"].items():
        print(f"  [{city}]")
        print(f"    {c['fsi_enriched']}")
        print(f"    {c['phase3_answers']}  (Full: passed to synthesize_all; "
              f"B1: hashed for provenance only, passed as graph_answers={{}})")

    print("\n--- Outputs that WOULD be created (none created in --dry-run) ---")
    n_files = 0
    for city, c in plan["cities"].items():
        for key, path in sorted(c["replicate_files"].items()):
            print(f"  {path}")
            n_files += 1
    print(f"  {plan['manifest_path']}")
    n_files += 1
    print(f"  Total planned output files: {n_files}")

    print("\n--- Output-root safety check ---")
    print(f"  OUTPUT_ROOT = {OUTPUT_ROOT}")
    print(f"  DATA_ROOT   = {DATA_ROOT}")
    for city, c in plan["cities"].items():
        for key, path in c["replicate_files"].items():
            resolved = path.resolve()
            under_data = resolved == DATA_ROOT or DATA_ROOT in resolved.parents
            print(f"  {path} -> under data/? {under_data}")
            assert not under_data, "dry-run safety check itself failed — aborting"
    print("  CONFIRMED: no planned output path resolves under data/")

    n_calls = len(CITIES) * len(CONDITIONS) * N_REPLICATES * 6
    print("\n--- Planned Ollama generation calls (none made in --dry-run) ---")
    print(f"  {len(CITIES)} cities x {len(CONDITIONS)} conditions x "
          f"{N_REPLICATES} replicates x 6 sections = {n_calls}")

    print("\nDry run complete. No files were written, no Ollama generation occurred.")


def _run_real(plan: dict, overwrite_replicates: bool):
    sys.path.insert(0, str(SRC / "phase_5"))
    sys.path.insert(0, str(SRC))
    from config_loader import OLLAMA  # noqa: E402
    from data_extractor import extract_facts  # noqa: E402
    from text_synthesizer import synthesize_all  # noqa: E402

    # ── Pre-generation: hash every read-only source artefact ──────────────
    pre_hashes: dict[str, dict[str, str]] = {}
    for city, c in plan["cities"].items():
        pre_hashes[city] = {
            "fsi_enriched_sha256": _sha256(c["fsi_enriched"]),
            "phase3_answers_sha256": _sha256(c["phase3_answers"]),
        }

    model_digest = _get_model_digest(plan["ollama"]["model"])
    ollama_version = _get_ollama_version()
    git_commit = _get_git_commit()
    run_started_at = datetime.now(timezone.utc).isoformat()

    created_files = []

    for city, c in plan["cities"].items():
        cfg = c["cfg"]
        records = [
            json.loads(l) for l in c["fsi_enriched"].read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        phase3_answers = json.loads(c["phase3_answers"].read_text(encoding="utf-8"))
        facts = extract_facts(records)
        language = cfg.get("language", "en")

        out_dir = _assert_safe_output_path(c["out_dir"] / "_dir_marker.json").parent
        out_dir.mkdir(parents=True, exist_ok=True)

        for condition in CONDITIONS:
            graph_answers = phase3_answers if condition == "full" else {}
            for rep in range(1, N_REPLICATES + 1):
                out_path = _assert_safe_output_path(c["replicate_files"][f"{condition}_rep{rep}"])

                if out_path.exists() and not overwrite_replicates:
                    print(f"  [skip] {out_path} exists (use --overwrite-replicates to replace)")
                    continue

                print(f"  [{city}] {condition} replicate {rep}/{N_REPLICATES} — synthesizing 6 sections...")
                t0 = datetime.now(timezone.utc).isoformat()
                prose = synthesize_all(
                    facts, graph_answers, OLLAMA,
                    cfg["city"], cfg["country"], language=language,
                )
                t1 = datetime.now(timezone.utc).isoformat()

                record = {
                    "city": city,
                    "condition": condition,
                    "replicate": rep,
                    "model": OLLAMA["model"],
                    "model_digest": model_digest,
                    "temperature": 0,
                    "num_ctx": OLLAMA["num_ctx"],
                    "language": language,
                    "timestamp_start": t0,
                    "timestamp_end": t1,
                    "source_fsi_enriched_sha256": pre_hashes[city]["fsi_enriched_sha256"],
                    "source_phase3_answers_sha256": pre_hashes[city]["phase3_answers_sha256"],
                    "prose": prose,
                }
                out_path.write_text(
                    json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                created_files.append(str(out_path.relative_to(ROOT)))
                print(f"    -> {out_path}")

    # ── Post-generation: recompute source hashes, abort loudly on drift ───
    post_hashes: dict[str, dict[str, str]] = {}
    changed = []
    for city, c in plan["cities"].items():
        post_hashes[city] = {
            "fsi_enriched_sha256": _sha256(c["fsi_enriched"]),
            "phase3_answers_sha256": _sha256(c["phase3_answers"]),
        }
        for key in ("fsi_enriched_sha256", "phase3_answers_sha256"):
            if pre_hashes[city][key] != post_hashes[city][key]:
                changed.append(f"{city}:{key}")

    integrity_status = "OK"
    if changed:
        integrity_status = "FAILED"
        print("\n" + "!" * 70)
        print("INTEGRITY CHECK FAILED — source artefacts changed during generation:")
        for c in changed:
            print(f"  {c}")
        print("This means something modified a Phase-0-4 artefact while this")
        print("script was running. The generated replicates above may not be")
        print("reproducible against the inputs recorded in this manifest.")
        print("!" * 70)

    manifest_path = _assert_safe_output_path(plan["manifest_path"])
    manifest = {
        "created_at": run_started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "git_commit": git_commit,
        "ollama_version": ollama_version,
        "model": OLLAMA["model"],
        "model_digest": model_digest,
        "num_ctx": OLLAMA["num_ctx"],
        "temperature": 0,
        "cities": CITIES,
        "n_replicates_per_condition": N_REPLICATES,
        "conditions": CONDITIONS,
        "input_hashes_pre_generation": pre_hashes,
        "input_hashes_post_generation": post_hashes,
        "integrity_check": integrity_status,
        "integrity_changed_files": changed,
        "created_files": created_files,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nManifest written -> {manifest_path}")

    if integrity_status != "OK":
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Phase-5 Full-vs-B1 replication/stability experiment (additive, read-only inputs)."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the full plan; perform no writes and no Ollama generation.")
    parser.add_argument("--overwrite-replicates", action="store_true",
                        help="Allow overwriting an existing replicate JSON. Off by default.")
    args = parser.parse_args()

    plan = _plan()

    if args.dry_run:
        _print_dry_run(plan)
        return

    _run_real(plan, overwrite_replicates=args.overwrite_replicates)


if __name__ == "__main__":
    main()
