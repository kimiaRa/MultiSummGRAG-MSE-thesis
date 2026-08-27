#!/usr/bin/env python3
"""
enrich_set_c.py — dump the per-city facts dicts and annotate Set C with the
candidate slots each claimed value could have come from.

Run from the repo root:

    venv/bin/python3 evaluation/scripts/enrich_set_c.py

Produces:
    data/<city>/output/facts_<city>.json      one per city (also useful elsewhere)
    evaluation/results/e5_set_c_enriched.csv  Set C + a candidate_slots column

Why: E1 labelled these claims AUDITABLE_DIRECT because the value exists somewhere
in the facts dict. To judge whether the value was used for the RIGHT quantity you
need to know which key(s) hold it. This script does that lookup for you, so
annotation becomes "does the sentence's usage match one of these slots?" rather
than a manual search through JSON.

No network calls. No pipeline stage is run. Read-only except for the two outputs.
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

CITIES = ["barcelona", "brighton", "dublin", "london", "milan"]
SET_C = ROOT / "evaluation" / "results" / "e5_set_c.csv"
OUT = ROOT / "evaluation" / "results" / "e5_set_c_enriched.csv"


def load_facts(city):
    """Build the facts dict for one city, dump it, return it."""
    from phase_5.data_extractor import extract_facts  # noqa: E402

    jsonl = ROOT / "data" / city / "output" / "fsi_enriched.jsonl"
    if not jsonl.exists():
        print(f"  ! {city}: {jsonl} not found — skipping")
        return None

    records = []
    with jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    facts = extract_facts(records)
    dest = ROOT / "data" / city / "output" / f"facts_{city}.json"
    dest.write_text(json.dumps(facts, indent=2, default=str), encoding="utf-8")
    print(f"  {city:10s} {len(records):4d} records -> {dest.relative_to(ROOT)}")
    return facts


def flatten(obj, prefix="", out=None):
    """Recursively map dotted key path -> numeric value."""
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            # keep list entries readable: use the entry's own name if it has one
            label = None
            if isinstance(v, dict):
                for key in ("name", "district", "label", "type", "category"):
                    if key in v and isinstance(v[key], str):
                        label = v[key]
                        break
            flatten(v, f"{prefix}[{label or i}]", out)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        out[prefix] = float(obj)
    return out


def candidates(flat, value, tol=1e-9):
    """Every key path in the facts dict holding this exact value."""
    hits = [k for k, v in flat.items() if abs(v - value) < tol]
    hits.sort(key=len)          # shortest paths first — usually the most general
    return hits


def main():
    if not SET_C.exists():
        sys.exit(f"Set C not found at {SET_C}\n"
                 "Save the e5_set_c.csv you were given to that path first.")

    print("Dumping facts dicts:")
    facts_by_city = {}
    for city in CITIES:
        f = load_facts(city)
        if f is not None:
            facts_by_city[city] = flatten(f)

    rows = list(csv.DictReader(SET_C.open(encoding="utf-8")))
    print(f"\nEnriching {len(rows)} Set C rows...")

    n_none = n_one = n_many = 0
    for r in rows:
        city = (r.get("city") or "").strip().lower()
        flat = facts_by_city.get(city)
        raw = (r.get("raw_number") or "").replace(",", "").replace("%", "").strip()
        try:
            val = float(raw)
        except ValueError:
            r["candidate_slots"] = "(unparseable value)"
            r["n_candidates"] = "0"
            n_none += 1
            continue

        if flat is None:
            r["candidate_slots"] = "(no facts dict for this city)"
            r["n_candidates"] = "0"
            n_none += 1
            continue

        hits = candidates(flat, val)
        r["n_candidates"] = str(len(hits))
        if not hits:
            r["candidate_slots"] = "(value not found — re-check E1 labelling)"
            n_none += 1
        else:
            shown = hits[:8]
            r["candidate_slots"] = " | ".join(f"{k}={val:g}" for k in shown) + \
                                   (f"  (+{len(hits)-8} more)" if len(hits) > 8 else "")
            if len(hits) == 1:
                n_one += 1
            else:
                n_many += 1

    hdr = list(rows[0].keys())
    for c in ("candidate_slots", "n_candidates"):
        if c not in hdr:
            hdr.append(c)
    # put the working columns before the empty annotation columns
    for c in ("MY_LABEL", "MY_NOTE"):
        if c in hdr:
            hdr.remove(c)
            hdr.append(c)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr)
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {OUT.relative_to(ROOT)}")
    print(f"  {n_one:2d} claims match exactly one slot   (usually quick to judge)")
    print(f"  {n_many:2d} claims match several slots     (read the sentence carefully)")
    print(f"  {n_none:2d} claims match no slot           (investigate — see note below)")
    if n_none:
        print("\n  NOTE: a claim labelled AUDITABLE_DIRECT that matches no slot means the\n"
              "  E1 label and this lookup disagree. Check whether E1 matched against a\n"
              "  stringified value this flattener skips. Report any such cases.")
    print("\nNext: load e5_set_c_enriched.csv into the annotation tool and fill MY_LABEL.")


if __name__ == "__main__":
    main()
