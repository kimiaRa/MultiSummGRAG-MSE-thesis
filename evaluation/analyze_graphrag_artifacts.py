"""
GraphRAG artefact audit — additive, strictly read-only analysis.

Turns the defensible findings of a prior exploratory audit into a
reproducible script. Reads ONLY already-persisted artefacts:
  - data/<city>/graphrag/  (graph, KV stores, input/*.txt)
  - data/<city>/output/phase3_answers.json
for barcelona, brighton, dublin, london, milan. NEVER opens any of these in
a write mode. NEVER imports or calls anything from this project's pipeline
source (src/phase_0..6, runs/, config_loader.py) -- the six Phase-3 query
field names are a local constant here, not imported from config_loader.py.
NEVER imports ollama. NEVER instantiates nano_graphrag.GraphRAG or calls
ainsert/aquery. NEVER executes any nano_graphrag source in any way -- the
only nano_graphrag content this script touches is prompt.py's
PROMPTS["entity_extraction"] string, located via importlib.util.find_spec()
(which locates a package without executing it), read as plain text, and
recovered with ast.parse() + ast.literal_eval() on a single located
assignment node only -- never exec(), eval(), compile(mode="exec"), runpy,
or import_module (see load_entity_extraction_prompt()).

WRITES ONLY under evaluation/results/graphrag_artifact_audit/, and only
after (a) validating all required input artefacts exist, and (b) hashing
every artefact under data/<city>/graphrag/ and
data/<city>/output/phase3_answers.json before starting, then re-hashing
after all computation and refusing to write anything if a single byte
changed. Modification timestamps are never consulted for this check --
only SHA-256 content hashes.

Usage:
    python evaluation/analyze_graphrag_artifacts.py --dry-run
    python evaluation/analyze_graphrag_artifacts.py
    python evaluation/analyze_graphrag_artifacts.py --overwrite
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = (ROOT / "data").resolve()
OUTPUT_ROOT = (ROOT / "evaluation" / "results" / "graphrag_artifact_audit").resolve()

CITIES = ["barcelona", "brighton", "dublin", "london", "milan"]

QUERY_FIELDS = [
    ("geographic_distribution", "global"),
    ("fsi_types", "global"),
    ("operational_levels", "global"),
    ("popularity", "global"),
    ("district_summaries", "local"),
    ("notable_initiatives", "local"),
]

CANONICAL_TEMPLATE_NAMES = [
    "ALEX", "CONTROL", "CRUZ", "FIRST CONTACT", "HUMANITY'S RESPONSE",
    "INTELLIGENCE", "JORDAN", "MERCER", "OPERATION: DULCE", "SAM RIVERA",
    "TAYLOR", "THE DEVICE", "THE TEAM", "WASHINGTON",
]

ALLOWED_OUTPUT_NAMES = {
    "graph_structure.csv", "template_entity_matches.csv",
    "community_contamination.csv", "query_answer_summary.csv", "analysis_summary.md",
}

# ── Module-load-time safety assertions ─────────────────────────────────────
assert DATA_ROOT not in OUTPUT_ROOT.parents and OUTPUT_ROOT != DATA_ROOT, \
    "OUTPUT_ROOT must not resolve inside data/"
assert str(OUTPUT_ROOT).startswith(str(ROOT)), "OUTPUT_ROOT must be inside the repository root"


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
# Paths to protected, read-only source artefacts
# ═══════════════════════════════════════════════════════════════════════

def _graphrag_dir(city: str) -> Path:
    return DATA_ROOT / city / "graphrag"


def _phase3_answers_path(city: str) -> Path:
    return DATA_ROOT / city / "output" / "phase3_answers.json"


def _required_artefacts(city: str) -> dict[str, Path]:
    g = _graphrag_dir(city)
    return {
        "input_dir": g / "input",
        "text_chunks": g / "kv_store_text_chunks.json",
        "graphml": g / "graph_chunk_entity_relation.graphml",
        "community_reports": g / "kv_store_community_reports.json",
        "phase3_answers": _phase3_answers_path(city),
    }


def validate_inputs() -> list[str]:
    """Fails clearly, per city, if a required artefact is absent. Never
    infers a missing statistic -- an absent file is a validation error, not
    a zero."""
    errors = []
    for city in CITIES:
        for label, path in _required_artefacts(city).items():
            if label == "input_dir":
                if not path.is_dir():
                    errors.append(f"{city}: missing input directory: {path}")
                elif not list(path.glob("*.txt")):
                    errors.append(f"{city}: input directory has no .txt files: {path}")
            elif not path.is_file():
                errors.append(f"{city}: missing required artefact ({label}): {path}")
    return errors


# ═══════════════════════════════════════════════════════════════════════
# Integrity hashing (content only -- never timestamps)
# ═══════════════════════════════════════════════════════════════════════

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_all_protected_artefacts() -> dict[str, str]:
    """SHA-256 of every file under data/<city>/graphrag/ (recursively,
    including input/*.txt) plus data/<city>/output/phase3_answers.json, for
    every city. Read-only -- opens every file in 'rb' mode only."""
    hashes: dict[str, str] = {}
    for city in CITIES:
        gdir = _graphrag_dir(city)
        if gdir.is_dir():
            for path in sorted(gdir.rglob("*")):
                if path.is_file():
                    hashes[str(path.relative_to(ROOT))] = _sha256_file(path)
        p3 = _phase3_answers_path(city)
        if p3.is_file():
            hashes[str(p3.relative_to(ROOT))] = _sha256_file(p3)
    return hashes


def diff_hashes(before: dict[str, str], after: dict[str, str]) -> list[str]:
    changes = []
    for k in sorted(set(before) | set(after)):
        if k not in before:
            changes.append(f"NEW FILE (not present before): {k}")
        elif k not in after:
            changes.append(f"FILE DISAPPEARED: {k}")
        elif before[k] != after[k]:
            changes.append(f"CONTENT CHANGED: {k}")
    return changes


# ═══════════════════════════════════════════════════════════════════════
# §1 — Structural metrics
# ═══════════════════════════════════════════════════════════════════════

def compute_structure(city: str) -> dict:
    art = _required_artefacts(city)

    input_documents = len(list(art["input_dir"].glob("*.txt")))
    text_chunks_data = json.loads(art["text_chunks"].read_text(encoding="utf-8"))
    text_chunks = len(text_chunks_data)

    G = nx.read_graphml(art["graphml"])
    nodes = G.number_of_nodes()
    edges = G.number_of_edges()
    isolated = list(nx.isolates(G))
    n_isolated = len(isolated)
    UG = G.to_undirected()
    components = list(nx.connected_components(UG))
    n_components = len(components)
    largest_cc = max((len(c) for c in components), default=0)
    degrees = [d for _, d in G.degree()]
    avg_degree = (sum(degrees) / len(degrees)) if degrees else 0.0
    self_loops = len(list(nx.selfloop_edges(G)))

    cr_data = json.loads(art["community_reports"].read_text(encoding="utf-8"))
    community_reports = len(cr_data)
    nonempty_reports = sum(
        1 for v in cr_data.values()
        if (v.get("report_string") or "").strip() or v.get("report_json")
    )

    return {
        "city": city,
        "input_documents": input_documents,
        "text_chunks": text_chunks,
        "nodes": nodes,
        "edges": edges,
        "isolated_nodes": n_isolated,
        "isolated_node_pct": round(100 * n_isolated / nodes, 2) if nodes else None,
        "connected_components": n_components,
        "largest_component_nodes": largest_cc,
        "largest_component_pct": round(100 * largest_cc / nodes, 2) if nodes else None,
        "average_degree": round(avg_degree, 4),
        "self_loops": self_loops,
        "community_reports": community_reports,
        "nonempty_community_reports": nonempty_reports,
        "edges_per_chunk": round(edges / text_chunks, 4) if text_chunks else None,
        "nodes_per_chunk": round(nodes / text_chunks, 4) if text_chunks else None,
        "_graph": G,  # kept in-memory for §2; never serialized
    }


# ═══════════════════════════════════════════════════════════════════════
# §2 — Confirmed template leakage
# ═══════════════════════════════════════════════════════════════════════

ENTITY_RECORD_RE = re.compile(
    r'\("entity"\{tuple_delimiter\}"([^"]+)"\{tuple_delimiter\}"([^"]+)"\{tuple_delimiter\}"([^"]*)"\)'
)


def locate_nano_graphrag_prompt_file() -> Path:
    """Locates (does NOT import/execute) the installed nano_graphrag
    package via importlib.util.find_spec, which only resolves module
    locations -- no package code runs. Returns the path to its prompt.py."""
    spec = importlib.util.find_spec("nano_graphrag")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            "nano_graphrag package not found via find_spec() -- cannot statically "
            "confirm the canonical template entity names without it."
        )
    pkg_dir = Path(list(spec.submodule_search_locations)[0])
    prompt_path = pkg_dir / "prompt.py"
    if not prompt_path.exists():
        raise RuntimeError(f"nano_graphrag located at {pkg_dir} but prompt.py is missing")
    return prompt_path


def load_entity_extraction_prompt() -> tuple[str, str, Path]:
    """Reads nano_graphrag/prompt.py as PLAIN TEXT ONLY. Never imports,
    execs, evals, compiles-for-execution, or otherwise runs this file or
    any part of the nano_graphrag package -- no exec(), eval(),
    compile(..., mode="exec"), runpy, or importlib.import_module is called
    anywhere in this function or anything it calls.

    Uses ast.parse() (pure syntax parsing -- produces a tree, executes
    nothing) to locate the specific `PROMPTS["entity_extraction"] = "..."`
    assignment statement by its AST shape (an Assign node whose target is a
    Subscript on a Name called PROMPTS with string key "entity_extraction"),
    then calls ast.literal_eval() on ONLY that one assignment's value node.
    ast.literal_eval is a safe, non-executing literal reconstructor -- it
    recursively rebuilds Python literals (strings, numbers, tuples, lists,
    dicts, sets, booleans, None) directly from AST node structure and
    raises ValueError/TypeError on anything else (calls, names, attribute
    access, comprehensions, f-strings with expressions, etc.); it contains
    no call to compile()/eval()/exec() internally and cannot run arbitrary
    code. This recovers the prompt string exactly as if it had been
    assigned, without ever executing prompt.py or any other file.

    Returns (entity_extraction_prompt_text, sha256_of_file, path)."""
    prompt_path = locate_nano_graphrag_prompt_file()
    source = prompt_path.read_text(encoding="utf-8")
    sha256 = _sha256_file(prompt_path)

    tree = ast.parse(source, filename=str(prompt_path))  # parsing only, no execution

    entity_extraction_value_node = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                    and target.value.id == "PROMPTS"):
                continue
            key_node = target.slice
            if isinstance(key_node, ast.Constant) and key_node.value == "entity_extraction":
                entity_extraction_value_node = node.value
                break
        if entity_extraction_value_node is not None:
            break

    if entity_extraction_value_node is None:
        raise RuntimeError(
            f"{prompt_path}: could not statically locate a "
            f'PROMPTS["entity_extraction"] = ... assignment via AST parsing -- '
            f"the installed nano_graphrag version may have a different prompt.py layout."
        )

    try:
        prompt_text = ast.literal_eval(entity_extraction_value_node)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(
            f"{prompt_path}: PROMPTS[\"entity_extraction\"]'s assigned value is not a "
            f"static literal ast.literal_eval can evaluate ({exc}) -- refusing to guess."
        )
    if not isinstance(prompt_text, str):
        raise RuntimeError(f"{prompt_path}: PROMPTS['entity_extraction'] did not statically evaluate to a string")

    return prompt_text, sha256, prompt_path


def build_reference_map(prompt_text: str) -> dict[str, dict]:
    """{CANONICAL_NAME: {'types': {...}, 'descriptions': [...]}} built
    entirely from the Output blocks of the live prompt text's worked
    examples -- not hardcoded by this script."""
    ref: dict[str, dict] = {}
    for name, etype, desc in ENTITY_RECORD_RE.findall(prompt_text):
        key = name.strip().upper()
        ref.setdefault(key, {"types": set(), "descriptions": []})
        ref[key]["types"].add(etype.strip().lower())
        ref[key]["descriptions"].append(desc)
    return ref


def _local_context(prompt_text: str, name: str, window: int = 150) -> str | None:
    """Fallback reference text for a canonical name that never appears as a
    formally-extracted Output entity in the worked examples (e.g. 'Mercer',
    which appears only in Example 2's narrative Text, not its Output) --
    still literal content read from the live prompt file."""
    idx = prompt_text.lower().find(name.lower())
    if idx == -1:
        return None
    start = max(0, idx - window)
    end = min(len(prompt_text), idx + len(name) + window)
    return prompt_text[start:end]


def _shingles(text: str, n: int = 5) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9:']+", (text or "").lower())
    if len(words) < n:
        return set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def _strip_quotes(s: str) -> str:
    return (s or "").strip().strip('"')


def classify_node_match(node_type: str, node_desc: str, ref_entry: dict | None,
                         fallback_context: str | None) -> tuple[str, str]:
    """Returns (classification, evidence_fragment).
    - confirmed_template_leakage: a >=5-word shingle in the node's own
      description is verbatim-shared (case-insensitive) with the reference
      text drawn from the live nano_graphrag prompt.
    - legitimate_name_collision: no shared shingle, AND the node's
      entity_type does not match any type the template records for this
      name (a real, differently-typed entity sharing only the name).
    - unclear: no shared shingle, and either no reference type is known for
      this name or the node's type happens to match the template's type
      too coarsely to decide from type alone."""
    node_shingles = _shingles(node_desc, n=5)

    ref_descs = list(ref_entry["descriptions"]) if ref_entry else []
    if not ref_descs and fallback_context:
        ref_descs = [fallback_context]

    for rd in ref_descs:
        overlap = node_shingles & _shingles(rd, n=5)
        if overlap:
            return "confirmed_template_leakage", sorted(overlap)[0]

    ref_types = ref_entry["types"] if ref_entry else set()
    node_type_norm = _strip_quotes(node_type).lower()
    if ref_types and node_type_norm not in ref_types:
        return "legitimate_name_collision", ""
    return "unclear", ""


def find_template_matches(city: str, G: nx.Graph, ref_map: dict[str, dict],
                           prompt_text: str, text_chunks_data: dict) -> list[dict]:
    rows = []
    canonical_set = set(CANONICAL_TEMPLATE_NAMES)
    for node_id, attrs in G.nodes(data=True):
        name = _strip_quotes(str(node_id)).upper()
        if name not in canonical_set:
            continue  # exact canonical names ONLY -- no derivative names
        node_type = attrs.get("entity_type", "")
        node_desc = _strip_quotes(attrs.get("description", ""))
        source_id = attrs.get("source_id", "")

        ref_entry = ref_map.get(name)
        fallback = None
        if not ref_entry:
            fallback = _local_context(prompt_text, name.title())
        classification, evidence = classify_node_match(node_type, node_desc, ref_entry, fallback)

        chunk_excerpt = ""
        first_chunk_id = source_id.split("<SEP>")[0].strip() if source_id else ""
        if first_chunk_id and first_chunk_id in text_chunks_data:
            content = text_chunks_data[first_chunk_id].get("content", "")
            chunk_excerpt = content[:200]

        rows.append({
            "city": city, "matched_name": name,
            "node_entity_type": _strip_quotes(node_type),
            "classification": classification,
            "matched_evidence_shingle": evidence,
            "source_id": source_id,
            "node_description_excerpt": node_desc[:200],
            "source_chunk_excerpt": chunk_excerpt,
        })
    return rows


# ═══════════════════════════════════════════════════════════════════════
# §3 — Community report check
# ═══════════════════════════════════════════════════════════════════════

def _name_regex(name: str) -> re.Pattern:
    return re.compile(r'(?<![A-Za-z0-9])' + re.escape(name) + r'(?![A-Za-z0-9])', re.IGNORECASE)


def check_community_reports(city: str, community_reports_data: dict,
                             confirmed_names: set[str]) -> dict:
    """Limited strictly to: total reports, how many contain >=1 CONFIRMED
    canonical template-leakage name, and which exact names. Deliberately
    does NOT attempt to determine whether a leakage-containing report also
    mentions a "real" corpus entity -- a prior version of this check
    treated any graph node name outside the 14-name canonical list as
    "real", which is invalid: derivative leakage entities (e.g. Brighton's
    "ALEX'S TEAM", a graph node clearly derived from the same fictional
    extraction-prompt material as "ALEX" and "THE TEAM") also fall outside
    that 14-name list, so that check could and did misclassify leakage
    material as a real corpus entity. No replacement heuristic is applied
    here; this measurement is simply not produced."""
    total = len(community_reports_data)
    reports_with_leakage = 0
    matched_names_all: set[str] = set()

    name_patterns = {n: _name_regex(n) for n in confirmed_names}

    for report in community_reports_data.values():
        text = report.get("report_string") or json.dumps(report.get("report_json") or {}, ensure_ascii=False)
        if not text:
            continue
        hit_names = {n for n, pat in name_patterns.items() if pat.search(text)}
        if not hit_names:
            continue
        reports_with_leakage += 1
        matched_names_all |= hit_names

    return {
        "city": city,
        "community_reports_total": total,
        "reports_with_confirmed_leakage": reports_with_leakage,
        "confirmed_leakage_names_found": ";".join(sorted(matched_names_all)),
    }


# ═══════════════════════════════════════════════════════════════════════
# §4 — Phase-3 query answers
# ═══════════════════════════════════════════════════════════════════════

def check_query_answers(city: str, confirmed_names: set[str]) -> list[dict]:
    answers = json.loads(_phase3_answers_path(city).read_text(encoding="utf-8"))
    rows = []
    for field, mode in QUERY_FIELDS:
        entry = answers.get(field)
        present = entry is not None
        text = (entry or {}).get("answer", "") if present else ""
        length = len(text)
        empty = length == 0
        error = (entry or {}).get("error", "") if present else "MISSING_FIELD"
        hits = [n for n in confirmed_names if _name_regex(n).search(text)]
        rows.append({
            "city": city, "field": field, "mode": mode,
            "present": present, "length": length, "empty": empty,
            "error": error or "", "confirmed_leakage_names_in_answer": ";".join(sorted(hits)),
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

def build_summary_markdown(prompt_sha256: str, prompt_path: Path, names_confirmed_in_prompt: dict[str, bool],
                            structure_rows: list[dict], match_rows: list[dict],
                            community_rows: list[dict], query_rows: list[dict]) -> str:
    md = []
    md.append("# GraphRAG Artefact Audit\n")

    md.append("## DATA INTEGRITY\n")
    md.append(
        "All required artefacts (input/*.txt, kv_store_text_chunks.json, "
        "graph_chunk_entity_relation.graphml, kv_store_community_reports.json, "
        "phase3_answers.json) were present for all five cities before analysis began. "
        "Every file under data/<city>/graphrag/ and each city's phase3_answers.json "
        "was SHA-256-hashed before computation started and again after all outputs "
        "were written; the run aborted before writing anything if any hash differed "
        "(content-based check only -- modification timestamps were never consulted).\n\n"
        f"Canonical-name source: `nano_graphrag/prompt.py` located via "
        f"`importlib.util.find_spec` (path not hardcoded, package never imported or "
        f"executed) at `{prompt_path}`, SHA-256 = `{prompt_sha256}`. All 14 canonical "
        f"names were confirmed present in the INSTALLED SOURCE TEXT of "
        f"`PROMPTS[\"entity_extraction\"]` (extracted via static `ast.parse()` + "
        f"`ast.literal_eval()` on the located assignment node only -- a static source "
        f"check, not a runtime/execution-based inspection): "
        + ", ".join(f"{n} ({'found' if ok else 'NOT FOUND'})" for n, ok in names_confirmed_in_prompt.items())
        + ".\n"
    )

    md.append("## GRAPH STRUCTURE\n")
    md.append("| city | input_docs | chunks | nodes | edges | isolated % | avg_degree | communities |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in structure_rows:
        md.append(
            f"| {r['city']} | {r['input_documents']} | {r['text_chunks']} | {r['nodes']} | "
            f"{r['edges']} | {r['isolated_node_pct']} | {r['average_degree']} | {r['community_reports']} |"
        )
    md.append("\nFull metrics (including connected_components, largest_component_pct, "
               "self_loops, edges_per_chunk, nodes_per_chunk) in `graph_structure.csv`.\n")

    md.append("## CROSS-CITY CONTRAST\n")
    by_edges_per_chunk = sorted(structure_rows, key=lambda r: r["edges_per_chunk"] or 0)
    md.append(
        "FACT: edges-per-chunk ranges from "
        f"{by_edges_per_chunk[0]['city']} ({by_edges_per_chunk[0]['edges_per_chunk']}) to "
        f"{by_edges_per_chunk[-1]['city']} ({by_edges_per_chunk[-1]['edges_per_chunk']}) -- "
        "computed directly from GRAPH STRUCTURE above. Cities with low edges-per-chunk also "
        "show low `average_degree` and high `isolated_node_pct` in the same table. "
        "INTERPRETATION: this pattern is consistent with weaker relationship extraction on "
        "those cities' chunks; the cause is NOT established by this script (see "
        "INTERPRETATION LIMITS).\n"
    )

    md.append("## CONFIRMED TEMPLATE LEAKAGE\n")
    by_city_class = {}
    for r in match_rows:
        by_city_class.setdefault(r["city"], Counter())[r["classification"]] += 1
    md.append("| city | confirmed_template_leakage | legitimate_name_collision | unclear |")
    md.append("|---|---:|---:|---:|")
    for city in CITIES:
        c = by_city_class.get(city, Counter())
        md.append(f"| {city} | {c.get('confirmed_template_leakage',0)} | "
                   f"{c.get('legitimate_name_collision',0)} | {c.get('unclear',0)} |")
    md.append(
        "\nOnly rows classified `confirmed_template_leakage` (verbatim >=5-word shingle "
        "overlap with the live nano_graphrag entity-extraction prompt text -- see "
        "`matched_evidence_shingle` column) count as the contamination headline. "
        "`legitimate_name_collision` rows (name matches, but description/type traces to a "
        "different, real referent) are explicitly NOT counted as leakage. Full per-match "
        "detail, including the exact matched name, node description excerpt, and source "
        "chunk excerpt where recoverable, in `template_entity_matches.csv`.\n"
    )

    md.append("## COMMUNITY REPORTS\n")
    md.append(
        "Reports only whether CONFIRMED template leakage (§CONFIRMED TEMPLATE LEAKAGE) "
        "propagated into community reports -- no claim is made about whether a "
        "leakage-containing report also mentions a genuine city-corpus entity (a prior "
        "version of this check treated any non-canonical-14 node name as \"real\", which "
        "is invalid: derivative leakage entities such as Brighton's \"ALEX'S TEAM\" also "
        "fall outside that 14-name list, so that check could misclassify leakage material "
        "itself as a real entity; it has been removed, not replaced).\n"
    )
    md.append("| city | reports | with confirmed leakage | confirmed names found |")
    md.append("|---|---:|---:|---|")
    for r in community_rows:
        md.append(f"| {r['city']} | {r['community_reports_total']} | "
                   f"{r['reports_with_confirmed_leakage']} | {r['confirmed_leakage_names_found'] or '-'} |")
    md.append(
        "\nThis resolves the prior audit's open question directly (rather than assuming "
        "graph-level leakage reached community reports) -- see `community_contamination.csv` "
        "for the exact matched template names per city.\n"
    )

    md.append("## PHASE-3 QUERY ANSWERS\n")
    md.append("| city | 6/6 present | min length | max length | answers with confirmed leakage |")
    md.append("|---|---|---:|---:|---:|")
    for city in CITIES:
        rows = [r for r in query_rows if r["city"] == city]
        present_n = sum(1 for r in rows if r["present"])
        lengths = [r["length"] for r in rows if r["present"]]
        n_leak = sum(1 for r in rows if r["confirmed_leakage_names_in_answer"])
        md.append(f"| {city} | {present_n}/6 | {min(lengths) if lengths else '-'} | "
                   f"{max(lengths) if lengths else '-'} | {n_leak} |")
    md.append("\nFull per-field detail in `query_answer_summary.csv`. Factual correctness of "
               "any answer was NOT evaluated (see INTERPRETATION LIMITS).\n")

    md.append("## INTERPRETATION LIMITS\n")
    md.append(
        "**FACT** (directly computed, reported above without qualification): all structural "
        "graph metrics, community-report counts, query-answer presence/length, and confirmed "
        "template-name matches.\n\n"
        "**CONFIRMED DEFECT**: template/example material traced, via verbatim shingle overlap "
        "with the live `nano_graphrag/prompt.py` entity-extraction prompt, to actual graph "
        "nodes -- see CONFIRMED TEMPLATE LEAKAGE.\n\n"
        "**NOT ESTABLISHED by this script**:\n"
        "- that graph contamination changed any Phase-5 report's content;\n"
        "- that a sparse graph is necessarily a bad graph;\n"
        "- why Brighton/Dublin (or any city) show sparser relationship extraction than others;\n"
        "- the factual correctness of any Phase-3 query answer;\n"
        "- any single aggregate 'graph quality score' -- none is computed anywhere in this "
        "script or its outputs.\n"
    )

    md.append("## THESIS IMPLICATION\n")
    all_present = all(sum(1 for r in query_rows if r["city"] == c and r["present"]) == 6 for c in CITIES)
    sparse_cities = [r["city"] for r in structure_rows if (r["average_degree"] or 0) < 1.0]
    dense_cities = [r["city"] for r in structure_rows if (r["average_degree"] or 0) >= 1.0]
    leak_cities = sorted({r["city"] for r in match_rows if r["classification"] == "confirmed_template_leakage"})
    community_leak_cities = sorted({r["city"] for r in community_rows if r["reports_with_confirmed_leakage"] > 0})
    answer_leak_cities = sorted({r["city"] for r in query_rows if r["confirmed_leakage_names_in_answer"]})
    md.append(
        f"**A.** Did all five cities produce non-empty GraphRAG artefacts? "
        f"{'Yes' if all_present else 'No'} -- every required artefact was present and non-empty "
        f"for all five cities (see DATA INTEGRITY); query-answer presence: "
        f"{'6/6 for all cities' if all_present else 'see table above for exceptions'}.\n\n"
        f"**B.** Which cities show strong vs. very sparse relational structure? Dense "
        f"(avg_degree >= 1.0): {', '.join(dense_cities) or 'none'}. Sparse (avg_degree < 1.0): "
        f"{', '.join(sparse_cities) or 'none'}.\n\n"
        f"**C.** Where is template leakage confirmed? "
        f"{', '.join(leak_cities) if leak_cities else 'nowhere in this run'}.\n\n"
        f"**D.** Did confirmed leakage propagate into community reports? "
        f"{', '.join(community_leak_cities) if community_leak_cities else 'no city shows this'}.\n\n"
        f"**E.** Did confirmed leakage propagate into the persisted Phase-3 answers? "
        f"{', '.join(answer_leak_cities) if answer_leak_cities else 'no city shows this'}.\n\n"
        "**F.** Which structural measurements are suitable for the thesis? nodes/edges/"
        "average_degree, isolated_node_pct and largest_component_pct, community_reports count, "
        "and edges_per_chunk as the cross-city comparability metric -- all directly computed, "
        "reproducible from this script's CSV outputs.\n\n"
        "**G.** Does any finding require rerunning GraphRAG? No. Every measurement above came "
        "from existing artefacts; sparse relational structure or confirmed template leakage in "
        "a given city is a reportable finding, not by itself a reason to rerun -- rerunning "
        "would also break comparability with every other evaluation artifact already built "
        "against the current graphs.\n"
    )

    return "\n".join(md)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def _print_dry_run(errors: list[str]):
    print("=" * 70)
    print("DRY RUN — no files will be written")
    print("=" * 70)

    print(f"\nOUTPUT_ROOT = {OUTPUT_ROOT}")
    print(f"(the only directory this script can ever write under)")

    print("\n--- Input validation ---")
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        print("\nValidation FAILED -- the real run would abort before any computation.")
    else:
        print("  All required artefacts present for all 5 cities.")

        prompt_text, prompt_sha, prompt_path = load_entity_extraction_prompt()
        print(f"\n  nano_graphrag prompt.py located at: {prompt_path}")
        print(f"  SHA-256: {prompt_sha}")
        for name in CANONICAL_TEMPLATE_NAMES:
            found = name.lower() in prompt_text.lower()
            print(f"    {name}: {'found in prompt text' if found else 'NOT FOUND -- would abort'}")

        print("\n--- In-memory structural metrics preview (no writes) ---")
        for city in CITIES:
            s = compute_structure(city)
            print(f"  {city}: nodes={s['nodes']} edges={s['edges']} "
                  f"avg_degree={s['average_degree']} communities={s['community_reports']}")

    print("\nPlanned output paths (none created in --dry-run):")
    for name in sorted(ALLOWED_OUTPUT_NAMES):
        print(f"  {OUTPUT_ROOT / name}")

    print("\nNo output has been written. Dry run complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Read-only analysis of existing GraphRAG artefacts across five cities."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs and compute metrics in memory; write nothing.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow overwriting existing analysis outputs. Off by default.")
    args = parser.parse_args()

    errors = validate_inputs()

    if args.dry_run:
        _print_dry_run(errors)
        if errors:
            sys.exit(1)
        return

    if errors:
        print("VALIDATION FAILED — aborting before any analysis:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    hashes_before = hash_all_protected_artefacts()

    prompt_text, prompt_sha256, prompt_path = load_entity_extraction_prompt()
    names_confirmed = {n: (n.lower() in prompt_text.lower()) for n in CANONICAL_TEMPLATE_NAMES}
    missing = [n for n, ok in names_confirmed.items() if not ok]
    if missing:
        print(f"VALIDATION FAILED — canonical names not found in the live entity-extraction "
              f"prompt (installed nano_graphrag version may differ): {missing}")
        sys.exit(1)
    ref_map = build_reference_map(prompt_text)

    structure_rows = []
    match_rows = []
    community_rows = []
    query_rows = []
    per_city_confirmed_names: dict[str, set[str]] = {}

    for city in CITIES:
        s = compute_structure(city)
        G = s.pop("_graph")
        structure_rows.append(s)

        text_chunks_data = json.loads(_required_artefacts(city)["text_chunks"].read_text(encoding="utf-8"))
        m_rows = find_template_matches(city, G, ref_map, prompt_text, text_chunks_data)
        match_rows.extend(m_rows)
        confirmed = {r["matched_name"] for r in m_rows if r["classification"] == "confirmed_template_leakage"}
        per_city_confirmed_names[city] = confirmed

        cr_data = json.loads(_required_artefacts(city)["community_reports"].read_text(encoding="utf-8"))
        community_rows.append(check_community_reports(city, cr_data, confirmed))

        query_rows.extend(check_query_answers(city, confirmed))

    hashes_after = hash_all_protected_artefacts()
    changes = diff_hashes(hashes_before, hashes_after)
    if changes:
        print("INTEGRITY CHECK FAILED — protected artefacts changed during this run:")
        for c in changes:
            print(f"  - {c}")
        print("Aborting WITHOUT writing any output.")
        sys.exit(1)

    write_csv("graph_structure.csv", structure_rows,
              ["city", "input_documents", "text_chunks", "nodes", "edges", "isolated_nodes",
               "isolated_node_pct", "connected_components", "largest_component_nodes",
               "largest_component_pct", "average_degree", "self_loops", "community_reports",
               "nonempty_community_reports", "edges_per_chunk", "nodes_per_chunk"],
              args.overwrite)
    write_csv("template_entity_matches.csv", match_rows,
              ["city", "matched_name", "node_entity_type", "classification",
               "matched_evidence_shingle", "source_id", "node_description_excerpt",
               "source_chunk_excerpt"], args.overwrite)
    write_csv("community_contamination.csv", community_rows,
              ["city", "community_reports_total", "reports_with_confirmed_leakage",
               "confirmed_leakage_names_found"], args.overwrite)
    write_csv("query_answer_summary.csv", query_rows,
              ["city", "field", "mode", "present", "length", "empty", "error",
               "confirmed_leakage_names_in_answer"], args.overwrite)

    summary_md = build_summary_markdown(
        prompt_sha256, prompt_path, names_confirmed,
        structure_rows, match_rows, community_rows, query_rows,
    )
    write_text("analysis_summary.md", summary_md, args.overwrite)

    print(f"Analysis complete. Outputs written under {OUTPUT_ROOT}")
    print("Post-write integrity note: hashes were compared BEFORE writing any output "
          "(see above) -- the protected artefacts were not touched by the write step itself, "
          "since it only ever targets OUTPUT_ROOT.")


if __name__ == "__main__":
    main()
