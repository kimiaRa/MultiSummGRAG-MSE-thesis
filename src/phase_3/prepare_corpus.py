import json, shutil
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse


def _domain(url: str) -> str:
    return urlparse(url).netloc


def _record_header(record: dict) -> str:
    loc = record.get("location", {})
    pop = record.get("popularity", {})
    return "\n".join([
        f"# {record['title']}",
        f"",
        f"URL: {record['url']}",
        f"Location: {loc.get('country', '')}, {loc.get('city', '')}, {loc.get('district_name', 'Unknown district')}",
        f"Coordinates: {loc.get('lat')}, {loc.get('lng')}",
        f"",
        f"Type: {record.get('fsi_type', 'unknown')}",
        f"Operational level: {record.get('operational_level', 'unknown')}",
        f"Popularity: {pop.get('popularity_level', 'unknown')} (score: {pop.get('popularity_score', 0)})",
        f"",
        f"Description: {record.get('description', '')}",
    ])


def _merge_group(records: list[dict]) -> str:
    """
    Merges all records sharing a domain into one document.
    The record with the most text is used as the primary (its header leads).
    Subpage texts are appended as additional sections.
    """
    primary = max(records, key=lambda r: len(r.get("text", "")))
    others  = [r for r in records if r is not primary]

    sections = [_record_header(primary), "", "Full text:", primary.get("text", "")]

    for r in others:
        text = r.get("text", "").strip()
        if text:
            sections += ["", f"--- Additional page: {r['url']} ---", text]

    imgs = sum(len(r.get("images", [])) for r in records)
    sections.append(f"\nImages: {imgs} image(s) available.")

    return "\n".join(sections)


def prepare_corpus(in_file: Path, graph_dir: Path) -> list[str]:
    """
    Reads fsi_enriched.jsonl, merges records that share a domain into one
    document, and writes one .txt file per FSI into graph_dir/input/.
    Returns list of document texts.
    """
    input_dir = graph_dir / "input"
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    records = [
        json.loads(l)
        for l in in_file.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[_domain(r["url"])].append(r)

    documents = []
    for i, (domain, group) in enumerate(groups.items()):
        doc  = _merge_group(group)
        slug = domain.replace(".", "_")[:60]
        (input_dir / f"{i:03d}_{slug}.txt").write_text(doc, encoding="utf-8")
        documents.append(doc)

    print(f"✓ Prepared {len(documents)} documents from {len(records)} records "
          f"({len(records) - len(documents)} subpages merged) in {input_dir}")
    return documents