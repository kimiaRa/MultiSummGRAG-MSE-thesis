"""
B1 vs pipeline diff — read-only. Extracts the six prose sections from both
report_<city>.html (pipeline) and report_<city>_b1.html (ablation), computes
word counts, a per-section text diff summary, and a numeric-claim diff (every
number/percentage token found in each section's rendered prose).

Writes nothing except printing a JSON blob to stdout (captured by the caller
into the scratchpad) — no project files touched.
"""
import sys
import json
import re
import difflib
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).parent.parent

SECTIONS = [
    ("geographic", "Geographic Distribution"),
    ("types", "Types of Food Sharing Initiatives"),
    ("operational", "Operational and Funding Models"),
    ("reach", "Reach and Activity"),
    ("digital", "Digital Presence and Accessibility"),
    ("notable", "Notable Initiatives"),
]

NUM_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?%?\b")


def extract_sections(html_path: Path) -> dict[str, str]:
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
    out = {}
    for key, title in SECTIONS:
        h2 = soup.find("h2", string=title)
        if h2 is None:
            out[key] = ""
            continue
        paras = []
        for sib in h2.next_siblings:
            if getattr(sib, "name", None) == "h2":
                break
            if getattr(sib, "name", None) == "p":
                paras.append(sib.get_text(" ", strip=True))
        out[key] = "\n\n".join(paras)
    return out


def word_count(text: str) -> int:
    return len(text.split())


def numbers_in(text: str) -> set[str]:
    return set(NUM_RE.findall(text))


def diff_city(city: str) -> dict:
    slug = city.lower().replace(" ", "_").replace("&", "and")
    pipeline_path = ROOT / f"data/{city}/output/report_{slug}.html"
    b1_path = ROOT / f"data/{city}/output/report_{slug}_b1.html"

    pipeline_sections = extract_sections(pipeline_path)
    b1_sections = extract_sections(b1_path)

    result = {"city": city, "sections": {}}
    total_pipeline_words = 0
    total_b1_words = 0

    for key, title in SECTIONS:
        p_text = pipeline_sections.get(key, "")
        b_text = b1_sections.get(key, "")
        p_wc = word_count(p_text)
        b_wc = word_count(b_text)
        total_pipeline_words += p_wc
        total_b1_words += b_wc

        p_nums = numbers_in(p_text)
        b_nums = numbers_in(b_text)

        sm = difflib.SequenceMatcher(None, p_text, b_text)
        similarity = round(sm.ratio(), 4)

        result["sections"][key] = {
            "title": title,
            "pipeline_word_count": p_wc,
            "b1_word_count": b_wc,
            "word_count_delta": b_wc - p_wc,
            "char_similarity_ratio": similarity,
            "identical": p_text.strip() == b_text.strip(),
            "numbers_only_in_pipeline": sorted(p_nums - b_nums),
            "numbers_only_in_b1": sorted(b_nums - p_nums),
            "numbers_in_both": sorted(p_nums & b_nums),
        }

    result["total_pipeline_words"] = total_pipeline_words
    result["total_b1_words"] = total_b1_words
    return result


if __name__ == "__main__":
    cities = sys.argv[1:] or ["barcelona", "brighton", "dublin", "london", "milan"]
    out = {c: diff_city(c) for c in cities}
    print(json.dumps(out, indent=2, ensure_ascii=False))
