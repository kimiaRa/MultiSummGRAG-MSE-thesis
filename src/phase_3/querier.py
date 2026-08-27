import asyncio, json
from pathlib import Path
from nano_graphrag import QueryParam
from graph_builder import make_graph_rag

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import SUMMARY_QUERIES

async def _run_queries(rag, queries: list[dict]) -> dict:
    async def _one(q: dict) -> tuple[str, dict]:
        print(f"  Querying: {q['field']}...")
        try:
            answer = await rag.aquery(
                q["query"],
                param=QueryParam(mode=q["mode"]),
            )
            return q["field"], {"query": q["query"], "mode": q["mode"], "answer": answer}
        except Exception as e:
            print(f"    WARNING: query failed for {q['field']}: {e}")
            return q["field"], {"query": q["query"], "mode": q["mode"],
                                "answer": "", "error": str(e)}

    pairs = await asyncio.gather(*[_one(q) for q in queries])
    return dict(pairs)


def query_all(working_dir: str, out_file: Path) -> dict:
    rag     = make_graph_rag(working_dir)
    results = asyncio.run(_run_queries(rag, SUMMARY_QUERIES))

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"✓ {len(results)} query answers saved to {out_file}")
    return results