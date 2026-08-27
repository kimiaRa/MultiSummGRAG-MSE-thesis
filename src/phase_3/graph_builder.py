import asyncio, logging
import ollama as ollama_client
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import OLLAMA

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)


# ── LLM functions ─────────────────────────────────────────────────────────────

def _make_ollama_llm(model: str, num_ctx: int):
    """Factory that returns an async LLM function bound to a specific model."""
    async def _llm(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
        kwargs.pop("max_tokens", None)
        kwargs.pop("response_format", None)

        hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        args_hash = compute_args_hash(model, messages)
        if hashing_kv is not None:
            cached = await hashing_kv.get_by_id(args_hash)
            if cached is not None:
                return cached["return"]

        client   = ollama_client.AsyncClient()
        response = await client.chat(
            model=model,
            messages=messages,
            options={"temperature": 0, "num_ctx": num_ctx},
        )
        result = response["message"]["content"]

        # strip Qwen3 thinking blocks if present
        if "<think>" in result:
            import re
            result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()

        if hashing_kv is not None:
            await hashing_kv.upsert(
                {args_hash: {"return": result, "model": model}}
            )
        return result

    return _llm


# cheap model: llama3.2-graphrag — fine-tuned for entity/relationship extraction JSON
cheap_llm = _make_ollama_llm(OLLAMA["llm_model"], num_ctx=8192)

# best model: qwen3:14b — better multilingual understanding for community summaries and queries
best_llm  = _make_ollama_llm(OLLAMA["model"], num_ctx=OLLAMA["num_ctx"])


# ── Embedding function ────────────────────────────────────────────────────────

async def ollama_embed(texts: list[str]) -> np.ndarray:
    client = ollama_client.AsyncClient()

    async def _embed_one(text: str) -> list[float]:
        response = await client.embeddings(model=OLLAMA["embed_model"], prompt=text,
                                           options={"num_ctx": 8192})
        return response["embedding"]

    embeddings = await asyncio.gather(*[_embed_one(t) for t in texts])
    return np.array(embeddings, dtype=np.float32)

ollama_embed.embedding_dim = OLLAMA["embed_dim"]


# ── JSON repair ───────────────────────────────────────────────────────────────

def convert_response_to_json(response: str) -> dict:
    import json, re
    response = response.strip()
    # strip thinking blocks
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    response = re.sub(r"```json|```", "", response).strip()
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if match:
        response = match.group()
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {}


# ── GraphRAG factory ──────────────────────────────────────────────────────────

def make_graph_rag(working_dir: str) -> GraphRAG:
    return GraphRAG(
        working_dir=working_dir,
        enable_naive_rag=False,
        best_model_func=best_llm,       # qwen3:14b — used for summaries and queries
        cheap_model_func=cheap_llm,     # llama3.2-graphrag — used for entity extraction
        best_model_max_async=2,         # qwen3:14b is larger; keep concurrency modest
        cheap_model_max_async=6,        # llama3.2-graphrag is fast (2GB); run more in parallel
        embedding_func=ollama_embed,
        embedding_batch_num=8,          # was 4
        embedding_func_max_async=6,     # was 3
        convert_response_to_json_func=convert_response_to_json,
        chunk_token_size=1800,
        chunk_overlap_token_size=100,   # was 50; more overlap = better cross-chunk context
    )


def build_index(documents: list[str], working_dir: str):
    rag = make_graph_rag(working_dir)
    print(f"Indexing {len(documents)} documents — this may take 10–30 minutes...")
    asyncio.run(_insert_all(rag, documents))
    print("✓ Graph index built")
    return rag


async def _insert_all(rag: GraphRAG, documents: list[str]):
    await rag.ainsert(documents)
