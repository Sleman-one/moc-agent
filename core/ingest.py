import json
import asyncio
import os
import sys
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

# ── Configuration ────────────────────────────────────────────────────────────

WORKING_DIR = "lightrag_storage"
CHUNKS_PATH = "data/processed/chunks_final_v2.json"

# BGE-M3 always produces 1024-dimensional vectors
EMBEDDING_DIM = 1024

# ── LightRAG setup ───────────────────────────────────────────────────────────


def build_rag() -> LightRAG:
    """
    Create and return a configured LightRAG instance.

    Two functions are passed in:
    - llm_model_func: called during ingestion to extract entities/relations
    - embedding_func: called during ingestion and retrieval to embed text
    """
    os.makedirs(WORKING_DIR, exist_ok=True)

    return LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="qwen2.5:7b",
        llm_model_kwargs={
            "host": "http://localhost:11434",
            "options": {"num_ctx": 4096},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts,
                embed_model="bge-m3",
                host="http://localhost:11434",
            ),
        ),
    )


# ── Ingestion ─────────────────────────────────────────────────────────────────


async def ingest():
    rag = build_rag()
    await rag.initialize_storages()

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)

    total = len(chunks)
    print(f"Loaded {total} chunks — starting ingestion...\n")

    for i, chunk in enumerate(chunks):
        # Prepend the source law to the content so it becomes part of the
        # embedded text. Without this, retrieval cannot distinguish CPL from
        # ECL chunks — both would look identical in vector space on legal topics
        # that appear in both laws.
        text = f"المصدر: {chunk['source_law']}\n{chunk['content']}"

        await rag.ainsert(text)

        law_short = "CPL" if "حماية" in chunk["source_law"] else "ECL"
        print(f"[{i+1:>3}/{total}] {law_short} — {chunk['article_header'][:60]}")

    print("\nIngestion complete.")
    print(f"Storage written to: {WORKING_DIR}/")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(ingest())
