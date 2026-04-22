"""
ingest_prod.py — Ingest 108 chunks into LightRAG via vLLM + BGE-M3.

Run:
  cd /workspace && python ingest_prod.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time

import numpy as np

from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc, setup_logger

from rag_config import (
    CHUNKS_PATH,
    EMBEDDING_DIM,
    LLM_MODEL,
    WORKDIR,
    embedding_func,
    llm_model_func,
)

# ------------------------------------------------------------------ #
# Logging
# ------------------------------------------------------------------ #

setup_logger("lightrag", level="INFO")
log = logging.getLogger("ingest")
log.setLevel(logging.INFO)
if not log.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    )
    log.addHandler(h)

# ------------------------------------------------------------------ #
# Smoke test — fail fast if vLLM or BGE-M3 isn't responding
# ------------------------------------------------------------------ #


async def smoke_test() -> None:
    log.info("=== SMOKE TEST ===")

    t0 = time.perf_counter()
    try:
        out = await asyncio.wait_for(
            llm_model_func("Reply with exactly the three characters: OK."),
            timeout=60,
        )
    except asyncio.TimeoutError:
        raise RuntimeError(
            "LLM timed out after 60s. Most likely: thinking mode still on. "
            "Check vLLM logs for <think> tokens."
        )
    log.info("LLM OK in %.2fs. Reply: %r", time.perf_counter() - t0, out[:60])

    t0 = time.perf_counter()
    try:
        emb = await asyncio.wait_for(embedding_func(["مرحبا", "hello"]), timeout=30)
    except asyncio.TimeoutError:
        raise RuntimeError("Embedding timed out after 30s.")

    if emb.shape != (2, EMBEDDING_DIM):
        raise RuntimeError(
            f"Bad embedding shape: {emb.shape}, expected (2, {EMBEDDING_DIM})"
        )
    if np.isnan(emb).any():
        raise RuntimeError("Embedding contains NaN.")
    log.info("Embedding OK in %.2fs.", time.perf_counter() - t0)


# ------------------------------------------------------------------ #
# Chunks + ingestion
# ------------------------------------------------------------------ #


def load_chunks() -> list[dict]:
    data = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    cpl = sum(1 for r in data if "مشروع_نظام_حماية_المستهلك" in r["source_law"])
    ecl = len(data) - cpl
    log.info("Loaded %d chunks (CPL=%d, ECL=%d)", len(data), cpl, ecl)
    return data


async def build_rag() -> LightRAG:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    rag = LightRAG(
        working_dir=str(WORKDIR),
        llm_model_func=llm_model_func,
        llm_model_name=LLM_MODEL,
        llm_model_max_async=4,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=8192,
            func=embedding_func,
        ),
        embedding_func_max_async=8,
        embedding_batch_num=32,
        max_parallel_insert=2,
        chunk_token_size=4096,  # one article per chunk — don't re-split
        chunk_overlap_token_size=0,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    log.info("LightRAG initialised at %s", WORKDIR)
    return rag


async def ingest_all(rag: LightRAG, chunks: list[dict]) -> None:
    """Checkpoint after each chunk so a crash doesn't force a full restart."""
    checkpoint = WORKDIR / "ingest_checkpoint.json"
    done: set[str] = (
        set(json.loads(checkpoint.read_text(encoding="utf-8")))
        if checkpoint.exists()
        else set()
    )
    if done:
        log.info("Resuming — %d chunks already done.", len(done))

    remaining = [c for c in chunks if c["id"] not in done]
    log.info("Ingesting %d / %d chunks.", len(remaining), len(chunks))

    for i, chunk in enumerate(remaining, start=1):
        log.info(
            "[%d/%d] %s  (%d chars)",
            i,
            len(remaining),
            chunk["article_header"],
            len(chunk["content"]),
        )
        t0 = time.perf_counter()
        await rag.ainsert(
            input=[chunk["content"]],
            ids=[chunk["id"]],
            file_paths=[chunk["source_law"]],
        )
        log.info("  done in %.1fs", time.perf_counter() - t0)

        done.add(chunk["id"])
        checkpoint.write_text(
            json.dumps(sorted(done), ensure_ascii=False), encoding="utf-8"
        )

    log.info("Done — %d chunks ingested.", len(done))


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #


async def main() -> None:
    if not CHUNKS_PATH.exists():
        log.error("%s not found — are you in /workspace?", CHUNKS_PATH)
        sys.exit(1)

    await smoke_test()
    chunks = load_chunks()
    rag = await build_rag()

    t0 = time.perf_counter()
    try:
        await ingest_all(rag, chunks)
    finally:
        finalize = getattr(rag, "finalize_storages", None)
        if callable(finalize):
            try:
                await finalize()
            except Exception as e:
                log.warning("finalize_storages: %s", e)

    log.info("Total time: %.1fs", time.perf_counter() - t0)


if __name__ == "__main__":
    asyncio.run(main())
