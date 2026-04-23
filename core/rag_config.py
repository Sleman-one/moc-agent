"""
rag_config.py — Shared config for ingestion and evaluation.

Both ingest_prod.py and evaluate_retrieval.py import from this file.

Critical design choices (learned the hard way):
- Embedding uses a direct httpx POST, NOT LightRAG's openai_embed wrapper.
  The wrapper is an EmbeddingFunc object (not a plain function) and fights
  with dimension validation when wrapped again by LightRAG internally.
- LLM uses LightRAG's openai_complete_if_cache — it integrates with the
  priority queue and retry logic, which we need.
- Qwen3.6 requires enable_thinking=False in extra_body. Without it, every
  entity-extraction call generates thousands of <think> tokens, hits
  LLM_TIMEOUT, and ingestion deadlocks. qwen2.5:7b ignores this flag safely.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from dotenv import load_dotenv

from lightrag.llm.openai import openai_complete_if_cache

# ------------------------------------------------------------------ #
# Load environment variables from .env
# ------------------------------------------------------------------ #

load_dotenv()

# ------------------------------------------------------------------ #
# Endpoints + models — all read from .env
# ------------------------------------------------------------------ #

VLLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434/v1")
VLLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")

LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
EMBEDDING_DIM = 1024

CHUNKS_PATH = Path("data/processed/chunks_final_v2.json")
GOLDEN_SET_PATH = Path("tests/golden_set.xlsx")
WORKDIR = Path("lightrag_storage_prod")

# ------------------------------------------------------------------ #
# LightRAG env vars — read at import time inside LightRAG
# ------------------------------------------------------------------ #

os.environ.setdefault("LLM_TIMEOUT", "300")
os.environ.setdefault("FORCE_LLM_SUMMARY_ON_MERGE", "8")
os.environ.setdefault("SUMMARY_MAX_TOKENS", "1200")
os.environ.setdefault("SUMMARY_LENGTH_RECOMMENDED", "600")
os.environ.setdefault("SUMMARY_CONTEXT_SIZE", "12000")

# ------------------------------------------------------------------ #
# Qwen3 settings — enable_thinking=False is critical for Qwen3.6
# qwen2.5:7b ignores this flag safely, so it's always set to False
# ------------------------------------------------------------------ #

_enable_thinking = os.getenv("ENABLE_THINKING", "false").lower() == "true"

QWEN_EXTRA_BODY: dict[str, Any] = {
    "chat_template_kwargs": {"enable_thinking": _enable_thinking},
    "presence_penalty": 1.5,
}


async def llm_model_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list | None = None,
    keyword_extraction: bool = False,
    **kwargs: Any,
) -> str:
    kwargs.pop("hashing_kv", None)
    return await openai_complete_if_cache(
        model=LLM_MODEL,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        base_url=VLLM_BASE_URL,
        api_key=VLLM_API_KEY,
        temperature=0.8,
        max_tokens=8192,
        extra_body=QWEN_EXTRA_BODY,
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    """Direct POST to /v1/embeddings — bypasses LightRAG's wrapper."""
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{EMBEDDING_BASE_URL}/embeddings",
            headers={"Authorization": f"Bearer {VLLM_API_KEY}"},
            json={"model": EMBEDDING_MODEL, "input": texts},
        )
        r.raise_for_status()
        data = r.json()["data"]
    return np.array([row["embedding"] for row in data], dtype=np.float32)
