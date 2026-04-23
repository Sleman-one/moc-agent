"""
evaluate_retrieval.py — Run the 20 golden questions against the production LightRAG storage.

Uses only_need_context=True so no LLM generation happens during retrieval.
Shows top 3 retrieved article headers per question so we can see retrieval
ranking quality, not just whether the right article appears somewhere.

Tests naive mode at chunk_top_k = 3, 5, 10, 20 to find the minimum k
that still achieves 20/20 — that number is used in qa_pipeline.py.

Run:
  cd ~/projects/moc-agent && uv run python core/evaluate_retrieval.py
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import re
import sys

import pandas as pd

from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc, setup_logger

from core.rag_config import (
    EMBEDDING_DIM,
    GOLDEN_SET_PATH,
    LLM_MODEL,
    WORKDIR,
    embedding_func,
    llm_model_func,
)

PASS_GATE = 16

# ------------------------------------------------------------------ #
# Logging
# ------------------------------------------------------------------ #

setup_logger("lightrag", level="WARNING")
log = logging.getLogger("eval")
log.setLevel(logging.INFO)
if not log.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
    log.addHandler(h)


# ------------------------------------------------------------------ #
# RAG setup — loads existing storage, does not re-ingest
# ------------------------------------------------------------------ #


async def build_rag() -> LightRAG:
    if not WORKDIR.exists():
        log.error("%s does not exist — run ingest_prod.py first.", WORKDIR)
        sys.exit(1)

    rag = LightRAG(
        working_dir=str(WORKDIR),
        llm_model_func=llm_model_func,
        llm_model_name=LLM_MODEL,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=8192,
            func=embedding_func,
        ),
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


# ------------------------------------------------------------------ #
# Helper — extract top N article headers from raw context string
# ------------------------------------------------------------------ #


def extract_top_headers(result: str, n: int = 3) -> list[str]:
    """
    LightRAG returns chunks as JSON objects inside a markdown code block.
    Each chunk has a 'content' field whose first line is the article header.
    We parse them in order — first appearance = highest ranked.
    """
    headers = []
    for match in re.finditer(r"\{[^{}]+\}", result, re.DOTALL):
        try:
            chunk = _json.loads(match.group())
            content = chunk.get("content", "")
            header = content.split("\n")[0].strip()
            if header and header not in headers:
                headers.append(header)
            if len(headers) == n:
                break
        except Exception:
            continue
    return headers


# ------------------------------------------------------------------ #
# Evaluation
# ------------------------------------------------------------------ #


async def evaluate_mode(
    rag: LightRAG,
    df: pd.DataFrame,
    mode: str,
    chunk_top_k: int = 20,
) -> tuple[int, list[int]]:
    passed = 0
    failed: list[int] = []

    print()
    print("=" * 72)
    print(f" mode={mode}  chunk_top_k={chunk_top_k} ".center(72, "="))
    print("=" * 72)

    for _, row in df.iterrows():
        qid = int(row["id"])
        question = row["question_ar"]
        expected_raw = str(row["expected_article"])
        qtype = row["question_type"]

        # Out-of-scope: retrieval isn't meaningful — manual pass.
        if qtype == "out_of_scope":
            passed += 1
            print(f"{qid:>3}  ✓  [out_of_scope — manual pass]")
            continue

        expected = [a.strip() for a in expected_raw.split("+")]

        try:
            result = await rag.aquery(
                question,
                param=QueryParam(
                    mode=mode,
                    only_need_context=True,
                    chunk_top_k=chunk_top_k,
                ),
            )
        except Exception as e:
            failed.append(qid)
            print(f"{qid:>3}  ✗  [ERROR] {type(e).__name__}: {e}")
            continue

        top_headers = extract_top_headers(result, n=3)
        hit = any(a in result for a in expected)

        if hit:
            passed += 1
            status = "✓"
        else:
            failed.append(qid)
            status = "✗"

        print(f"{qid:>3}  {status}  {question[:50]}")
        for i, h in enumerate(top_headers, 1):
            marker = " ←" if any(a in h for a in expected) else ""
            print(f"         top{i}: {h}{marker}")
        if not hit:
            print(f"       expected: {expected_raw}")

    print("-" * 72)
    verdict = "PASS" if passed >= PASS_GATE else "FAIL"
    print(f"  mode={mode} k={chunk_top_k}: {passed}/20  (gate: {PASS_GATE})  {verdict}")
    if failed:
        print(f"  Failed IDs: {failed}")

    return passed, failed


async def main() -> None:
    if not GOLDEN_SET_PATH.exists():
        log.error("%s not found.", GOLDEN_SET_PATH)
        sys.exit(1)

    df = pd.read_excel(GOLDEN_SET_PATH)
    log.info("Loaded %d golden questions.", len(df))

    rag = await build_rag()

    results = {}
    for k in [3, 5, 10, 20]:
        score, _ = await evaluate_mode(rag, df, mode="naive", chunk_top_k=k)
        results[k] = score

    print()
    print("=" * 72)
    print(" SUMMARY ".center(72, "="))
    print("=" * 72)
    for k, score in results.items():
        verdict = "✓ PASS" if score >= PASS_GATE else "✗ FAIL"
        print(f"  chunk_top_k={k:>2}:  {score}/20  {verdict}")

    best_k = min(k for k, s in results.items() if s == 20)
    print()
    print(f"  → Minimum k for 20/20: chunk_top_k={best_k}")
    print(f"    Use this value in qa_pipeline.py QueryParam.")


if __name__ == "__main__":
    asyncio.run(main())
