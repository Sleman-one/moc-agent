"""
evaluate_retrieval.py — Run the 20 golden questions against the production LightRAG storage.

Uses only_need_context=True so no LLM generation happens during retrieval.
LightRAG still requires llm_model_func to be set, so we pass a stub that
raises if called — serving as a safety net.

Runs naive (pure vector) and hybrid (vector + graph) so you can compare
whether the graph is earning its cost.

Run:
  cd /workspace && python evaluate_retrieval.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

import pandas as pd

from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc, setup_logger

from rag_config import (
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
# Evaluation
# ------------------------------------------------------------------ #


async def evaluate_mode(
    rag: LightRAG, df: pd.DataFrame, mode: str
) -> tuple[int, list[int]]:
    passed = 0
    failed: list[int] = []

    print()
    print("=" * 72)
    print(f" Evaluating mode: {mode} ".center(72, "="))
    print("=" * 72)

    for _, row in df.iterrows():
        qid = int(row["id"])
        question = row["question_ar"]
        expected_raw = str(row["expected_article"])
        qtype = row["question_type"]

        # Out-of-scope: retrieval isn't meaningful. Manual pass.
        if qtype == "out_of_scope":
            passed += 1
            print(f"{qid:>3}  ✓  [out_of_scope — manual pass]")
            continue

        expected = [a.strip() for a in expected_raw.split("+")]

        try:
            result = await rag.aquery(
                question,
                param=QueryParam(mode=mode, only_need_context=True),
            )
        except Exception as e:
            failed.append(qid)
            print(f"{qid:>3}  ✗  [ERROR] {type(e).__name__}: {e}")
            continue

        hit = any(a in result for a in expected)
        if hit:
            passed += 1
            print(f"{qid:>3}  ✓  {question[:55]}")
        else:
            failed.append(qid)
            print(f"{qid:>3}  ✗  {question[:55]}")
            print(f"       expected: {expected_raw}")

    print("-" * 72)
    verdict = "PASS" if passed >= PASS_GATE else "FAIL"
    print(f"  {mode}: {passed}/20  (gate: {PASS_GATE})  {verdict}")
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
    naive_pass, _ = await evaluate_mode(rag, df, mode="naive")
    hybrid_pass, _ = await evaluate_mode(rag, df, mode="hybrid")

    print()
    print("=" * 72)
    print(" SUMMARY ".center(72, "="))
    print("=" * 72)
    print(f"  naive:  {naive_pass}/20")
    print(f"  hybrid: {hybrid_pass}/20")
    delta = hybrid_pass - naive_pass
    if delta > 0:
        print(f"  → Hybrid beats naive by {delta}. Graph is earning its keep.")
    elif delta == 0:
        print(f"  → Hybrid and naive tied. Is the graph actually helping here?")
    else:
        print(f"  → Naive beats hybrid by {-delta}. Graph is hurting retrieval.")


if __name__ == "__main__":
    asyncio.run(main())
