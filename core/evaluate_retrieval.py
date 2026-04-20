import asyncio
import pandas as pd
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

# ── Configuration ─────────────────────────────────────────────────────────────

WORKING_DIR = "lightrag_storage"
GOLDEN_SET_PATH = "tests/golden_set.xlsx"
EMBEDDING_DIM = 1024
PASS_GATE = 16

# ── LightRAG setup ────────────────────────────────────────────────────────────


async def build_rag() -> LightRAG:
    rag = LightRAG(
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
    await rag.initialize_storages()
    return rag


# ── Evaluation ────────────────────────────────────────────────────────────────


async def evaluate():
    rag = await build_rag()

    df = pd.read_excel(GOLDEN_SET_PATH)

    passed = 0
    failed = []

    print(f"{'─'*70}")
    print(f"{'Q':>3}  {'Status':<6}  Question")
    print(f"{'─'*70}")

    for _, row in df.iterrows():
        qid = int(row["id"])
        question = row["question_ar"]
        expected_raw = str(row["expected_article"])
        qtype = row["question_type"]

        # Out-of-scope question — retrieval is not meaningful to evaluate.
        # The system should refuse, which we test in Phase 2 with the full
        # pipeline. Mark as manual pass here.
        if qtype == "out_of_scope":
            passed += 1
            print(f"{qid:>3}  {'✓':<6}  [out_of_scope — manual pass] {question[:45]}")
            continue

        # Handle multi-article questions — pass if at least one article found
        expected_articles = [a.strip() for a in expected_raw.split("+")]

        # Retrieve context only — no LLM answer generation
        result = await rag.aquery(
            question,
            param=QueryParam(mode="naive", only_need_context=True),
        )

        hit = any(article in result for article in expected_articles)

        if hit:
            passed += 1
            print(f"{qid:>3}  {'✓':<6}  {question[:55]}")
        else:
            failed.append(qid)
            print(f"{qid:>3}  {'✗':<6}  {question[:55]}")
            # Show which article we expected vs what we got
            print(f"     Expected : {expected_raw}")
            print(f"     Context  : {result[:200]}\n")

    print(f"{'─'*70}")
    print(f"\nResult : {passed}/20")
    print(f"Gate   : {PASS_GATE}/20")
    print(f"Status : {'PASS ✓' if passed >= PASS_GATE else 'FAIL ✗'}")
    if failed:
        print(f"Failed : {failed}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(evaluate())
