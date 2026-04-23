"""
qa_pipeline.py — Retrieve relevant legal articles and generate a cited answer.

retrieve() — calls LightRAG naive vector search, returns raw context.
generate() — direct POST to /v1/chat/completions with our system prompt.
ask()      — chains both steps. This is what the Streamlit UI calls.
"""

from __future__ import annotations

import httpx

from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc, setup_logger

from core.rag_config import (
    EMBEDDING_DIM,
    LLM_MODEL,
    VLLM_API_KEY,
    VLLM_BASE_URL,
    WORKDIR,
    embedding_func,
    llm_model_func,
)

setup_logger("lightrag", level="WARNING")

# ------------------------------------------------------------------ #
# System prompt
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = """أنت مساعد قانوني متخصص في نظام حماية المستهلك ونظام التجارة الإلكترونية في المملكة العربية السعودية.

قواعد يجب الالتزام بها في كل إجابة:
١. أجب بلغة عربية واضحة ومبسطة يفهمها أي مستهلك، وتجنب المصطلحات القانونية المعقدة.
٢. استشهد برقم المادة القانونية في كل إجابة، مثال: (وفقاً للمادة الحادية والأربعين من نظام حماية المستهلك).
٣. إذا لم تجد الإجابة في المواد القانونية المقدمة، قل بوضوح: "لا تتوفر لديّ معلومات كافية حول هذا الموضوع في النظام الحالي."
٤. إذا كان السؤال خارج نطاق نظام حماية المستهلك ونظام التجارة الإلكترونية، اعتذر بأدب وأوضح أن اختصاصك يقتصر على هذين النظامين فقط.
٥. لا تخترع معلومات أو أرقام مواد غير موجودة في السياق المقدم."""

# ------------------------------------------------------------------ #
# RAG singleton — initialized once, reused across all calls
# ------------------------------------------------------------------ #

_rag: LightRAG | None = None


async def _get_rag() -> LightRAG:
    global _rag
    if _rag is not None:
        return _rag

    if not WORKDIR.exists():
        raise RuntimeError(f"{WORKDIR} does not exist — run ingest_prod.py first.")

    _rag = LightRAG(
        working_dir=str(WORKDIR),
        llm_model_func=llm_model_func,
        llm_model_name=LLM_MODEL,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=8192,
            func=embedding_func,
        ),
    )
    await _rag.initialize_storages()
    await initialize_pipeline_status()
    return _rag


# ------------------------------------------------------------------ #
# Retrieve
# ------------------------------------------------------------------ #


async def retrieve(question: str) -> str:
    rag = await _get_rag()
    return await rag.aquery(
        question,
        param=QueryParam(
            mode="naive",
            only_need_context=True,
            chunk_top_k=20,
        ),
    )


# ------------------------------------------------------------------ #
# Generate
# ------------------------------------------------------------------ #


async def generate(question: str, context: str) -> str:
    user_message = f"""المواد القانونية ذات الصلة:
{context}

السؤال: {question}"""

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            f"{VLLM_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {VLLM_API_KEY}"},
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                "temperature": 0.3,
                "max_tokens": 10240,
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": True},
                    "thinking_budget": 8192,
                },
            },
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]

    # Strip <think>...</think> block — return only the answer
    import re

    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return content


# ------------------------------------------------------------------ #
# Public interface
# ------------------------------------------------------------------ #


async def ask(question: str) -> str:
    context = await retrieve(question)
    return await generate(question, context)
