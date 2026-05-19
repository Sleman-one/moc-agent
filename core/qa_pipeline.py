"""
qa_pipeline.py — Retrieve relevant legal articles and generate a cited answer.

rewrite_query() — converts a follow-up message into a standalone question
                  using the last 2 Q&A pairs from history as context.
retrieve()      — calls LightRAG naive vector search on the rewritten query.
generate()      — direct POST to /v1/chat/completions with our system prompt.
                  Receives the ORIGINAL user question so the answer feels
                  natural to what the user actually typed.
ask()           — chains all three steps. This is what the router calls.
"""

from __future__ import annotations

import re

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
# System prompts
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = """أنت مساعد قانوني متخصص في نظام حماية المستهلك ونظام التجارة الإلكترونية في المملكة العربية السعودية.

قواعد يجب الالتزام بها في كل إجابة:
١. أجب بلغة عربية واضحة ومبسطة يفهمها أي مستهلك، وتجنب المصطلحات القانونية المعقدة.
٢. استشهد برقم المادة القانونية في كل إجابة، مثال: (وفقاً للمادة الحادية والأربعين من نظام حماية المستهلك).
٣. إذا لم تجد الإجابة في المواد القانونية المقدمة، قل بوضوح: "لا تتوفر لديّ معلومات كافية حول هذا الموضوع في النظام الحالي."
٤. إذا كان السؤال خارج نطاق نظام حماية المستهلك ونظام التجارة الإلكترونية، اعتذر بأدب وأوضح أن اختصاصك يقتصر على هذين النظامين فقط.
٥. لا تخترع معلومات أو أرقام مواد غير موجودة في السياق المقدم."""


REWRITER_SYSTEM_PROMPT = """أنت مساعد يعيد صياغة أسئلة المستخدم لتكون مفهومة بدون الحاجة إلى السياق السابق.

المهمة:
- إذا كان السؤال الحالي مفهوماً بمفرده، أعده كما هو دون تغيير.
- إذا كان السؤال الحالي يعتمد على المحادثة السابقة (مثل: "وإذا رفض؟"، "كم مدتها؟"، "وماذا لو..."), أعد صياغته ليصبح سؤالاً كاملاً مفهوماً بدون السياق.

قواعد صارمة:
- لا تجيب على السؤال. مهمتك فقط إعادة الصياغة.
- لا تضف معلومات قانونية أو أرقام مواد. فقط أعد صياغة السؤال.
- أخرج السؤال المعاد صياغته فقط، بدون أي شرح أو مقدمة.

أمثلة:

مثال 1 — سؤال يعتمد على السياق:
المحادثة السابقة:
المستخدم: ما حقي في إرجاع منتج معيب؟
المساعد: حسب نظام حماية المستهلك، يحق لك إرجاع المنتج المعيب خلال مدة محددة...
السؤال الحالي: وإذا رفض البائع؟
الناتج: ما حقي إذا رفض البائع إرجاع منتج معيب؟

مثال 2 — سؤال مستقل:
المحادثة السابقة:
المستخدم: ما هي مدة الضمان على الأجهزة الكهربائية؟
المساعد: مدة الضمان حسب النظام...
السؤال الحالي: هل يحق لي إلغاء الاشتراك في خدمة إلكترونية؟
الناتج: هل يحق لي إلغاء الاشتراك في خدمة إلكترونية؟

مثال 3 — لا يوجد سياق سابق:
السؤال الحالي: ما حقي في الإرجاع؟
الناتج: ما حقي في الإرجاع؟"""


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
# Rewrite
# ------------------------------------------------------------------ #


def _extract_qa_pairs(history: list[dict], max_pairs: int = 2) -> list[tuple[str, str]]:
    """
    Walk history backward and collect up to max_pairs complete user/assistant
    pairs from messages tagged source="qa". A pair is one user message
    followed by one assistant message (in chat order, not history order).

    Untagged messages are treated as qa for backward compatibility — if a
    message has no `source` field, we include it. This matters for any
    history written before fix 1 was deployed.

    Returns oldest pair first, so the LLM sees the conversation in
    chronological order.
    """
    # Filter to qa-tagged messages (or untagged for backward compat)
    qa_messages = [m for m in history if m.get("source", "qa") == "qa"]

    # Walk backward collecting complete pairs.
    # A pair is (user_message, assistant_reply) where the assistant
    # message comes right after the user message in the filtered list.
    pairs: list[tuple[str, str]] = []
    i = len(qa_messages) - 1
    while i >= 1 and len(pairs) < max_pairs:
        if (
            qa_messages[i].get("role") == "assistant"
            and qa_messages[i - 1].get("role") == "user"
        ):
            pairs.append((qa_messages[i - 1]["content"], qa_messages[i]["content"]))
            i -= 2
        else:
            i -= 1

    # We collected newest pair first; flip to chronological order
    pairs.reverse()
    return pairs


async def rewrite_query(message: str, history: list[dict]) -> str:
    """
    Rewrite a user message into a standalone question using prior Q&A turns
    as context. Runs every Q&A turn — there is no detection step.

    - Reads up to the last 2 user/assistant pairs tagged source="qa".
    - Sends them + the current message to the LLM with a strict prompt.
    - Returns the rewritten standalone question (or the original message
      verbatim if the LLM call fails — degraded retrieval is better than
      no answer).

    The returned string is used ONLY for retrieval. The original message
    is what gets shown to the answer-generation model and saved in history.

    Args:
        message: the user's current message.
        history: state["history"] from the router, list of message dicts.

    Returns:
        The rewritten standalone question, or the original on failure.
    """
    pairs = _extract_qa_pairs(history, max_pairs=2)

    # Build the user-message payload.
    # Two shapes depending on whether we have any prior pairs.
    if pairs:
        pairs_text = "\n".join(f"المستخدم: {u}\nالمساعد: {a}" for u, a in pairs)
        user_payload = (
            f"المحادثة السابقة:\n{pairs_text}\n"
            f"السؤال الحالي: {message}\n"
            f"الناتج:"
        )
    else:
        # First Q&A turn — no prior context.
        user_payload = f"السؤال الحالي: {message}\nالناتج:"

    # Call the rewriter LLM.
    # On any failure (network, timeout, garbage output) fall back to the
    # original message so retrieval still runs.
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"{VLLM_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {VLLM_API_KEY}"},
                json={
                    "model": LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": REWRITER_SYSTEM_PROMPT},
                        {"role": "user", "content": user_payload},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 2048,
                    "extra_body": {
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                },
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]

        # Strip any think block in case the server-side reasoning-parser
        # leaves one behind (qwen3 parser sometimes does).
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        # Empty output = treat as failure, fall back to original.
        if not content:
            return message

        return content

    except Exception:
        # Network errors, timeouts, malformed responses — fall back to original.
        # Retrieval runs on the un-rewritten message; same as pre-rewriter
        # behavior. We never propagate this failure to the user.
        return message


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
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return content


# ------------------------------------------------------------------ #
# Public interface
# ------------------------------------------------------------------ #


async def ask(question: str, history: list[dict] | None = None) -> str:
    """
    Public Q&A entry point.

    Three steps:
        1. rewrite_query() — turn follow-ups into standalone questions
           using up to 2 prior Q&A pairs from history.
        2. retrieve()      — run vector search on the REWRITTEN query.
        3. generate()      — produce the answer using the ORIGINAL question
           so it sounds natural to what the user actually typed.

    The rewritten query exists only for this turn and is never saved.
    History stays clean — only the original user message and the final
    answer are appended by the router.

    Args:
        question: the user's current message (original, unmodified).
        history:  state["history"] from the router. Optional for backward
                  compatibility — if not provided, the rewriter still runs
                  but with no prior context (returns the message unchanged
                  in most cases).

    Returns:
        The generated legal answer in Arabic.
    """
    if history is None:
        history = []

    rewritten = await rewrite_query(question, history)
    context = await retrieve(rewritten)
    return await generate(question, context)
