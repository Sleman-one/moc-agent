"""
classifier.py — Intent classifier for incoming user messages.

Single responsibility: decide whether a message is a legal question
or the start of a complaint. Returns one of two strings — nothing else.

Only called by the router when no complaint session is active.
Once a session is active, the router bypasses this entirely and sends
the message directly to ComplaintSession.handle().

guided_choice constrains vLLM to output exactly one of the two valid
strings — no JSON parsing, no fallback handling needed.
"""

from __future__ import annotations

import httpx

from core.rag_config import LLM_MODEL, VLLM_API_KEY, VLLM_BASE_URL

# ------------------------------------------------------------------ #
# Valid outputs — passed to guided_choice
# ------------------------------------------------------------------ #

LEGAL_QUESTION = "legal_question"
START_COMPLAINT = "start_complaint"

# ------------------------------------------------------------------ #
# Classifier
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = """أنت مساعد يصنّف رسائل المستخدمين إلى نوعين فقط:
- legal_question: المستخدم يريد معرفة حقوقه أو يسأل سؤالاً قانونياً
- start_complaint: المستخدم يريد تقديم شكوى ضد متجر أو تاجر

أعد الكلمة المناسبة فقط."""

USER_PROMPT = """أمثلة:

رسالة: "ما هي حقوقي إذا وصلني منتج معطوب؟"
التصنيف: legal_question

رسالة: "اشتريت منتجاً تالفاً وأريد أعرف حقوقي القانونية"
التصنيف: legal_question

رسالة: "كم مدة الضمان على المنتجات؟"
التصنيف: legal_question

رسالة: "أبي أقدم شكوى على متجر noon"
التصنيف: start_complaint

رسالة: "المتجر ما رد علي ومو راضي يرجع فلوسي، أبي أشتكي"
التصنيف: start_complaint

رسالة: "أريد الإبلاغ عن تاجر"
التصنيف: start_complaint

---

الآن صنّف هذه الرسالة:
رسالة: "{message}"
التصنيف:"""


async def classify(message: str) -> str:
    """
    Classify a user message as a legal question or the start of a complaint.

    Returns:
        "legal_question"  — route to qa_pipeline.ask()
        "start_complaint" — create a ComplaintSession

    Uses guided_choice to guarantee the output is one of the two valid
    strings — no parsing or fallback needed.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"{VLLM_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {VLLM_API_KEY}"},
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT.format(message=message)},
                ],
                "temperature": 0.0,  # classification must be deterministic
                "max_tokens": 10,  # output is at most one short string
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": False},
                    "guided_choice": [LEGAL_QUESTION, START_COMPLAINT],
                },
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
