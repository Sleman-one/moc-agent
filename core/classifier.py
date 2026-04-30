"""
classifier.py — Intent classifier for incoming user messages.

Single responsibility: decide whether a message is a legal question
or the start of a complaint. Returns one of two strings — nothing else.

Only called by the router when no complaint session is active.
Once a session is active, the router bypasses this entirely and sends
the message directly to ComplaintSession.handle().

Why no guided_choice:
    vLLM's --reasoning-parser qwen3 flag (required for thinking mode in
    qa_pipeline.py) is incompatible with guided_choice and guided_json.
    Using them together leaves content=None. Instead we use a strict
    prompt and re.search to extract the label from the response.
    With a 35B model and clear instructions this is reliable. The fallback
    is "legal_question" — safer to answer a Q&A than to misroute.
"""

from __future__ import annotations

import re

import httpx

from core.rag_config import LLM_MODEL, VLLM_API_KEY, VLLM_BASE_URL

# ------------------------------------------------------------------ #
# Valid outputs
# ------------------------------------------------------------------ #

LEGAL_QUESTION = "legal_question"
START_COMPLAINT = "start_complaint"

# ------------------------------------------------------------------ #
# Prompts
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = """أنت مساعد يصنّف رسائل المستخدمين إلى نوعين فقط.
يجب أن تنتهي إجابتك دائماً بأحد هذين التصنيفين بالضبط:
legal_question — إذا كان المستخدم يريد معرفة حقوقه أو يسأل سؤالاً قانونياً
start_complaint — إذا كان المستخدم يريد تقديم شكوى ضد متجر أو تاجر"""

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

رسالة: "اشتريت من noon منتجاً معطوباً الأسبوع الماضي وعندي سؤال قانوني"
التصنيف: legal_question

---

الآن صنّف هذه الرسالة وأنهِ إجابتك بالتصنيف المناسب:
رسالة: "{message}"
التصنيف:"""


# ------------------------------------------------------------------ #
# Classifier
# ------------------------------------------------------------------ #


async def classify(message: str) -> str:
    """
    Classify a user message as a legal question or the start of a complaint.

    Returns:
        "legal_question"  — route to qa_pipeline.ask()
        "start_complaint" — create a ComplaintSession

    Uses re.search to find the label anywhere in the model response —
    forgiving enough to handle any surrounding text the model adds.
    Fallback is "legal_question" — safer failure mode.
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
                "temperature": 0.0,
                "max_tokens": 256,
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            },
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"] or ""

    # Search for either valid label anywhere in the response
    match = re.search(r"legal_question|start_complaint", content)
    if match:
        return match.group(0)

    # Fallback — answering a Q&A question is safer than misrouting to complaint
    return LEGAL_QUESTION
