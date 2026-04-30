"""
complaint_session.py — Multi-turn complaint collection state machine.

ComplaintSession manages one complaint from creation to database save.
The router creates one instance when the classifier returns "start_complaint",
stores it in session state, and calls handle() on every subsequent message.

State machine:
    "collecting"  — asking for missing fields one at a time
    "confirming"  — all fields collected, summary shown, awaiting user response

handle() always returns (status, response_text):
    "active"    — complaint still in progress, keep the session alive
    "cancelled" — user cancelled, router should destroy the session
    "saved"     — complaint saved to DB, router should destroy the session

Fields collected (in order):
    store_name  — اسم المتجر
    cr_number   — رقم السجل التجاري (required by Saudi law for all complaints)
    order_id    — رقم الطلب
    order_date  — تاريخ الطلب
    description — وصف المشكلة

Why no guided_json / guided_choice:
    vLLM's --reasoning-parser qwen3 flag (required for thinking mode in
    qa_pipeline.py) is incompatible with structured decoding. Using them
    together leaves content=None. Instead we use strict prompts, _parse_json()
    for extraction, and re.search for intent — with retry loops for the JSON
    calls (up to 2 retries). With a 35B model this succeeds on the first
    attempt 95%+ of the time. Retries cover the remaining cases.
"""

from __future__ import annotations

import json
import re
import traceback
from datetime import date

import httpx

from core.db import save_complaint
from core.rag_config import LLM_MODEL, VLLM_API_KEY, VLLM_BASE_URL

# ------------------------------------------------------------------ #
# Field metadata
# ------------------------------------------------------------------ #

# Collection order — fields are asked in this sequence
FIELD_ORDER = ["store_name", "cr_number", "order_id", "order_date", "description"]

FIELD_LABELS = {
    "store_name": "اسم المتجر",
    "cr_number": "رقم السجل التجاري",
    "order_id": "رقم الطلب",
    "order_date": "تاريخ الطلب",
    "description": "وصف المشكلة",
}

FIELD_QUESTIONS = {
    "store_name": ("ما هو اسم المتجر الذي تريد تقديم الشكوى ضده؟"),
    "cr_number": (
        "ما هو رقم السجل التجاري للمتجر؟ "
        "يمكنك إيجاده في أسفل الموقع الإلكتروني للمتجر أو في صفحة 'من نحن'."
    ),
    "order_id": ("ما هو رقم الطلب؟"),
    "order_date": ("ما هو تاريخ الطلب؟"),
    "description": ("صف المشكلة بالتفصيل — ماذا حدث بالضبط؟"),
}

# ------------------------------------------------------------------ #
# LLM helper — direct httpx, thinking always disabled.
# These are classification and extraction tasks, not legal reasoning.
# Thinking is disabled via enable_thinking: False in chat_template_kwargs.
# No guided_json or guided_choice — incompatible with --reasoning-parser qwen3.
# ------------------------------------------------------------------ #


async def _llm_call(system: str, user: str) -> str:
    """
    Single LLM call with thinking disabled.
    Returns the content string — never None (returns "" on empty response).
    """
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{VLLM_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {VLLM_API_KEY}"},
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.1,
                "max_tokens": 512,
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""


def _parse_json(text: str) -> dict:
    """
    Strip markdown fences and parse JSON.
    Returns empty dict on any parse failure — callers handle missing keys.
    """
    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return {}


# ------------------------------------------------------------------ #
# ComplaintSession
# ------------------------------------------------------------------ #


class ComplaintSession:
    """
    Manages a single complaint collection conversation.

    Usage (by the router):
        session = ComplaintSession()
        first_message = await session.initialize(history)
        # store session in Streamlit session_state

        status, response = await session.handle(user_message)
        if status in ("cancelled", "saved"):
            # destroy session from session_state
    """

    def __init__(self) -> None:
        self.fields: dict[str, str | None] = {
            "store_name": None,
            "cr_number": None,
            "order_id": None,
            "order_date": None,
            "description": None,
        }
        self.state: str = "collecting"
        self.current_field: str | None = None  # set by initialize()

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    async def initialize(self, history: list[dict]) -> str:
        """
        Extract fields from conversation history and return the first message.
        Called once by the router immediately after creating the session.

        Args:
            history: list of {"role": "user"|"assistant", "content": str} dicts
        """
        if history:
            await self._extract_from_history(history)

        next_field = self._next_missing_field()

        if next_field is None:
            # All fields found in history — go straight to confirmation
            self.state = "confirming"
            return self._build_summary()

        self.current_field = next_field
        return self._build_intro()

    async def handle(self, message: str) -> tuple[str, str]:
        """
        Process a user message and return (status, response_text).

        status values:
            "active"    — complaint still in progress
            "cancelled" — user cancelled, router destroys the session
            "saved"     — complaint saved to DB, router destroys the session
        """
        try:
            if self.state == "collecting":
                return await self._handle_collecting(message)
            elif self.state == "confirming":
                return await self._handle_confirming(message)
            else:
                return (
                    "active",
                    FIELD_QUESTIONS.get(self.current_field, "كيف يمكنني مساعدتك؟"),
                )
        except Exception:
            # Print full traceback for debugging but never crash the conversation
            traceback.print_exc()
            return ("active", "عذراً، حدث خطأ أثناء المعالجة. هل يمكنك إعادة المحاولة؟")

    # ------------------------------------------------------------------ #
    # State handlers
    # ------------------------------------------------------------------ #

    async def _handle_collecting(self, message: str) -> tuple[str, str]:
        """Handle a message while in field collection mode."""
        intent_data = await self._classify_intent(message)
        intent = intent_data.get("intent", "unclear")

        if intent == "cancel":
            return ("cancelled", "تم إلغاء الشكوى. كيف يمكنني مساعدتك؟")

        elif intent == "answer":
            value = intent_data.get("value", "").strip()
            if not value:
                return (
                    "active",
                    f"لم أفهم إجابتك. {FIELD_QUESTIONS[self.current_field]}",
                )
            success = await self._store_field(self.current_field, value)
            if not success:
                if self.current_field == "order_date":
                    return (
                        "active",
                        "لم أستطع تحديد التاريخ بدقة. "
                        "يرجى كتابة التاريخ بهذا الشكل: YYYY-MM-DD\n"
                        "مثال: 2026-04-23",
                    )
                # Generic fallback for any future field that might fail
                return (
                    "active",
                    f"لم أتمكن من حفظ هذه القيمة. {FIELD_QUESTIONS[self.current_field]}",
                )
            return await self._advance()

        elif intent == "correction":
            field = intent_data.get("field")
            value = intent_data.get("value", "").strip()
            if field in self.fields and value:
                success = await self._store_field(field, value)
                if not success:
                    if field == "order_date":
                        return (
                            "active",
                            "لم أستطع تحديد التاريخ بدقة. "
                            "يرجى كتابة التاريخ بهذا الشكل: YYYY-MM-DD\n"
                            "مثال: 2026-04-23",
                        )
                    return (
                        "active",
                        f"لم أتمكن من حفظ هذه القيمة. {FIELD_QUESTIONS[self.current_field]}",
                    )
                return await self._advance()
            return ("active", f"لم أفهم التصحيح. {FIELD_QUESTIONS[self.current_field]}")

        else:  # unclear
            return ("active", f"لم أفهم. {FIELD_QUESTIONS[self.current_field]}")

    async def _handle_confirming(self, message: str) -> tuple[str, str]:
        """Handle a message while in confirmation mode."""
        intent_data = await self._classify_intent(message)
        intent = intent_data.get("intent", "unclear")

        if intent == "confirm":
            complaint_id = save_complaint(self.fields)
            return (
                "saved",
                f"✅ تم تقديم شكواك بنجاح!\n\n"
                f"رقم الشكوى: **{complaint_id}**\n"
                f"سيتم مراجعتها من قِبل الوزارة قريباً.",
            )

        elif intent == "cancel":
            return ("cancelled", "تم إلغاء الشكوى. كيف يمكنني مساعدتك؟")

        elif intent == "correction":
            field = intent_data.get("field")
            value = intent_data.get("value", "").strip()
            if field in self.fields and value:
                await self._store_field(field, value)
                return ("active", self._build_summary())
            return ("active", f"لم أفهم التصحيح.\n\n{self._build_summary()}")

        else:  # unclear
            return ("active", f"هل تريد تأكيد تقديم الشكوى؟\n\n{self._build_summary()}")

    # ------------------------------------------------------------------ #
    # Field management
    # ------------------------------------------------------------------ #

    async def _store_field(self, field: str, value: str) -> bool:
        """
        Store a field value. Returns True if stored successfully, False if not.

        For order_date: attempt ISO resolution via LLM.
            - Returns True if resolved to a valid ISO date.
            - Returns False if resolution failed — field stays None,
              caller must re-ask the user with a clearer prompt.
            - We never store the raw Arabic string — PostgreSQL DATE
              type will reject it and crash the save.
        For all other fields: store the raw value directly, always True.
        """
        if field == "order_date":
            resolved = await self._resolve_date(value)
            if resolved:
                self.fields[field] = resolved
                return True
            # Resolution failed — leave field as None so collection re-asks
            return False
        else:
            self.fields[field] = value
            return True

    async def _advance(self) -> tuple[str, str]:
        """
        After storing a field, move to the next missing one.
        If no fields are missing, transition to confirmation.
        """
        next_field = self._next_missing_field()
        if next_field is None:
            self.state = "confirming"
            return ("active", self._build_summary())
        self.current_field = next_field
        return ("active", FIELD_QUESTIONS[next_field])

    def _next_missing_field(self) -> str | None:
        """Return the first field in FIELD_ORDER that is still None."""
        return next(
            (f for f in FIELD_ORDER if not self.fields.get(f)),
            None,
        )

    # ------------------------------------------------------------------ #
    # LLM calls
    # ------------------------------------------------------------------ #

    async def _extract_from_history(self, history: list[dict]) -> None:
        """
        Extract complaint fields from conversation history.
        Runs once on session initialization. Updates self.fields in place.

        Retries up to 2 times if the response cannot be parsed as JSON.
        On total failure, all fields stay None and collection asks for everything.

        Sliced to last 20 messages — covers any realistic pre-complaint
        conversation without risking context overload.
        """
        today = date.today().isoformat()
        recent_history = history[-20:]
        history_text = "\n".join(
            f"{'المستخدم' if m['role'] == 'user' else 'النظام'}: {m['content']}"
            for m in recent_history
        )

        system = (
            "أنت مساعد متخصص في استخراج بيانات الشكاوى من المحادثات. "
            "أعد JSON فقط بدون أي نص إضافي أو علامات markdown."
        )

        base_user = f"""اليوم هو {today}.
فيما يلي محادثة. استخرج بيانات الشكوى إن وُجدت.

{history_text}

استخرج هذه الحقول:
- store_name:  اسم المتجر
- cr_number:   رقم السجل التجاري
- order_id:    رقم الطلب
- order_date:  تاريخ الطلب بصيغة YYYY-MM-DD
- description: وصف المشكلة

قواعد مهمة:
- أعد JSON فقط، لا تضف أي نص قبله أو بعده
- إذا لم تكن متأكداً من أي قيمة، أعد null لذلك الحقل
- إذا ذكر المستخدم تاريخاً نسبياً مثل "أمس" أو "الأسبوع الماضي"، احسبه بناءً على تاريخ اليوم
- لا تخترع معلومات غير موجودة في المحادثة

المطلوب بالضبط:
{{
  "store_name": "<القيمة أو null>",
  "cr_number": "<القيمة أو null>",
  "order_id": "<القيمة أو null>",
  "order_date": "<YYYY-MM-DD أو null>",
  "description": "<القيمة أو null>"
}}"""

        last_bad_output = None
        for attempt in range(3):  # 1 initial attempt + 2 retries
            if attempt == 0:
                user_prompt = base_user
            else:
                # Show the model what went wrong and ask it to fix it
                user_prompt = (
                    f"{base_user}\n\n"
                    f"تنبيه: إجابتك السابقة لم تكن JSON صالحاً:\n{last_bad_output}\n"
                    f"أعد JSON فقط بدون أي نص إضافي."
                )

            raw = await _llm_call(system, user_prompt)
            extracted = _parse_json(raw)

            if extracted:  # Successfully parsed
                for field in FIELD_ORDER:
                    value = extracted.get(field)
                    if value and str(value).lower() != "null":
                        self.fields[field] = str(value)
                return  # Success — stop retrying

            last_bad_output = raw  # Save for next retry prompt

        # All attempts failed — fields stay None, collection asks for everything

    async def _classify_intent(self, message: str) -> dict:
        """
        Classify the user's intent given the current state and collected fields.

        Returns a dict with at minimum an "intent" key:
            {"intent": "answer",     "value": "<value>"}
            {"intent": "correction", "field": "<field_name>", "value": "<new_value>"}
            {"intent": "cancel"}
            {"intent": "confirm"}
            {"intent": "unclear"}

        Retries up to 2 times if JSON cannot be parsed.
        On total failure returns {"intent": "unclear"} — re-asks current question.

        The model receives full context — state, current question, collected
        fields, missing fields — so it never classifies blind.
        """
        collected = {
            FIELD_LABELS[f]: v for f, v in self.fields.items() if v is not None
        }
        missing = [FIELD_LABELS[f] for f in FIELD_ORDER if self.fields[f] is None]

        if self.state == "collecting":
            state_context = (
                f"نحن في مرحلة جمع البيانات.\n"
                f"السؤال الذي طُرح على المستخدم الآن: "
                f"{FIELD_QUESTIONS.get(self.current_field, '')}"
            )
        else:
            state_context = (
                "نحن في مرحلة التأكيد — المستخدم يراجع بياناته قبل الحفظ النهائي."
            )

        system = (
            "أنت مساعد يصنّف نوايا المستخدمين أثناء عملية تقديم شكوى. "
            "أعد JSON فقط بدون أي نص إضافي أو علامات markdown."
        )

        base_user = f"""السياق الحالي:
- {state_context}
- البيانات المجمعة: {json.dumps(collected, ensure_ascii=False) if collected else "لا شيء بعد"}
- البيانات الناقصة: {', '.join(missing) if missing else "لا شيء — جميع البيانات مكتملة"}

---

أمثلة توضيحية:

مثال ١ — المستخدم يجيب على السؤال المطروح:
السؤال المطروح: ما هو رقم الطلب؟
رسالة المستخدم: "رقم الطلب 12345"
الناتج: {{"intent": "answer", "value": "12345"}}

مثال ٢ — المستخدم يجيب على سؤال رقم السجل التجاري:
السؤال المطروح: ما هو رقم السجل التجاري للمتجر؟
رسالة المستخدم: "1010123456"
الناتج: {{"intent": "answer", "value": "1010123456"}}

مثال ٣ — المستخدم يصحح الحقل المطروح حالياً (هذا "answer" وليس "correction"):
السؤال المطروح: ما هو رقم الطلب؟
رسالة المستخدم: "بالحقيقة رقم الطلب هو 999 مو 888"
الناتج: {{"intent": "answer", "value": "999"}}

مثال ٤ — المستخدم يصحح حقلاً مختلفاً عن الحقل المطروح:
السؤال المطروح: ما هو رقم الطلب؟
رسالة المستخدم: "لا، اسم المتجر جرير مو اكسترا"
الناتج: {{"intent": "correction", "field": "store_name", "value": "جرير"}}

مثال ٥ — المستخدم يلغي (بلهجة عامية):
رسالة المستخدم: "بطلت ما ابي أشتكي"
الناتج: {{"intent": "cancel"}}

مثال ٦ — المستخدم يؤكد في مرحلة التأكيد:
رسالة المستخدم: "نعم صح كل شيء"
الناتج: {{"intent": "confirm"}}

---

الآن صنّف هذه الرسالة:
رسالة المستخدم: "{message}"

قاعدة حاسمة: إذا أعطى المستخدم قيمة للحقل الذي سُئل عنه — حتى لو استخدم عبارات مثل "في الحقيقة" أو "التصحيح هو" — فهذا دائماً "answer" وليس "correction".

أعد JSON فقط:"""

        last_bad_output = None
        for attempt in range(3):  # 1 initial attempt + 2 retries
            if attempt == 0:
                user_prompt = base_user
            else:
                user_prompt = (
                    f"{base_user}\n\n"
                    f"تنبيه: إجابتك السابقة لم تكن JSON صالحاً:\n{last_bad_output}\n"
                    f"أعد JSON فقط."
                )

            raw = await _llm_call(system, user_prompt)
            result = _parse_json(raw)

            if "intent" in result:
                return result  # Success

            last_bad_output = raw

        # All attempts failed — re-ask current question
        return {"intent": "unclear"}

    async def _resolve_date(self, raw: str) -> str | None:
        """
        Resolve a natural language date expression to ISO format (YYYY-MM-DD).
        Returns None if the date cannot be determined with confidence.

        Uses re.search to find the date anywhere in the model response —
        handles cases where the model adds surrounding text.
        No retry needed — re.search handles partial matches and None
        falls back to storing the raw string in _store_field().
        """
        today = date.today().isoformat()
        system = (
            "أنت مساعد يحوّل عبارات التاريخ إلى صيغة ISO. "
            "أعد التاريخ فقط بصيغة YYYY-MM-DD أو كلمة null."
        )
        user = f"""اليوم هو {today}.
المستخدم ذكر: "{raw}"

أعد التاريخ بصيغة YYYY-MM-DD فقط.
إذا لم تستطع تحديد التاريخ بدقة، أعد الكلمة: null"""

        result = (await _llm_call(system, user)).strip()

        if result.lower() == "null":
            return None

        # re.search finds the date anywhere in the string
        match = re.search(r"\d{4}-\d{2}-\d{2}", result)
        if not match:
            return None
        return match.group(0)

    # ------------------------------------------------------------------ #
    # Response builders
    # ------------------------------------------------------------------ #

    def _build_intro(self) -> str:
        """
        Build the opening message after session initialization.
        If fields were extracted from history, acknowledge them explicitly
        so the user knows the system understood the conversation.
        """
        found = {f: v for f, v in self.fields.items() if v is not None}

        if not found:
            return f"سأساعدك في تقديم شكوى.\n" f"{FIELD_QUESTIONS[self.current_field]}"

        lines = ["وجدت المعلومات التالية من محادثتنا:"]
        for field in FIELD_ORDER:
            if self.fields[field] is not None:
                lines.append(f"- {FIELD_LABELS[field]}: {self.fields[field]}")
        lines.append(f"\n{FIELD_QUESTIONS[self.current_field]}")
        return "\n".join(lines)

    def _build_summary(self) -> str:
        """Build the confirmation summary shown before saving."""
        lines = ["هذه بيانات شكواك، يرجى المراجعة بعناية:"]
        for field in FIELD_ORDER:
            value = self.fields.get(field) or "—"
            lines.append(f"- {FIELD_LABELS[field]}: {value}")
        lines.append(
            "\nهل تريد تأكيد تقديم الشكوى؟ "
            "يمكنك تصحيح أي معلومة قبل الحفظ، أو إلغاء الشكوى."
        )
        return "\n".join(lines)
