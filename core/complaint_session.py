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
# guided_json schemas
# vLLM structured decoding — guarantees the model output matches
# the required JSON structure, eliminating malformed response risk.
# Two schemas: one for intent classification, one for field extraction.
# _resolve_date uses re.search instead — its output is a plain string.
# ------------------------------------------------------------------ #

_INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["answer", "correction", "cancel", "confirm", "unclear"],
        },
        # field and value are optional — only present for "answer" and "correction"
        "field": {
            "type": "string",
            "enum": [
                "store_name",
                "cr_number",
                "order_id",
                "order_date",
                "description",
            ],
        },
        "value": {"type": "string"},
    },
    "required": ["intent"],
}

_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "store_name": {"type": ["string", "null"]},
        "cr_number": {"type": ["string", "null"]},
        "order_id": {"type": ["string", "null"]},
        "order_date": {"type": ["string", "null"]},
        "description": {"type": ["string", "null"]},
    },
    "required": ["store_name", "cr_number", "order_id", "order_date", "description"],
}

# ------------------------------------------------------------------ #
# LLM helper — direct httpx, thinking always disabled.
# These are classification and extraction tasks, not legal reasoning.
# schema: optional guided_json dict passed to vLLM extra_body.
# ------------------------------------------------------------------ #


async def _llm_call(
    system: str,
    user: str,
    schema: dict | None = None,
) -> str:
    """Single LLM call. Returns raw text content."""
    extra_body: dict = {"chat_template_kwargs": {"enable_thinking": False}}
    if schema:
        extra_body["guided_json"] = schema

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
                "extra_body": extra_body,
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


def _parse_json(text: str) -> dict:
    """
    Strip markdown fences and parse JSON.
    With guided_json enabled this is rarely needed, but kept as a safety net.
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
            await self._store_field(self.current_field, value)
            return await self._advance()

        elif intent == "correction":
            field = intent_data.get("field")
            value = intent_data.get("value", "").strip()
            if field in self.fields and value:
                await self._store_field(field, value)
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

    async def _store_field(self, field: str, value: str) -> None:
        """
        Store a field value.
        For order_date: attempt ISO resolution via LLM.
        For all other fields: store the raw value directly.
        """
        if field == "order_date":
            resolved = await self._resolve_date(value)
            # If resolution fails, store the raw string — better than losing the data
            self.fields[field] = resolved if resolved else value
        else:
            self.fields[field] = value

    async def _advance(self) -> tuple[str, str]:
        """
        After storing a field, move to the next missing one.
        If no fields are missing, transition to confirmation.
        Extracted as a method to avoid duplicating this logic in both state handlers.
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

        Sliced to last 20 messages — complaint-relevant data is almost always
        recent, and 20 messages covers any realistic pre-complaint conversation
        without risking context overload on a 32K context window.
        """
        today = date.today().isoformat()
        recent_history = history[-20:]
        history_text = "\n".join(
            f"{'المستخدم' if m['role'] == 'user' else 'النظام'}: {m['content']}"
            for m in recent_history
        )

        system = (
            "أنت مساعد متخصص في استخراج بيانات الشكاوى من المحادثات. " "أعد JSON فقط."
        )
        user = f"""اليوم هو {today}.
فيما يلي محادثة. استخرج بيانات الشكوى إن وُجدت.

{history_text}

استخرج هذه الحقول:
- store_name:  اسم المتجر
- cr_number:   رقم السجل التجاري
- order_id:    رقم الطلب
- order_date:  تاريخ الطلب بصيغة YYYY-MM-DD
- description: وصف المشكلة

قواعد:
- إذا لم تكن متأكداً من أي قيمة، أعد null لذلك الحقل.
- إذا ذكر المستخدم تاريخاً نسبياً مثل "أمس" أو "الأسبوع الماضي"، احسبه بناءً على تاريخ اليوم.
- لا تخترع معلومات غير موجودة في المحادثة."""

        raw = await _llm_call(system, user, schema=_EXTRACTION_SCHEMA)
        extracted = _parse_json(raw)

        for field in FIELD_ORDER:
            value = extracted.get(field)
            if value and str(value).lower() != "null":
                self.fields[field] = str(value)

    async def _classify_intent(self, message: str) -> dict:
        """
        Classify the user's intent given the current state and collected fields.

        Returns a dict with at minimum an "intent" key:
            {"intent": "answer",     "value": "<value>"}
            {"intent": "correction", "field": "<field_name>", "value": "<new_value>"}
            {"intent": "cancel"}
            {"intent": "confirm"}
            {"intent": "unclear"}

        The model receives full context — state, current question, collected
        fields, missing fields — so it never classifies blind.

        Few-shot examples protect the two hardest edge cases:
            1. User corrects the field currently being asked about → "answer"
            2. Gulf dialect cancellation expressions → "cancel"
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
            "أنت مساعد يصنّف نوايا المستخدمين أثناء عملية تقديم شكوى. " "أعد JSON فقط."
        )
        user = f"""السياق الحالي:
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

قاعدة حاسمة: إذا أعطى المستخدم قيمة للحقل الذي سُئل عنه — حتى لو استخدم عبارات مثل "في الحقيقة" أو "التصحيح هو" — فهذا دائماً "answer" وليس "correction"."""

        raw = await _llm_call(system, user, schema=_INTENT_SCHEMA)
        result = _parse_json(raw)

        if "intent" not in result:
            return {"intent": "unclear"}
        return result

    async def _resolve_date(self, raw: str) -> str | None:
        """
        Resolve a natural language date expression to ISO format (YYYY-MM-DD).
        Returns None if the date cannot be determined with confidence.

        Uses re.search (not re.match) to find the date anywhere in the
        model's response — handles cases where the model adds prefix text.
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

        # re.search finds the date anywhere in the string,
        # even if the model adds surrounding text
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
