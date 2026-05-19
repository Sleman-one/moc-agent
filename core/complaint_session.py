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
from core.qa_pipeline import ask as qa_ask
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
# Validation retry policy
#
# Fields with validation (order_date, cr_number) can fail to parse.
# Each failure increments a per-field counter. After MAX_FIELD_ATTEMPTS
# failures on the same field, the session is cancelled and the user is
# directed to the Ministry of Commerce app.
#
# VALIDATION_REASK holds the format-specific message shown on each
# failure BEFORE the limit is reached. VALIDATION_LIMIT_MESSAGE is the
# cancellation message shown when the limit is hit.
# ------------------------------------------------------------------ #

MAX_FIELD_ATTEMPTS = 3

VALIDATION_REASK = {
    "order_date": (
        "لم أستطع تحديد التاريخ بدقة. "
        "يرجى كتابة التاريخ بهذا الشكل: YYYY-MM-DD\n"
        "مثال: 2026-04-23"
    ),
    "cr_number": (
        "رقم السجل التجاري يجب أن يكون 10 أرقام بالضبط. "
        "يمكنك إيجاده في أسفل الموقع الإلكتروني للمتجر "
        "أو في صفحة 'من نحن'. ما هو رقم السجل التجاري؟"
    ),
}

VALIDATION_LIMIT_MESSAGE = (
    "عذراً، لم أتمكن من فهم المعلومة بعد عدة محاولات. "
    "لتقديم شكواك بشكل أدق، يمكنك تقديمها مباشرة من خلال "
    "تطبيق وزارة التجارة. كيف يمكنني مساعدتك بشيء آخر؟"
)

# ------------------------------------------------------------------ #
# LLM helper — direct httpx, thinking always disabled.
# These are classification and extraction tasks, not legal reasoning.
# max_tokens=2048 required — reasoning-parser qwen3 needs larger budget
# even when thinking is disabled.
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
                "max_tokens": 2048,
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

        # Failed-attempt counters for fields that have validation.
        # Incremented on each validation failure, reset to 0 on success.
        # At 3 failures on the same field, the session is cancelled and
        # the user is directed to the Ministry of Commerce app.
        # The counter is on the field, not the state — it persists when
        # the session moves from "collecting" to "confirming".
        self.field_attempts: dict[str, int] = {
            "order_date": 0,
            "cr_number": 0,
        }

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    async def initialize(self, history: list[dict]) -> str:
        """
        Extract fields from conversation history and return the first message.
        Called once by the router immediately after creating the session.

        We do NOT pass the full history to the extractor. We pass only the
        last 3 user messages (the current trigger + the 2 before it). This
        prevents data from previous complaints or unrelated Q&A turns from
        polluting the extraction.

        Why this slice is safe:
            - The trigger message is the last entry in history (router
              appends before calling initialize).
            - 2 user messages back is enough to capture the case where
              the user described their problem before asking to file
              ("I bought from Jarir, my order didn't arrive, I want to
              file a complaint" — three messages).
            - Anything older than 3 user messages is almost certainly
              about something else.
            - Assistant replies are skipped — the bot's own text never
              contains complaint data.
            - The `source` tag is ignored. The 2-message window is the
              protection, not tag filtering. This handles the case where
              the user described a complaint in a Q&A-shaped message
              before a complaint session was active.

        Args:
            history: list of {"role": str, "content": str, "source": str}
                     dicts (source field is optional for backward compat).
        """
        sliced = self._slice_history_for_extraction(history)

        if sliced:
            await self._extract_from_history(sliced)

        next_field = self._next_missing_field()

        if next_field is None:
            # All fields found in history — go straight to confirmation
            self.state = "confirming"
            return self._build_summary()

        self.current_field = next_field
        return self._build_intro()

    async def handle(
        self, message: str, history: list[dict] | None = None
    ) -> tuple[str, str]:
        """
        Process a user message and return (status, response_text).

        status values:
            "active"    — complaint still in progress
            "cancelled" — user cancelled, router destroys the session
            "saved"     — complaint saved to DB, router destroys the session

        The `history` argument is passed through to qa_pipeline.ask() when
        the user asks a mid-complaint legal question (intent "legal_question").
        Optional for backward compatibility — if omitted, the rewriter
        won't have prior Q&A context but will still run.
        """
        try:
            if self.state == "collecting":
                return await self._handle_collecting(message, history)
            elif self.state == "confirming":
                return await self._handle_confirming(message, history)
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

    async def _handle_collecting(
        self, message: str, history: list[dict] | None = None
    ) -> tuple[str, str]:
        """
        Handle a message while in field collection mode.

        `history` is forwarded to the legal-question handler so the Q&A
        rewriter can use prior turns as context. Optional for back-compat.
        """
        intent_data = await self._classify_intent(message)
        intent = intent_data.get("intent", "unclear")

        if intent == "cancel":
            return ("cancelled", "تم إلغاء الشكوى. كيف يمكنني مساعدتك؟")

        elif intent == "legal_question":
            return await self._handle_legal_question(message, history)

        elif intent == "answer":
            value = intent_data.get("value", "").strip()
            if not value:
                return (
                    "active",
                    f"لم أفهم إجابتك. {FIELD_QUESTIONS[self.current_field]}",
                )
            success = await self._store_field(self.current_field, value)
            if not success:
                # Validated field failed to parse. The helper increments
                # the attempt counter and either re-asks or cancels the
                # session if the limit is reached.
                if self.current_field in self.field_attempts:
                    return self._handle_validation_failure(self.current_field)
                # Generic fallback for a non-validated field (should not
                # happen — only order_date and cr_number can fail).
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
                    if field in self.field_attempts:
                        return self._handle_validation_failure(field)
                    return (
                        "active",
                        f"لم أتمكن من حفظ هذه القيمة. {FIELD_QUESTIONS[self.current_field]}",
                    )
                return await self._advance()
            return ("active", f"لم أفهم التصحيح. {FIELD_QUESTIONS[self.current_field]}")

        else:  # unclear
            return ("active", f"لم أفهم. {FIELD_QUESTIONS[self.current_field]}")

    async def _handle_confirming(
        self, message: str, history: list[dict] | None = None
    ) -> tuple[str, str]:
        """
        Handle a message while in confirmation mode.

        `history` is forwarded to the legal-question handler so the Q&A
        rewriter can use prior turns as context. Optional for back-compat.
        """
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

        elif intent == "legal_question":
            return await self._handle_legal_question(message, history)

        elif intent == "correction":
            field = intent_data.get("field")
            value = intent_data.get("value", "").strip()
            if field in self.fields and value:
                success = await self._store_field(field, value)
                if not success:
                    if field in self.field_attempts:
                        status, msg = self._handle_validation_failure(field)
                        # If cancelled, return as-is. If still active,
                        # re-show the summary so the user keeps full context.
                        if status == "cancelled":
                            return (status, msg)
                        return ("active", f"{msg}\n\n{self._build_summary()}")
                    return (
                        "active",
                        f"لم أتمكن من حفظ هذه القيمة.\n\n{self._build_summary()}",
                    )
                return ("active", self._build_summary())
            return ("active", f"لم أفهم التصحيح.\n\n{self._build_summary()}")

        else:  # unclear
            return ("active", f"هل تريد تأكيد تقديم الشكوى؟\n\n{self._build_summary()}")

    async def _handle_legal_question(
        self, message: str, history: list[dict] | None
    ) -> tuple[str, str]:
        """
        Handle a mid-complaint legal question without interrupting the complaint.

        Flow:
            1. Call qa_pipeline.ask() with the user's message + history.
            2. Build a combined response: legal answer + separator + a
               re-prompt of the current question (field question if
               collecting, summary if confirming).
            3. Return ("active", combined_response).

        No field is stored. No state change. The session continues where
        it left off after the user reads the answer.

        On failure (network error, qa_ask raises): return a brief apology
        and re-prompt. The complaint stays alive — the user just loses
        the legal answer.
        """
        try:
            legal_answer = await qa_ask(message, history or [])
        except Exception:
            traceback.print_exc()
            # Failure fallback — apologize and re-prompt without crashing
            return (
                "active",
                "عذراً، لم أتمكن من الإجابة على سؤالك الآن. "
                f"نعود إلى الشكوى — {self._current_reprompt()}",
            )

        # Build combined response: answer + separator + re-prompt.
        # The separator visually marks where the legal answer ends and the
        # complaint flow resumes.
        return (
            "active",
            f"{legal_answer}\n\n---\n\nنعود إلى الشكوى. {self._current_reprompt()}",
        )

    def _current_reprompt(self) -> str:
        """
        Return the text we should show after a legal-question pivot to
        bring the user back to where they were:

            - In "collecting" state → the current field's question.
            - In "confirming" state → the full summary again.
        """
        if self.state == "collecting" and self.current_field:
            return FIELD_QUESTIONS[self.current_field]
        if self.state == "confirming":
            return self._build_summary()
        # Defensive fallback — shouldn't be reachable in practice.
        return "كيف يمكنني مساعدتك؟"

    # ------------------------------------------------------------------ #
    # Field management
    # ------------------------------------------------------------------ #

    async def _store_field(self, field: str, value: str) -> bool:
        """
        Store a field value. Returns True if stored successfully, False if not.

        Validation rules per field:

        order_date — attempt ISO resolution via LLM.
            - Returns True if resolved to a valid ISO date.
            - Returns False if resolution failed — field stays None,
              caller re-asks with a clearer prompt.
            - We never store the raw Arabic string — PostgreSQL DATE
              type will reject it and crash the save.

        cr_number — must be exactly 10 digits after stripping non-digits.
            - Arabic-Indic digits (٠-٩) are normalized to Western.
            - Surrounding text and separators (dashes, spaces) are stripped.
            - Returns True with the cleaned 10-digit string if valid.
            - Returns False if not exactly 10 digits — caller re-asks
              with a format-specific message.

        All other fields: store the raw value directly, always True.

        On successful storage of a validated field, that field's attempt
        counter is reset to 0 — a user who later corrects the field gets
        a fresh set of attempts.
        """
        if field == "order_date":
            resolved = await self._resolve_date(value)
            if resolved:
                self.fields[field] = resolved
                self.field_attempts["order_date"] = 0
                return True
            # Resolution failed — leave field as None so collection re-asks
            return False

        if field == "cr_number":
            cleaned = self._clean_cr_number(value)
            if cleaned is None:
                # Not 10 digits after cleaning — leave field as None
                return False
            self.fields[field] = cleaned
            self.field_attempts["cr_number"] = 0
            return True

        # All other fields: store raw, no validation
        self.fields[field] = value
        return True

    def _handle_validation_failure(self, field: str) -> tuple[str, str]:
        """
        Called when _store_field() returns False for a validated field.

        Increments that field's failure counter, then:
            - If the counter has reached MAX_FIELD_ATTEMPTS, cancels the
              session and returns the app-referral message.
            - Otherwise, returns the field-specific re-ask message so the
              user can try again.

        The counter is on the field and persists across the collecting
        and confirming states. It is reset to 0 in _store_field() when
        the field is eventually stored successfully.

        Returns:
            ("cancelled", message) if the attempt limit is reached,
            ("active", reask_message) otherwise.
        """
        self.field_attempts[field] += 1

        if self.field_attempts[field] >= MAX_FIELD_ATTEMPTS:
            return ("cancelled", VALIDATION_LIMIT_MESSAGE)

        return ("active", VALIDATION_REASK[field])

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

    @staticmethod
    def _slice_history_for_extraction(history: list[dict]) -> list[dict]:
        """
        Take the last 3 user messages from history (current trigger + 2 before).
        Skip assistant replies. Ignore the `source` tag.

        This window is short enough that we won't accidentally reach into
        a previous complaint, while still capturing the common case where
        the user describes their problem across a few messages before
        asking to file.

        Returns a list of user-message dicts, oldest first. Empty list
        if history has no user messages.
        """
        user_messages = [m for m in history if m.get("role") == "user"]
        return user_messages[-3:]

    # ------------------------------------------------------------------ #
    # LLM calls
    # ------------------------------------------------------------------ #

    async def _extract_from_history(self, history: list[dict]) -> None:
        """
        Extract complaint fields from conversation history.
        Runs once on session initialization. Updates self.fields in place.

        Retries up to 2 times if the response cannot be parsed as JSON.
        On total failure, all fields stay None and collection asks for everything.

        The `history` argument is expected to be already sliced by
        initialize() — typically the last 3 user messages, no assistant
        replies. We do not slice further here.
        """
        today = date.today().isoformat()
        history_text = "\n".join(f"المستخدم: {m['content']}" for m in history)

        system = (
            "أنت مساعد متخصص في استخراج بيانات الشكاوى من المحادثات. "
            "أعد JSON فقط بدون أي نص إضافي أو علامات markdown."
        )

        base_user = f"""اليوم هو {today}.
فيما يلي رسائل من المستخدم. استخرج بيانات الشكوى إن وُجدت.

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
                user_prompt = (
                    f"{base_user}\n\n"
                    f"تنبيه: إجابتك السابقة لم تكن JSON صالحاً:\n{last_bad_output}\n"
                    f"أعد JSON فقط بدون أي نص إضافي."
                )

            raw = await _llm_call(system, user_prompt)
            extracted = _parse_json(raw)

            if extracted:
                for field in FIELD_ORDER:
                    value = extracted.get(field)
                    if value and str(value).lower() != "null":
                        self.fields[field] = str(value)
                return

            last_bad_output = raw

        # All attempts failed — fields stay None, collection asks for everything

    async def _classify_intent(self, message: str) -> dict:
        """
        Classify the user's intent given the current state and collected fields.

        Returns a dict with at minimum an "intent" key:
            {"intent": "answer",         "value": "<value>"}
            {"intent": "correction",     "field": "<field_name>", "value": "<new_value>"}
            {"intent": "cancel"}
            {"intent": "confirm"}
            {"intent": "legal_question"}
            {"intent": "unclear"}

        Retries up to 2 times if JSON cannot be parsed.
        On total failure returns {"intent": "unclear"} — re-asks current question.

        Few-shot examples cover all edge cases including:
            1. Short answers (store names, IDs, numbers)
            2. Long descriptive answers (description field)
            3. Correcting the field currently being asked → always "answer"
            4. Correcting a different field → "correction"
            5. Gulf dialect cancellation
            6. Confirmation at summary step
            7. Long description that mentions a problem → "answer"
            8. Pure legal question mid-complaint → "legal_question"
            9. Description + embedded legal question → "answer"
               (special rule: do not throw away the description)
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

مثال ٧ — المستخدم يصف مشكلته بجملة طويلة (هذا دائماً "answer"):
السؤال المطروح: صف المشكلة بالتفصيل — ماذا حدث بالضبط؟
رسالة المستخدم: "استلمت منتجاً معطوباً ولم يرد المتجر على شكواي بعد أسبوعين"
الناتج: {{"intent": "answer", "value": "استلمت منتجاً معطوباً ولم يرد المتجر على شكواي بعد أسبوعين"}}

مثال ٨ — المستخدم يسأل سؤالاً قانونياً في منتصف الشكوى (ليس وصفاً للمشكلة):
السؤال المطروح: ما هو تاريخ الطلب؟
رسالة المستخدم: "بس قبل ما أكمل، هل أصلاً يحق لي أرجع المنتج؟"
الناتج: {{"intent": "legal_question"}}

مثال ٩ — السؤال المطروح هو حقل الوصف، والمستخدم يصف المشكلة ويُضمّن سؤالاً قانونياً:
السؤال المطروح: صف المشكلة بالتفصيل — ماذا حدث بالضبط؟
رسالة المستخدم: "اشتريت لابتوب وما يشتغل، وأبي أعرف هل يحق لي أرجعه"
الناتج: {{"intent": "answer", "value": "اشتريت لابتوب وما يشتغل، وأبي أعرف هل يحق لي أرجعه"}}

---

الآن صنّف هذه الرسالة:
رسالة المستخدم: "{message}"

قاعدة حاسمة: إذا أعطى المستخدم قيمة للحقل الذي سُئل عنه — حتى لو استخدم عبارات مثل "في الحقيقة" أو "التصحيح هو" — فهذا دائماً "answer" وليس "correction".
قاعدة إضافية: أي رسالة تُعدّ إجابةً مباشرةً على السؤال المطروح — بغض النظر عن طولها أو لغتها — هي دائماً "answer".
قاعدة الأسئلة القانونية: إذا كان المستخدم يسأل سؤالاً قانونياً عن حقوقه أو نظام حماية المستهلك أو نظام التجارة الإلكترونية بدلاً من الإجابة على السؤال المطروح، صنّفه "legal_question".
استثناء حقل الوصف: إذا كان السؤال المطروح هو وصف المشكلة، والمستخدم وصف المشكلة وأضاف سؤالاً قانونياً في نفس الرسالة، فهذا "answer" — لا تفقد وصف المشكلة.

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
                return result

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
        falls back to re-asking the user in _store_field().
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

    @staticmethod
    def _clean_cr_number(raw: str) -> str | None:
        """
        Validate and normalize a Saudi commercial registration number.

        Rules:
            - Translate Arabic-Indic digits (٠-٩) to Western (0-9).
            - Strip everything that isn't a digit (handles dashes,
              spaces, surrounding text like "رقم السجل: 1010123456").
            - Result must be exactly 10 digits.

        Returns:
            The cleaned 10-digit string if valid, otherwise None.
        """
        # Translate Arabic-Indic digits to Western
        arabic_to_western = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
        normalized = raw.translate(arabic_to_western)

        # Keep only digits
        digits_only = re.sub(r"\D", "", normalized)

        if len(digits_only) == 10:
            return digits_only
        return None

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
