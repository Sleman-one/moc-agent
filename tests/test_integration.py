"""
test_integration.py — Full integration test for Phase 4.5 fixes.

Exercises the complete router → classifier → complaint_session → qa_pipeline
stack against the live RunPod backend. Runs 19 scenarios grouped by which
fix they primarily exercise.

Run with:
    uv run -m tests.test_integration

Output:
    Streams to stdout AND writes to logs/test_integration.log.
    Final summary lists automated checks passed/failed and scenarios
    needing human review.

What's automated vs human-reviewed:
    Automated — state machine outcomes, counter values, history tagging,
                CR/date storage formats, session lifecycle, response text
                substring checks where unambiguous.
    Human    — LLM output quality: rewriter faithfulness, legal answer
                correctness, intent classification on edge cases. The
                script logs the model's choices for you to review.

Instrumentation:
    - save_complaint() is mocked so we don't pollute Postgres.
    - rewrite_query() is wrapped to log (original, rewritten) pairs.
    - _classify_intent() is wrapped to log every classification.
    These mocks are installed once at startup.

Each scenario:
    - Creates a fresh state via init_state().
    - Sends a scripted message sequence through router.handle().
    - Runs automated assertions where possible.
    - Records anything that needs human review.
"""

from __future__ import annotations

import asyncio
import sys
import traceback
from datetime import datetime
from pathlib import Path

from core.complaint_session import ComplaintSession
from core.router import handle, init_state

# ------------------------------------------------------------------ #
# Logging — tee stdout and stderr to a file so tracebacks land in log
# ------------------------------------------------------------------ #


class Tee:
    """
    Write to multiple streams simultaneously.

    Forwards unknown attributes (encoding, isatty, fileno, etc.) to the
    first stream so code that introspects sys.stdout doesn't blow up.
    """

    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> None:
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self) -> None:
        for s in self.streams:
            s.flush()

    def __getattr__(self, name):
        # Delegate things like .encoding, .isatty() to the first underlying
        # stream so callers that probe sys.stdout don't AttributeError.
        return getattr(self.streams[0], name)


# Install Tee for both stdout AND stderr so prints and tracebacks both
# land in the log file. Use sys.__stdout__/__stderr__ to keep a reference
# to the original streams in case anything bypasses our Tee.
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_PATH = LOG_DIR / f"test_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
_log_file = open(LOG_PATH, "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, _log_file)
sys.stderr = Tee(sys.__stderr__, _log_file)


# ------------------------------------------------------------------ #
# Mocks and instrumentation
# ------------------------------------------------------------------ #

# Capture rewriter input/output pairs.
# Each entry: {"scenario": str, "turn": int, "original": str, "rewritten": str}
REWRITE_LOG: list[dict] = []

# Capture every classifier call.
# Each entry: {"scenario": str, "turn": int, "message": str, "intent": dict}
INTENT_LOG: list[dict] = []

# Track the current scenario + turn for instrumentation context.
_CURRENT_SCENARIO = ""
_CURRENT_TURN = 0


def install_mocks() -> None:
    """Patch save_complaint, rewrite_query, _classify_intent for testing."""
    import core.complaint_session as cs_module
    import core.qa_pipeline as qa_module

    # ---- Mock save_complaint: return a fake ID, don't touch Postgres
    fake_id_counter = [1000]

    def fake_save_complaint(fields: dict) -> int:
        fake_id_counter[0] += 1
        return fake_id_counter[0]

    cs_module.save_complaint = fake_save_complaint

    # ---- Wrap rewrite_query to log every call
    original_rewrite = qa_module.rewrite_query

    async def wrapped_rewrite(message: str, history: list[dict]) -> str:
        rewritten = await original_rewrite(message, history)
        REWRITE_LOG.append(
            {
                "scenario": _CURRENT_SCENARIO,
                "turn": _CURRENT_TURN,
                "original": message,
                "rewritten": rewritten,
            }
        )
        return rewritten

    qa_module.rewrite_query = wrapped_rewrite
    # ask() captures rewrite_query at import time via closure; we patch the
    # name in qa_pipeline so future calls from ask() see the wrapper.
    # ask() actually calls rewrite_query directly by name, so the patch
    # on the module attribute does take effect.

    # ---- Wrap _classify_intent on the ComplaintSession class
    original_classify = ComplaintSession._classify_intent

    async def wrapped_classify(self, message: str) -> dict:
        result = await original_classify(self, message)
        INTENT_LOG.append(
            {
                "scenario": _CURRENT_SCENARIO,
                "turn": _CURRENT_TURN,
                "message": message,
                "state": self.state,
                "current_field": self.current_field,
                "intent": result,
            }
        )
        return result

    ComplaintSession._classify_intent = wrapped_classify


# ------------------------------------------------------------------ #
# Test harness
# ------------------------------------------------------------------ #

SEP = "─" * 70

# Track all results
RESULTS: dict[str, dict] = {}  # scenario_id -> {"checks": list, "human_review": list}
HUMAN_REVIEW: list[str] = []  # scenario_ids that need human eyes


class ScenarioContext:
    """Per-scenario state — turn counter, checks log, human-review log."""

    def __init__(self, scenario_id: str, title: str) -> None:
        self.id = scenario_id
        self.title = title
        self.turn = 0
        self.checks: list[tuple[bool, str]] = []  # (passed, description)
        self.human_review: list[str] = []
        global _CURRENT_SCENARIO, _CURRENT_TURN
        _CURRENT_SCENARIO = scenario_id
        _CURRENT_TURN = 0

    async def send(self, state: dict, message: str) -> str:
        """Send a message through the router, print user + bot turns."""
        self.turn += 1
        global _CURRENT_TURN
        _CURRENT_TURN = self.turn

        # Snapshot intent log length so we know which intent (if any)
        # was produced by THIS turn.
        intents_before = len(INTENT_LOG)

        print(f"\n[Turn {self.turn}] USER: {message}")
        try:
            response = await handle(message, state)
        except Exception as e:
            print(f"  ✗ EXCEPTION during handle(): {e}")
            traceback.print_exc()
            self.check(False, f"handle() raised: {e}")
            raise
        print(f"[Turn {self.turn}] BOT:  {response}")

        # Print intent if the classifier ran during this turn.
        # (Q&A turns don't invoke _classify_intent — only complaint sessions do.)
        new_intents = INTENT_LOG[intents_before:]
        for entry in new_intents:
            print(
                f"  [intent] state={entry['state']} "
                f"current_field={entry['current_field']} → "
                f"{entry['intent']}"
            )
        return response

    def last_intent(self) -> dict | None:
        """Return the most recent classifier output for this scenario, or None."""
        mine = [i for i in INTENT_LOG if i["scenario"] == self.id]
        return mine[-1]["intent"] if mine else None

    def check(self, passed: bool, description: str) -> None:
        """Record an automated check result."""
        self.checks.append((passed, description))
        icon = "✓" if passed else "✗"
        print(f"  {icon} {description}")

    def review(self, note: str) -> None:
        """Mark something that needs human review."""
        self.human_review.append(note)
        print(f"  ⚠ HUMAN REVIEW: {note}")

    def state_snapshot(self, state: dict) -> None:
        """Print the relevant parts of state for visibility."""
        sess = state["complaint_session"]
        if sess is None:
            print(f"  STATE: session=None")
            return
        print(
            f"  STATE: session=active, mode={sess.state}, "
            f"current_field={sess.current_field}, "
            f"attempts={dict(sess.field_attempts)}, "
            f"fields={ {k: v for k, v in sess.fields.items() if v is not None} }"
        )

    def finish(self) -> None:
        """Save scenario results."""
        RESULTS[self.id] = {
            "title": self.title,
            "checks": self.checks,
            "human_review": self.human_review,
        }
        if self.human_review:
            HUMAN_REVIEW.append(self.id)


def header(scenario_id: str, title: str) -> ScenarioContext:
    print(f"\n{SEP}")
    print(f"  {scenario_id}: {title}")
    print(SEP)
    return ScenarioContext(scenario_id, title)


# ------------------------------------------------------------------ #
# Group 1 — Baseline
# ------------------------------------------------------------------ #


async def s1_pure_qa_single_turn() -> None:
    ctx = header("S1", "Pure Q&A, single turn")

    state = init_state()
    response = await ctx.send(state, "ما حقي في إرجاع منتج معيب؟")

    # Automated: history has 2 messages, both tagged qa
    ctx.check(len(state["history"]) == 2, "history has 2 entries")
    ctx.check(
        all(m["source"] == "qa" for m in state["history"]),
        "all history entries tagged 'qa'",
    )
    ctx.check(state["complaint_session"] is None, "no complaint session created")

    # Human review: was the legal answer reasonable?
    ctx.review(
        "Read the bot's response — does it answer the refund question accurately?"
    )
    ctx.finish()


async def s2_pure_complaint_happy_path() -> None:
    ctx = header("S2", "Pure complaint, all fields fresh, save")

    state = init_state()

    await ctx.send(state, "أبي أقدم شكوى")
    ctx.check(state["complaint_session"] is not None, "session created on trigger")

    await ctx.send(state, "جرير")
    ctx.state_snapshot(state)

    # Use a dirty CR to also test fix 4 inline
    await ctx.send(state, "رقم السجل: 1010-123-456")
    ctx.check(
        state["complaint_session"].fields["cr_number"] == "1010123456",
        "dirty CR normalized to 10 digits",
    )

    await ctx.send(state, "ORD-789")
    await ctx.send(state, "2026-04-15")
    ctx.check(
        state["complaint_session"].fields["order_date"] == "2026-04-15",
        "date stored in ISO format",
    )

    await ctx.send(state, "وصلني المنتج معطوباً ورفضوا الإرجاع")
    ctx.check(
        state["complaint_session"].state == "confirming",
        "reached confirming state",
    )

    await ctx.send(state, "نعم تأكيد")
    ctx.check(state["complaint_session"] is None, "session destroyed after save")

    # Tagging check: the trigger message arrived when NO session was active,
    # so it's tagged "qa". All messages AFTER that are tagged "complaint"
    # (user msgs because a session was active; assistant msgs because the
    # complaint path produced them).
    h = state["history"]
    ctx.check(len(h) >= 2, "history has at least 2 entries")
    ctx.check(
        h[0]["source"] == "qa", "trigger message tagged 'qa' (session not yet active)"
    )
    ctx.check(
        all(m["source"] == "complaint" for m in h[1:]),
        "all entries after trigger tagged 'complaint'",
    )
    ctx.finish()


# ------------------------------------------------------------------ #
# Group 3 — Fix 2 (history slicing / cross-complaint pollution)
# ------------------------------------------------------------------ #


async def s3_two_complaints_same_session() -> None:
    ctx = header("S3", "File complaint A, then complaint B — no field leakage")

    state = init_state()

    # Complaint A — Jarir
    await ctx.send(state, "أبي أقدم شكوى")
    await ctx.send(state, "جرير")
    await ctx.send(state, "1010111111")
    await ctx.send(state, "A-001")
    await ctx.send(state, "2026-03-01")
    await ctx.send(state, "تأخر التوصيل")
    await ctx.send(state, "تأكيد")
    ctx.check(state["complaint_session"] is None, "first complaint saved")

    # Complaint B — Noon (no pre-description, fresh start)
    await ctx.send(state, "أبي أقدم شكوى ثانية")
    ctx.check(state["complaint_session"] is not None, "second session created")

    # Check no leakage from Jarir.
    # NOTE: this is a known design risk we discussed — the slicer takes the
    # last 3 user messages regardless of tag. Right after complaint A saves,
    # the most recent user messages are A's description and "تأكيد". The
    # extractor sees those + the new trigger. We hope it ignores them — but
    # the description is the most likely to leak.
    sess = state["complaint_session"]
    leaked = []
    if sess.fields["store_name"] == "جرير":
        leaked.append("store_name=جرير")
    if sess.fields["cr_number"] == "1010111111":
        leaked.append("cr_number=1010111111")
    if sess.fields["order_id"] == "A-001":
        leaked.append("order_id=A-001")
    if sess.fields["description"] == "تأخر التوصيل":
        leaked.append("description=تأخر التوصيل")

    ctx.check(not leaked, f"no fields leaked from complaint A (leaks: {leaked})")
    # The description is a known risk — flag for review even if it didn't leak
    # exactly, in case it was partially captured.
    if sess.fields["description"]:
        ctx.review(
            f"complaint B has description={sess.fields['description']!r} — "
            f"verify this isn't a leak from complaint A"
        )
    ctx.state_snapshot(state)
    ctx.finish()


async def s4_pre_complaint_description_extracted() -> None:
    ctx = header("S4", "User describes problem before triggering — fields extracted")

    state = init_state()

    # Pre-complaint description.
    # CRITICAL: this message must NOT trigger a complaint by itself (otherwise
    # the second turn arrives in an already-active session and the test breaks).
    # We frame it as a question to keep the top-level classifier on the Q&A path.
    await ctx.send(state, "اشتريت من نون وما وصلني الطلب، ما حقوقي؟")
    ctx.check(
        state["complaint_session"] is None,
        "first message stayed in Q&A path (no complaint triggered yet)",
    )

    await ctx.send(state, "أبي أقدم شكوى")
    ctx.check(
        state["complaint_session"] is not None, "session created on explicit trigger"
    )

    sess = state["complaint_session"]
    ctx.state_snapshot(state)

    # Extractor MAY auto-fill store_name; let's check what happened
    if sess.fields["store_name"] == "نون":
        ctx.check(True, "store_name auto-extracted as 'نون'")
    else:
        ctx.review(
            f"store_name auto-fill — expected 'نون' or None, "
            f"got: {sess.fields['store_name']!r}"
        )

    # Description may or may not be filled — also flag for review
    if sess.fields["description"]:
        ctx.review(
            f"description auto-extracted: {sess.fields['description']!r} "
            f"— is this acceptable as a description?"
        )

    ctx.finish()


async def s5_qa_doesnt_pollute_new_complaint() -> None:
    ctx = header(
        "S5", "Multiple Q&A turns before complaint — extractor shouldn't pollute"
    )

    state = init_state()

    # Three unrelated Q&A turns
    await ctx.send(state, "ما هي مدة الضمان على الأجهزة الكهربائية؟")
    await ctx.send(state, "هل يحق لي إلغاء طلب إلكتروني؟")
    await ctx.send(state, "ما هي عقوبة الغش التجاري؟")

    # Now trigger complaint with NO pre-description
    await ctx.send(state, "أبي أقدم شكوى")

    sess = state["complaint_session"]
    ctx.state_snapshot(state)

    # 2-message window will include the last 2 Q&A questions + trigger.
    # The extractor SHOULD ignore them (questions are not field values).
    # This is a model-judgment test — log what happened, flag for review.
    filled = {k: v for k, v in sess.fields.items() if v is not None}
    if filled:
        ctx.review(
            f"Extractor auto-filled {list(filled.keys())} from Q&A history. "
            f"Review whether these values are sensible: {filled}"
        )
    else:
        ctx.check(True, "no fields auto-filled from Q&A history (good)")

    ctx.finish()


# ------------------------------------------------------------------ #
# Group 4 — Fix 3 (query rewriter)
# ------------------------------------------------------------------ #


async def s6_followup_needs_context() -> None:
    ctx = header("S6", "Follow-up needs prior context to be answered correctly")

    state = init_state()

    await ctx.send(state, "ما حقي في إرجاع منتج معيب؟")
    await ctx.send(state, "وإذا رفض البائع؟")

    # The rewriter's output is in REWRITE_LOG — print the latest entry
    recent = [r for r in REWRITE_LOG if r["scenario"] == "S6"]
    for r in recent:
        print(f"  REWRITER: '{r['original']}' → '{r['rewritten']}'")

    # Human review: did the rewriter produce a standalone question?
    ctx.review(
        "Look at REWRITER lines above — did 'وإذا رفض البائع؟' get rewritten "
        "into a standalone question mentioning refund/return?"
    )
    ctx.review(
        "Did the bot's answer to the follow-up mention BOTH defective return "
        "AND seller refusal?"
    )
    ctx.finish()


async def s7_standalone_not_over_rewritten() -> None:
    ctx = header("S7", "Standalone question shouldn't be over-rewritten")

    state = init_state()

    await ctx.send(state, "ما هي مدة الضمان على الأجهزة الكهربائية؟")
    await ctx.send(state, "هل يحق لي إلغاء الاشتراك في خدمة إلكترونية؟")

    recent = [r for r in REWRITE_LOG if r["scenario"] == "S7" and r["turn"] == 2]
    for r in recent:
        print(f"  REWRITER turn 2: '{r['original']}' → '{r['rewritten']}'")

    ctx.review(
        "Look at REWRITER turn 2 — is the rewrite roughly equivalent to the original, "
        "or did it incorrectly drag in warranty context from turn 1?"
    )
    ctx.finish()


async def s8_rewriter_skips_complaint_history() -> None:
    ctx = header("S8", "Rewriter shouldn't see complaint history")

    state = init_state()

    # Q&A turn (will be tagged "qa")
    await ctx.send(state, "ما حقي في إرجاع منتج معيب؟")

    # Complete a complaint
    await ctx.send(state, "أبي أقدم شكوى")
    await ctx.send(state, "اكسترا")
    await ctx.send(state, "1010222222")
    await ctx.send(state, "ORD-X")
    await ctx.send(state, "2026-02-10")
    await ctx.send(state, "وصل المنتج خاطئاً")
    await ctx.send(state, "تأكيد")
    ctx.check(state["complaint_session"] is None, "complaint saved")

    # Now a new Q&A follow-up
    await ctx.send(state, "وكم مدة الضمان عادة؟")

    # The rewriter for the last turn should only have seen the FIRST Q&A pair.
    # The complaint turns in between are tagged "complaint" and filtered out.
    recent = [r for r in REWRITE_LOG if r["scenario"] == "S8"]
    final = recent[-1] if recent else None
    if final:
        print(f"  REWRITER final: '{final['original']}' → '{final['rewritten']}'")

    ctx.review(
        "Did the rewriter correctly drop the complaint history? "
        "The rewrite should reference 'refund'/'defective product' from the first Q&A, "
        "not the complaint details about Extra/ORD-X."
    )
    ctx.finish()


# ------------------------------------------------------------------ #
# Group 5 — Fix 4 (CR validation)
# ------------------------------------------------------------------ #


async def s9_cr_variations() -> None:
    ctx = header("S9", "CR number normalized across dirty inputs")

    # Test multiple CR forms within one complaint by correcting it.
    state = init_state()

    await ctx.send(state, "أبي أقدم شكوى")
    await ctx.send(state, "جرير")

    # First attempt: surrounding text
    await ctx.send(state, "رقم السجل التجاري: 1010333333")
    sess = state["complaint_session"]
    if sess.fields["cr_number"] == "1010333333":
        ctx.check(True, "CR with surrounding text normalized")
    else:
        ctx.check(
            False, f"expected cr_number=1010333333, got {sess.fields['cr_number']!r}"
        )

    # Continue to fill remaining fields
    await ctx.send(state, "ORD-9")
    await ctx.send(state, "2026-01-15")
    await ctx.send(state, "المنتج مختلف عن الإعلان")
    # Should be at confirming state now

    # At confirmation, correct CR using Arabic-Indic digits
    await ctx.send(state, "لا، صحح رقم السجل: ١٠١٠٤٤٤٤٤٤")
    if sess.fields["cr_number"] == "1010444444":
        ctx.check(True, "Arabic-Indic CR digits normalized to Western")
    else:
        ctx.check(
            False,
            f"expected cr_number=1010444444 after correction, got {sess.fields['cr_number']!r}",
        )

    # Confirm
    await ctx.send(state, "نعم تأكيد")
    ctx.check(state["complaint_session"] is None, "complaint saved")
    ctx.finish()


# ------------------------------------------------------------------ #
# Group 6 — Fix 5 (retry counter)
# ------------------------------------------------------------------ #


async def s10_cr_three_strikes_collecting() -> None:
    ctx = header("S10", "CR 3 strikes during collecting → cancel")

    state = init_state()

    await ctx.send(state, "أبي أقدم شكوى")
    await ctx.send(state, "جرير")
    # current_field should be cr_number now

    # Strike 1.  IMPORTANT: this assumes the classifier returns "answer" for
    # "abc" (with value="abc"), which then fails CR validation and increments
    # the counter. If the classifier returns "unclear", the counter stays at 0
    # and this scenario fails — that itself is a real bug (garbage input
    # would never trigger the safety limit).
    await ctx.send(state, "abc")
    sess = state["complaint_session"]
    ctx.check(
        sess and sess.field_attempts["cr_number"] == 1,
        "counter at 1 after first invalid CR (see intent above if 0)",
    )

    # Strike 2
    await ctx.send(state, "123")
    ctx.check(
        sess and sess.field_attempts["cr_number"] == 2,
        "counter at 2 after second invalid CR (see intent above if not 2)",
    )

    # Strike 3 — should cancel
    r = await ctx.send(state, "xyz")
    ctx.check(state["complaint_session"] is None, "session destroyed at 3rd strike")
    ctx.check(
        "تطبيق وزارة التجارة" in r,
        "cancellation message mentions Ministry of Commerce app",
    )
    ctx.finish()


async def s11_counter_resets_on_success() -> None:
    ctx = header("S11", "Counter resets on success — fresh budget after correction")

    state = init_state()

    await ctx.send(state, "أبي أقدم شكوى")
    await ctx.send(state, "جرير")

    # Fail CR twice
    await ctx.send(state, "abc")
    await ctx.send(state, "abc2")
    sess = state["complaint_session"]
    ctx.check(sess.field_attempts["cr_number"] == 2, "counter at 2 after 2 failures")

    # Succeed
    await ctx.send(state, "1010555555")
    ctx.check(
        sess.field_attempts["cr_number"] == 0,
        "counter reset to 0 on successful storage",
    )
    ctx.check(
        sess.fields["cr_number"] == "1010555555",
        "valid CR was stored",
    )

    # Complete the complaint to confirmation
    await ctx.send(state, "ORD-11")
    await ctx.send(state, "2026-02-20")
    await ctx.send(state, "وصل المنتج تالفاً")
    # Should be confirming

    # In confirming, try correcting CR to invalid value — counter starts fresh at 0
    await ctx.send(state, "صحح رقم السجل إلى abc")
    ctx.check(
        sess.field_attempts["cr_number"] == 1,
        "counter at 1 in confirming (fresh after earlier reset)",
    )
    ctx.check(
        state["complaint_session"] is not None,
        "session still alive — not cancelled despite this being the 3rd lifetime failure",
    )

    # Cancel manually so we don't pollute
    await ctx.send(state, "ألغ الشكوى")
    ctx.finish()


async def s12_date_three_strikes_confirming() -> None:
    ctx = header("S12", "Date 3 strikes in confirming-correction → cancel")

    state = init_state()

    # Quickly fill all fields cleanly
    await ctx.send(state, "أبي أقدم شكوى")
    await ctx.send(state, "اكسترا")
    await ctx.send(state, "1010666666")
    await ctx.send(state, "ORD-12")
    await ctx.send(state, "2026-03-10")
    await ctx.send(state, "وصل المنتج بمواصفات مختلفة عن الموقع")
    ctx.check(
        state["complaint_session"] and state["complaint_session"].state == "confirming",
        "reached confirming state",
    )

    # Now correct date to invalid value 3 times in a row.
    # CRITICAL: these inputs must be such that _resolve_date returns None.
    # Relative dates like "البارح" (yesterday) WILL be resolved by the LLM
    # because it has today's date as context. We use clearly-unresolvable
    # strings instead: "xyz", "abc", "asdf".
    await ctx.send(state, "صحح التاريخ إلى xyz")
    sess = state["complaint_session"]
    if sess:
        ctx.check(
            sess.field_attempts["order_date"] == 1,
            "date counter at 1 in confirming",
        )

    await ctx.send(state, "صحح التاريخ إلى abcdef")
    sess = state["complaint_session"]
    if sess:
        ctx.check(
            sess.field_attempts["order_date"] == 2,
            "date counter at 2 in confirming",
        )

    r = await ctx.send(state, "صحح التاريخ إلى qwerty")
    ctx.check(state["complaint_session"] is None, "cancelled at 3rd date strike")
    ctx.check(
        "تطبيق وزارة التجارة" in r,
        "cancellation message present",
    )
    ctx.finish()


async def s13_per_field_counter_independence() -> None:
    ctx = header("S13", "Per-field counters are independent")

    state = init_state()

    await ctx.send(state, "أبي أقدم شكوى")
    await ctx.send(state, "نون")

    # Fail CR twice
    await ctx.send(state, "abc")
    await ctx.send(state, "def")
    sess = state["complaint_session"]
    ctx.check(sess.field_attempts["cr_number"] == 2, "cr_number counter at 2")
    ctx.check(sess.field_attempts["order_date"] == 0, "order_date counter still at 0")

    # Provide valid CR — counter resets
    await ctx.send(state, "1010777777")
    ctx.check(sess.field_attempts["cr_number"] == 0, "cr_number counter reset")

    # Continue and fail date twice — use clearly-unresolvable strings, not
    # relative dates the LLM might successfully resolve.
    await ctx.send(state, "ORD-13")
    await ctx.send(state, "asdfasdf")
    await ctx.send(state, "qwerty")
    ctx.check(sess.field_attempts["order_date"] == 2, "order_date counter at 2")
    ctx.check(
        state["complaint_session"] is not None,
        "session still alive — no field has hit 3 yet",
    )

    # Cancel to clean up
    await ctx.send(state, "ألغي الشكوى")
    ctx.finish()


# ------------------------------------------------------------------ #
# Group 7 — Fix 6 (legal_question intent)
# ------------------------------------------------------------------ #


async def s14_pure_legal_question_mid_complaint() -> None:
    ctx = header("S14", "Pure legal question mid-complaint")

    state = init_state()

    await ctx.send(state, "أبي أقدم شكوى")
    await ctx.send(state, "جرير")
    await ctx.send(state, "1010888888")
    # current_field is now order_id

    # Pivot to a legal question
    fields_before = dict(state["complaint_session"].fields)
    current_field_before = state["complaint_session"].current_field
    r = await ctx.send(state, "بس قبل ما أكمل، هل يحق لي أرجع المنتج؟")

    sess = state["complaint_session"]
    ctx.check(sess is not None, "session still alive after legal question")
    ctx.check(
        sess.current_field == current_field_before,
        f"current_field unchanged (still {current_field_before})",
    )
    ctx.check(
        dict(sess.fields) == fields_before,
        "no field was stored",
    )

    intent = ctx.last_intent()
    if intent and intent.get("intent") == "legal_question":
        ctx.check(True, "classifier correctly returned 'legal_question'")
    else:
        ctx.check(
            False,
            f"classifier returned {intent!r} instead of 'legal_question'",
        )

    # Response should contain the separator + return phrase
    ctx.check("---" in r, "response contains separator")
    ctx.check("نعود إلى الشكوى" in r, "response contains return-to-complaint phrase")
    ctx.review(
        "Read the response — is the legal answer correct and the re-prompt clear?"
    )
    ctx.finish()


async def s15_description_field_with_embedded_question() -> None:
    ctx = header("S15", "Description + embedded legal question → 'answer'")

    state = init_state()

    await ctx.send(state, "أبي أقدم شكوى")
    await ctx.send(state, "جرير")
    await ctx.send(state, "1010999999")
    await ctx.send(state, "ORD-15")
    await ctx.send(state, "2026-04-01")
    # current_field is now description

    # Provide description WITH an embedded legal question
    msg = "اشتريت لابتوب وما يشتغل، وأبي أعرف هل يحق لي أرجعه"
    await ctx.send(state, msg)

    sess = state["complaint_session"]

    # Check what the classifier did
    intent = ctx.last_intent()
    intent_label = intent.get("intent") if intent else None

    if intent_label == "answer":
        ctx.check(True, "classifier correctly returned 'answer' (description rule)")
        # Use substring match — the model may slightly modify the value
        # (trailing punctuation, normalized whitespace) without it being a bug.
        stored = sess.fields["description"] if sess else None
        ctx.check(
            stored is not None and "لابتوب" in stored and "أرجعه" in stored,
            f"description stored with content including 'لابتوب' and 'أرجعه' (got: {stored!r})",
        )
    else:
        ctx.review(
            f"classifier returned {intent_label!r} instead of 'answer'. "
            f"Description field exception may not be working as expected."
        )
    ctx.finish()


async def s16_legal_question_on_non_description_field() -> None:
    ctx = header("S16", "Legal question while asked for store_name")

    state = init_state()

    await ctx.send(state, "أبي أقدم شكوى")
    # current_field is now store_name

    # Ask a legal question instead of providing store name
    r = await ctx.send(state, "قبل ما أكمل، ايش الفرق بين الضمان والاسترجاع؟")

    sess = state["complaint_session"]
    ctx.check(sess is not None, "session still alive")
    ctx.check(
        sess.fields["store_name"] is None,
        "store_name still None — legal question not stored as field",
    )

    intent = ctx.last_intent()
    intent_label = intent.get("intent") if intent else None
    if intent_label == "legal_question":
        ctx.check(True, "classified as 'legal_question'")
    else:
        ctx.review(f"classifier returned {intent_label!r} — should be 'legal_question'")

    ctx.review("Did the bot answer the warranty-vs-return question reasonably?")
    ctx.finish()


async def s17_multiple_legal_questions_in_row() -> None:
    ctx = header("S17", "Multiple legal questions in a row, then field answer")

    state = init_state()

    await ctx.send(state, "أبي أقدم شكوى")
    await ctx.send(state, "اكسترا")
    # current_field = cr_number

    # Three legal-leaning questions in a row.
    # NOTE: these are META-questions about the form ("do I have to give you
    # the CR?") not pure legal questions ("what are my rights?"). The model
    # could legitimately classify them as `legal_question` OR `unclear` —
    # both lead to acceptable behavior (no field stored, re-prompt). The
    # test below only asserts field state and session aliveness, not the
    # specific intent.
    await ctx.send(state, "هل لازم أعطيك رقم السجل؟")
    await ctx.send(state, "ايش لو ما عندي رقم السجل؟")
    await ctx.send(state, "وايش الفرق بين رقم السجل والرقم الضريبي؟")

    sess = state["complaint_session"]
    ctx.check(sess is not None, "session alive after 3 ambiguous questions")
    ctx.check(
        sess.fields["cr_number"] is None,
        "cr_number still None — no question got stored as field",
    )

    # Now provide actual CR
    await ctx.send(state, "1010000001")
    ctx.check(
        sess.fields["cr_number"] == "1010000001",
        "field answer accepted after multiple ambiguous questions",
    )

    # Cancel to clean up
    await ctx.send(state, "ألغي")
    ctx.review(
        "Look at intent prints — for each question, was the model's choice "
        "(legal_question or unclear) sensible? Did the bot return to the CR prompt?"
    )
    ctx.finish()


async def s18_legal_question_at_confirmation() -> None:
    ctx = header("S18", "Legal question at confirmation step")

    state = init_state()

    # Fill all fields
    await ctx.send(state, "أبي أقدم شكوى")
    await ctx.send(state, "جرير")
    await ctx.send(state, "1010101010")
    await ctx.send(state, "ORD-18")
    await ctx.send(state, "2026-04-10")
    await ctx.send(state, "المنتج وصل ناقصاً")
    ctx.check(
        state["complaint_session"].state == "confirming",
        "reached confirming state",
    )

    # Ask a legal question instead of confirming
    r = await ctx.send(state, "قبل التأكيد، هل تقديم الشكوى يضمن لي تعويض؟")

    sess = state["complaint_session"]
    ctx.check(sess is not None, "session still alive")
    ctx.check(sess.state == "confirming", "state still 'confirming'")

    intent = ctx.last_intent()
    intent_label = intent.get("intent") if intent else None
    if intent_label == "legal_question":
        ctx.check(True, "classified as 'legal_question'")
    else:
        ctx.review(f"classifier returned {intent_label!r} — should be 'legal_question'")
    ctx.review("Did the bot answer the compensation question AND re-show the summary?")

    # Now confirm
    await ctx.send(state, "نعم تأكيد")
    ctx.check(state["complaint_session"] is None, "complaint saved after confirmation")
    ctx.finish()


async def s19_ambiguous_classifier_cases() -> None:
    ctx = header("S19", "Ambiguous classifier inputs — log only")

    state = init_state()

    await ctx.send(state, "أبي أقدم شكوى")
    await ctx.send(state, "جرير")
    # current_field = cr_number

    # Each turn's intent is auto-printed by send() — we just record review notes.

    # Case 1: meta-question about what's being asked (NOT a legal question)
    await ctx.send(state, "ما رقم السجل التجاري؟")
    ctx.review(
        "Case 1: 'ما رقم السجل التجاري؟' — user asking what the question MEANS, "
        "not a legal question. Review the intent printed above."
    )

    # Case 2: very short, vague — could be unclear or legal_question
    await ctx.send(state, "هل لي حق؟")
    ctx.review(
        "Case 2: 'هل لي حق؟' — short, vague. unclear or legal_question both "
        "defensible. Review the intent printed above."
    )

    # Case 3: looks like a correction with embedded legal question.
    # current_field is cr_number (user is being asked CR), they mention store.
    await ctx.send(state, "اسم المتجر جرير، بس عندي سؤال عن الضمان")
    ctx.review(
        "Case 3: store correction + legal question combined — any of "
        "correction/legal_question/unclear is defensible. Review above."
    )

    # Cleanup
    if state["complaint_session"] is not None:
        await ctx.send(state, "ألغي الشكوى")
    ctx.finish()


# ------------------------------------------------------------------ #
# Main runner with summary
# ------------------------------------------------------------------ #


SCENARIOS = [
    s1_pure_qa_single_turn,
    s2_pure_complaint_happy_path,
    s3_two_complaints_same_session,
    s4_pre_complaint_description_extracted,
    s5_qa_doesnt_pollute_new_complaint,
    s6_followup_needs_context,
    s7_standalone_not_over_rewritten,
    s8_rewriter_skips_complaint_history,
    s9_cr_variations,
    s10_cr_three_strikes_collecting,
    s11_counter_resets_on_success,
    s12_date_three_strikes_confirming,
    s13_per_field_counter_independence,
    s14_pure_legal_question_mid_complaint,
    s15_description_field_with_embedded_question,
    s16_legal_question_on_non_description_field,
    s17_multiple_legal_questions_in_row,
    s18_legal_question_at_confirmation,
    s19_ambiguous_classifier_cases,
]


async def main() -> None:
    print(SEP)
    print("  Phase 4.5 — Full Integration Test")
    print(f"  Log file: {LOG_PATH}")
    print(SEP)

    install_mocks()

    for scenario in SCENARIOS:
        try:
            await scenario()
        except Exception:
            print(f"\n  ✗ Scenario {scenario.__name__} crashed:")
            traceback.print_exc()
            # Continue with the next scenario regardless

    # ---------------- summary ----------------
    print(f"\n{SEP}")
    print("  SUMMARY")
    print(SEP)

    total_passed = 0
    total_failed = 0
    failed_details: list[tuple[str, str]] = []

    for sid, result in RESULTS.items():
        passed = sum(1 for ok, _ in result["checks"] if ok)
        failed = sum(1 for ok, _ in result["checks"] if not ok)
        total_passed += passed
        total_failed += failed
        print(
            f"  {sid:5} {result['title'][:50]:50}  "
            f"✓{passed} ✗{failed}  review:{len(result['human_review'])}"
        )
        for ok, desc in result["checks"]:
            if not ok:
                failed_details.append((sid, desc))

    print(f"\n  Automated checks: {total_passed} passed, {total_failed} failed")
    if failed_details:
        print("\n  Failed checks:")
        for sid, desc in failed_details:
            print(f"    {sid}: {desc}")

    if HUMAN_REVIEW:
        print(f"\n  Scenarios needing human review:")
        for sid in HUMAN_REVIEW:
            for note in RESULTS[sid]["human_review"]:
                print(f"    {sid}: {note}")

    print(f"\n  Full log written to: {LOG_PATH}")
    print(SEP)


if __name__ == "__main__":
    asyncio.run(main())
