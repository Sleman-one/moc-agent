"""
test_complaint.py — End-to-end complaint flow test.

Tests three scenarios against the live RunPod stack.
Requires: PostgreSQL running locally, vLLM + BGE-M3 running on RunPod.

Run with:
    uv run -m tests.test_complaint

Assertions verify state machine correctness — session created, destroyed,
and transitioned at the right moments. Response text is printed for manual
review — LLM output is not asserted because it varies between runs.

Scenarios:
    1. Direct complaint — all fields collected one by one, including a
       date retry (user gives vague date, system asks for clarification,
       user provides ISO format), confirmed and saved to DB.
    2. Mid-conversation switch — Q&A first, then complaint with extraction
       from history, then cancel, then verify back to Q&A mode.
    3. Cancel at confirmation — reach summary step, correct a field,
       then cancel.
"""

from __future__ import annotations

import asyncio

from core.router import handle, init_state

# ------------------------------------------------------------------ #
# Formatting helpers
# ------------------------------------------------------------------ #

SEP = "─" * 60


def header(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def turn(n: int, role: str, text: str) -> None:
    label = "User     " if role == "user" else "Assistant"
    print(f"\n[Turn {n}] {label}: {text}")


def ok(msg: str) -> None:
    print(f"  ✓ {msg}")


# ------------------------------------------------------------------ #
# Scenario 1 — Direct complaint, date retry, confirmed and saved
# ------------------------------------------------------------------ #


async def scenario_direct_complaint() -> None:
    header("Scenario 1 — Direct complaint, date retry, confirmed and saved")

    state = init_state()

    # Turn 1: trigger complaint
    msg = "أريد تقديم شكوى"
    turn(1, "user", msg)
    r = await handle(msg, state)
    turn(1, "assistant", r)
    assert state["complaint_session"] is not None
    ok("ComplaintSession created")

    # Turn 2: store name
    msg = "noon"
    turn(2, "user", msg)
    r = await handle(msg, state)
    turn(2, "assistant", r)

    # Turn 3: CR number
    msg = "1010123456"
    turn(3, "user", msg)
    r = await handle(msg, state)
    turn(3, "assistant", r)

    # Turn 4: order ID
    msg = "ORD-2026-001"
    turn(4, "user", msg)
    r = await handle(msg, state)
    turn(4, "assistant", r)

    # Turn 5: vague date — system should ask for clarification
    msg = "الأسبوع الماضي"
    turn(5, "user", msg)
    r = await handle(msg, state)
    turn(5, "assistant", r)
    assert state["complaint_session"] is not None
    assert state["complaint_session"].fields["order_date"] is None
    ok("Date resolution failed correctly — field still None")

    # Turn 6: proper ISO date after clarification
    msg = "2026-04-23"
    turn(6, "user", msg)
    r = await handle(msg, state)
    turn(6, "assistant", r)
    assert state["complaint_session"].fields["order_date"] == "2026-04-23"
    ok("Date accepted after retry")

    # Turn 7: description — should trigger confirmation summary
    msg = "استلمت منتجاً معطوباً ولم يرد المتجر على شكواي بعد أسبوعين"
    turn(7, "user", msg)
    r = await handle(msg, state)
    turn(7, "assistant", r)
    assert state["complaint_session"] is not None
    assert state["complaint_session"].state == "confirming"
    ok("Reached confirmation step")

    # Turn 8: confirm
    msg = "نعم، تأكيد"
    turn(8, "user", msg)
    r = await handle(msg, state)
    turn(8, "assistant", r)
    assert state["complaint_session"] is None
    ok("Session destroyed after save")
    ok("Scenario 1 passed")


# ------------------------------------------------------------------ #
# Scenario 2 — Q&A first, mid-conversation switch, then cancel,
#              then verify back to Q&A mode
# ------------------------------------------------------------------ #


async def scenario_mid_conversation() -> None:
    header("Scenario 2 — Mid-conversation switch to complaint, then cancel")

    state = init_state()

    # Turn 1: Q&A question
    msg = "ما هي حقوقي إذا وصلني منتج معطوب؟"
    turn(1, "user", msg)
    r = await handle(msg, state)
    turn(1, "assistant", r)
    assert state["complaint_session"] is None
    ok("Routed to Q&A — no session created")

    # Turn 2: switch to complaint with store name and CR in the message
    # extraction should pick up fields from this message
    msg = "أبي أشتكي على متجر amazon، رقم سجله 1010654321 ورقم طلبي ORD-999"
    turn(2, "user", msg)
    r = await handle(msg, state)
    turn(2, "assistant", r)
    assert state["complaint_session"] is not None
    ok("Switched to complaint mode — session created")
    ok(f"Fields extracted: {state['complaint_session'].fields}")

    # Continue collecting remaining missing fields
    n = 3
    session = state["complaint_session"]
    answers = {
        "store_name": "amazon",
        "cr_number": "1010654321",
        "order_id": "ORD-999",
        "order_date": "2026-04-15",
        "description": "دفعت مقابل المنتج ولم يصلني",
    }
    while session is not None and session.state == "collecting":
        field = session.current_field
        msg = answers.get(field, "لا أعرف")
        turn(n, "user", msg)
        r = await handle(msg, state)
        turn(n, "assistant", r)
        n += 1
        session = state["complaint_session"]

    if state["complaint_session"] is not None:
        assert state["complaint_session"].state == "confirming"
        ok("Reached confirmation step")

    # Cancel at confirmation
    msg = "إلغاء"
    turn(n, "user", msg)
    r = await handle(msg, state)
    turn(n, "assistant", r)
    assert state["complaint_session"] is None
    ok("Session destroyed after cancel")

    # Verify next message routes to Q&A, not complaint
    msg = "ما هي مدة الضمان على المنتجات؟"
    turn(n + 1, "user", msg)
    r = await handle(msg, state)
    turn(n + 1, "assistant", r)
    assert state["complaint_session"] is None
    ok("Back to Q&A mode after cancel — no session created")
    ok("Scenario 2 passed")


# ------------------------------------------------------------------ #
# Scenario 3 — Reach confirmation, correct a field, then cancel
# ------------------------------------------------------------------ #


async def scenario_cancel_at_confirmation() -> None:
    header("Scenario 3 — Reach confirmation, correct a field, then cancel")

    state = init_state()

    # Collect all fields quickly with valid data
    exchanges = [
        "أريد تقديم شكوى",
        "jarir",
        "1010987654",
        "ORD-555",
        "2026-04-10",
        "المنتج لم يصل في الوقت المحدد وانتهت المناسبة",
    ]

    for n, msg in enumerate(exchanges, start=1):
        turn(n, "user", msg)
        r = await handle(msg, state)
        turn(n, "assistant", r)

    assert state["complaint_session"] is not None
    assert state["complaint_session"].state == "confirming"
    ok("Reached confirmation step")

    # Correct a field at confirmation step
    n = len(exchanges) + 1
    msg = "اسم المتجر خاطئ، هو جرير وليس jarir"
    turn(n, "user", msg)
    r = await handle(msg, state)
    turn(n, "assistant", r)
    assert state["complaint_session"] is not None
    assert state["complaint_session"].state == "confirming"
    ok(
        f"Correction applied — store_name: {state['complaint_session'].fields['store_name']}"
    )

    # Cancel
    n += 1
    msg = "بطلت، ما أبي أشتكي"
    turn(n, "user", msg)
    r = await handle(msg, state)
    turn(n, "assistant", r)
    assert state["complaint_session"] is None
    ok("Session destroyed after cancel at confirmation")
    ok("Scenario 3 passed")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #


async def main() -> None:
    print(SEP)
    print("  Phase 4 — Complaint flow end-to-end test")
    print(SEP)

    await scenario_direct_complaint()
    await scenario_mid_conversation()
    await scenario_cancel_at_confirmation()

    print(f"\n{SEP}")
    print("  All scenarios passed")
    print(SEP)


if __name__ == "__main__":
    asyncio.run(main())
