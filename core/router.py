"""
router.py — Single entry point for all user messages.

This is the only module Streamlit calls. It owns the routing logic
between Q&A and complaint collection, and manages session state.

State dict (lives in Streamlit's session_state):
    {
        "history":           [],    # list of {"role": str, "content": str}
        "complaint_session": None,  # ComplaintSession instance or None
    }

The router receives this dict, modifies it in place, and returns
the response text. Streamlit displays whatever is returned.
"""

from __future__ import annotations

from core.classifier import START_COMPLAINT, classify
from core.complaint_session import ComplaintSession
from core.qa_pipeline import ask


def init_state() -> dict:
    """
    Return a fresh state dict.
    Called by Streamlit on first load, and by the test script
    to start a clean scenario.
    """
    return {
        "history": [],
        "complaint_session": None,
    }


async def handle(message: str, state: dict) -> str:
    """
    Process one user message and return the response text.

    Updates state in place:
        - always appends user message and assistant response to history
        - creates ComplaintSession when complaint intent detected
        - destroys ComplaintSession when status is "cancelled" or "saved"

    Args:
        message: the raw user message
        state:   the dict returned by init_state(), stored in session_state

    Returns:
        response_text to display in the UI
    """
    # Always append the user message to history first.
    # This means initialize() always receives the full conversation
    # including the message that triggered the complaint flow.
    state["history"].append({"role": "user", "content": message})

    response = await _route(message, state)

    # Append assistant response to history so future turns have full context
    state["history"].append({"role": "assistant", "content": response})

    return response


async def _route(message: str, state: dict) -> str:
    """Internal routing logic — separated from handle() for clarity."""

    # ------------------------------------------------------------------ #
    # Active complaint session — bypass classifier entirely
    # ------------------------------------------------------------------ #

    if state["complaint_session"] is not None:
        session: ComplaintSession = state["complaint_session"]
        status, response = await session.handle(message)

        if status in ("cancelled", "saved"):
            state["complaint_session"] = None

        return response

    # ------------------------------------------------------------------ #
    # No active session — classify and route
    # ------------------------------------------------------------------ #

    intent = await classify(message)

    if intent == START_COMPLAINT:
        session = ComplaintSession()
        # Pass the full history — includes the triggering message.
        # initialize() extracts whatever fields it can find before asking
        # for the first missing one.
        response = await session.initialize(state["history"])
        state["complaint_session"] = session
        return response

    # Default: legal question — route to Q&A pipeline
    return await ask(message)
