"""
router.py — Single entry point for all user messages.

This is the only module Streamlit calls. It owns the routing logic
between Q&A and complaint collection, and manages session state.

State dict (lives in Streamlit's session_state):
    {
        "history":           [],    # list of {"role": str, "content": str, "source": str}
        "complaint_session": None,  # ComplaintSession instance or None
    }

Each history entry carries a `source` tag:
    "qa"        — message processed by the Q&A path
    "complaint" — message processed inside a complaint session

The user message is tagged based on whether a complaint session was
active when it arrived. The assistant message is tagged based on
which path produced the response — _route() returns the source so
handle() can apply it.

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
        - always appends user message and assistant response to history,
          each tagged with a `source` field
        - creates ComplaintSession when complaint intent detected
        - destroys ComplaintSession when status is "cancelled" or "saved"

    Args:
        message: the raw user message
        state:   the dict returned by init_state(), stored in session_state

    Returns:
        response_text to display in the UI
    """
    # Tag the user message based on whether a complaint session is active
    # at the moment of arrival. This is fixed at arrival time — even if the
    # message ends up triggering a new complaint session, the message itself
    # arrived while no session was active, so it's tagged "qa".
    user_source = "complaint" if state["complaint_session"] is not None else "qa"
    state["history"].append({"role": "user", "content": message, "source": user_source})

    response, response_source = await _route(message, state)

    # Tag the assistant message based on which path produced the response.
    state["history"].append(
        {"role": "assistant", "content": response, "source": response_source}
    )

    return response


async def _route(message: str, state: dict) -> tuple[str, str]:
    """
    Internal routing logic — separated from handle() for clarity.

    Returns:
        (response_text, source) where source is "qa" or "complaint".
        The source identifies which path produced the response, so the
        caller can tag the assistant message in history correctly.
    """

    # ------------------------------------------------------------------ #
    # Active complaint session — bypass classifier entirely
    # ------------------------------------------------------------------ #

    if state["complaint_session"] is not None:
        session: ComplaintSession = state["complaint_session"]
        # Pass history so the session can route a mid-complaint legal
        # question through the Q&A pipeline (with rewriter context).
        status, response = await session.handle(message, state["history"])

        if status in ("cancelled", "saved"):
            state["complaint_session"] = None

        return response, "complaint"

    # ------------------------------------------------------------------ #
    # No active session — classify and route
    # ------------------------------------------------------------------ #

    intent = await classify(message)

    if intent == START_COMPLAINT:
        session = ComplaintSession()
        # Pass the full history — includes the triggering message.
        # initialize() decides how much of it to actually use.
        response = await session.initialize(state["history"])
        state["complaint_session"] = session
        return response, "complaint"

    # Default: legal question — route to Q&A pipeline.
    # Pass history so the rewriter can resolve follow-ups using prior turns.
    return await ask(message, state["history"]), "qa"
