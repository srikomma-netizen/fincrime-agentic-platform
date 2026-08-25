"""Supervisor agent — conditional routing over specialized agents.

Manages workflow state, decides which specialized agent runs next based on the
current state and business rules, and handles retries / fallbacks. The routing
is expressed as a pure function so it works identically whether the graph is
executed by LangGraph or by the built-in fallback runner.
"""
from __future__ import annotations

from .hitl import needs_human
from .state import CaseState

MAX_RETRIES = 2

# canonical happy-path order the supervisor drives through
_SEQUENCE = ["retrieval", "risk", "policy", "summarize", "plan"]


def route(state: CaseState) -> str:
    """Return the name of the next node to execute (or 'END')."""
    # 1. drive the linear investigation sequence first
    for step in _SEQUENCE:
        if not _done(state, step):
            return step

    # 2. after planning, enforce human-in-the-loop where required
    if not state.get("human_decision"):
        required, prompt = needs_human(state)
        if required:
            state["requires_human"] = True
            state["human_prompt"] = prompt
            return "END"  # pause and await a human decision

    # 3. high-risk cases route through escalation before completing
    if state.get("review_path") == "siu_escalation" and not _done(state, "escalate"):
        return "escalate"

    return "END"


def _done(state: CaseState, step: str) -> bool:
    return {
        "retrieval": "retrieved" in state,
        "risk": "risk" in state,
        "policy": "citations" in state,
        "summarize": "summary" in state,
        "plan": "review_path" in state,
        "escalate": any(a.get("step") == "escalation_agent"
                        for a in state.get("audit", [])),
    }[step]
