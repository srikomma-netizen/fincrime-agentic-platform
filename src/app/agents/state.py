"""Shared workflow state passed between agents (LangGraph channel dict)."""
from __future__ import annotations

from typing import Any, TypedDict


class CaseState(TypedDict, total=False):
    # inputs
    case_id: str
    claim: dict[str, Any]           # raw claim (trusted boundary only)
    safe_claim: dict[str, Any]      # de-identified claim (LLM/tool boundary)
    role: str

    # retrieved context
    retrieved: dict[str, Any]       # claim history, payments, vendor, prior cases
    citations: list[dict[str, Any]]

    # assessment
    risk: dict[str, Any]

    # decisioning
    review_path: str
    summary: dict[str, Any]

    # human-in-the-loop
    requires_human: bool
    human_prompt: str
    human_decision: dict[str, Any]

    # bookkeeping
    audit: list[dict[str, Any]]
    next: str                       # supervisor's routing decision
    retries: int
