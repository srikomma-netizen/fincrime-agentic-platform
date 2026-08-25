"""Human-in-the-loop checkpoints.

A workflow pauses when a decision needs professional judgment:
  * high-risk / uncertain cases,
  * low-confidence LLM summaries,
  * missing information (access errors, empty retrievals),
  * any action flagged as sensitive.
"""
from __future__ import annotations

from .state import CaseState

LOW_CONFIDENCE = 0.55


def needs_human(state: CaseState) -> tuple[bool, str]:
    risk = state.get("risk", {})
    summary = state.get("summary", {})
    retrieved = state.get("retrieved", {})

    if risk.get("risk_level") == "high":
        return True, "High-risk case requires senior-investigator approval before action."
    if summary.get("confidence", 1.0) < LOW_CONFIDENCE:
        return True, "Summary confidence is low; analyst confirmation required."
    if retrieved.get("access_error"):
        return True, "Data access was restricted; human review required for missing context."
    if not summary.get("grounded", True):
        return True, "Summary is not grounded in retrieved policy; manual review required."
    return False, ""
