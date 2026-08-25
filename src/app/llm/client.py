"""Pluggable LLM client with structured-output (JSON-schema) support.

Providers:
  * mock       -> deterministic, offline, template-grounded (default)
  * anthropic  -> Claude models via the anthropic SDK
  * openai     -> GPT models via the openai SDK

All providers return a dict validated against the requested Pydantic schema by
the caller. The mock provider composes a grounded summary purely from the
retrieved evidence, so the platform runs end-to-end with no API key.
"""
from __future__ import annotations

import json
from typing import Any

from ..config import get_settings


class LLMClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.provider = self.settings.llm_provider
        self._impl = None
        if self.provider == "anthropic" and self.settings.anthropic_api_key:
            try:
                import anthropic
                self._impl = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
            except Exception:
                self.provider = "mock"
        elif self.provider == "openai" and self.settings.openai_api_key:
            try:
                from openai import OpenAI
                self._impl = OpenAI(api_key=self.settings.openai_api_key)
            except Exception:
                self.provider = "mock"
        else:
            self.provider = "mock"

    def structured(self, system: str, prompt: str, schema: dict[str, Any],
                   context: dict[str, Any]) -> dict[str, Any]:
        if self.provider == "anthropic":
            return self._anthropic(system, prompt, schema)
        if self.provider == "openai":
            return self._openai(system, prompt, schema)
        return self._mock(context)

    # ------------------------------------------------------------------ #
    def _anthropic(self, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        tool = {
            "name": "emit_case_summary",
            "description": "Return the structured case summary.",
            "input_schema": schema,
        }
        resp = self._impl.messages.create(
            model=self.settings.llm_model,
            max_tokens=1500,
            system=system,
            tools=[tool],
            tool_choice={"type": "tool", "name": "emit_case_summary"},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                return dict(block.input)
        raise RuntimeError("Anthropic returned no structured tool output")

    def _openai(self, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        resp = self._impl.chat.completions.create(
            model=self.settings.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "case_summary", "schema": schema, "strict": False},
            },
        )
        return json.loads(resp.choices[0].message.content)

    # ------------------------------------------------------------------ #
    def _mock(self, context: dict[str, Any]) -> dict[str, Any]:
        """Grounded, deterministic summary built only from supplied evidence."""
        claim = context["claim"]
        risk = context["risk"]
        citations = context.get("citations", [])
        evidence = context.get("evidence", [])

        level = risk["risk_level"]
        actions = {
            "low": ["Auto-clear and record decision in case log."],
            "medium": [
                "Assign to fraud analyst queue for standard review.",
                "Request supporting documentation from provider if amount unverified.",
            ],
            "high": [
                "Escalate to Special Investigations Unit (SIU).",
                "Place a temporary payment hold pending investigation.",
                "Cross-check vendor against prior confirmed-fraud cases.",
            ],
        }[level]

        headline = (
            f"{level.upper()} risk on claim {claim['claim_id']} "
            f"(${claim['amount']:,.2f}, {claim['procedure_code']})"
        )
        narrative = (
            f"Claim {claim['claim_id']} from provider {claim['provider_name']} was scored "
            f"at risk {risk['risk_score']:.2f} and anomaly {risk['anomaly_score']:.2f}. "
            f"Primary drivers: {', '.join(risk.get('top_factors', [])) or 'n/a'}. "
            f"Assessment is grounded in {len(citations)} retrieved policy passage(s)."
        )
        escalation = None
        if level == "high":
            escalation = (
                "Risk score exceeds the high-risk threshold and anomaly signals indicate "
                "irregular payment behavior requiring senior-investigator judgment."
            )
        confidence = round(0.6 + 0.35 * (1.0 if citations else 0.0)
                           - 0.1 * (1.0 if level == "medium" else 0.0), 3)
        return {
            "risk_level": level,
            "headline": headline,
            "narrative": narrative,
            "supporting_evidence": evidence or risk.get("top_factors", []),
            "recommended_actions": actions,
            "policy_citations": citations,
            "confidence": max(0.0, min(1.0, confidence)),
            "escalation_reason": escalation,
            "grounded": bool(citations),
        }
