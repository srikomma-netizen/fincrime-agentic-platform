"""Specialized agents (graph nodes).

Each function is a node that reads the shared CaseState, does one job, and
returns a partial state update. They are deliberately small and testable:

  retrieval_agent   -> pull claim history / payments / vendor / prior cases (via MCP)
  risk_agent        -> anomaly + supervised risk scoring, level assignment
  policy_agent      -> RAG retrieval of approved compliance policies
  summarizer_agent  -> grounded, structured LLM case summary
  planner_agent     -> map risk level to a review path & recommended actions
  escalation_agent  -> compose escalation packet for SIU
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..config import get_settings
from ..llm.client import LLMClient
from ..ml.anomaly import AnomalyDetector
from ..ml.risk_model import RiskModel
from ..rag.retriever import PolicyRetriever
from ..schemas import CaseSummary, RiskLevel, Role
from .state import CaseState


def _audit(state: CaseState, step: str, detail: str) -> list[dict[str, Any]]:
    trail = list(state.get("audit", []))
    trail.append({"step": step, "detail": detail,
                  "at": datetime.now(timezone.utc).isoformat()})
    return trail


class Agents:
    """Holds loaded models/tools so nodes are cheap to invoke."""

    def __init__(self, mcp_server, anomaly: AnomalyDetector, risk: RiskModel,
                 retriever: PolicyRetriever | None, llm: LLMClient) -> None:
        self.mcp = mcp_server
        self.anomaly = anomaly
        self.risk = risk
        self.retriever = retriever
        self.llm = llm
        self.settings = get_settings()

    # ---------------------------------------------------------------- #
    def retrieval_agent(self, state: CaseState) -> CaseState:
        claim = state["claim"]
        role = Role(state.get("role", "analyst"))
        retrieved: dict[str, Any] = {}
        try:
            retrieved["claim_history"] = self.mcp.call(
                "get_claim_history", {"member_id": claim["member_id"]}, role)
            retrieved["payments"] = self.mcp.call(
                "get_payment_records", {"provider_id": claim["provider_id"]}, role)
            if claim.get("vendor_id"):
                retrieved["vendor"] = self.mcp.call(
                    "get_vendor_details", {"vendor_id": claim["vendor_id"]}, role)
        except PermissionError as exc:
            retrieved["access_error"] = str(exc)
        return {
            "retrieved": retrieved,
            "audit": _audit(state, "retrieval_agent",
                            f"fetched {len(retrieved)} record set(s) via MCP"),
        }

    # ---------------------------------------------------------------- #
    def risk_agent(self, state: CaseState) -> CaseState:
        claim = state["claim"]
        anomaly_score = self.anomaly.score(claim)
        risk_score = self.risk.predict_proba(claim)
        factors = self.risk.top_factors(claim)

        # blend supervised + unsupervised; anomaly can only raise concern
        blended = max(risk_score, 0.5 * risk_score + 0.5 * anomaly_score)
        if blended <= self.settings.risk_low_max:
            level = RiskLevel.LOW
        elif blended >= self.settings.risk_high_min:
            level = RiskLevel.HIGH
        else:
            level = RiskLevel.MEDIUM

        risk = {
            "risk_score": round(float(blended), 4),
            "anomaly_score": round(float(anomaly_score), 4),
            "risk_level": level.value,
            "top_factors": factors,
            "model_versions": {
                "risk": self.risk.model.__class__.__name__,
                "anomaly": "IsolationForest",
            },
        }
        return {
            "risk": risk,
            "audit": _audit(state, "risk_agent",
                            f"risk={risk['risk_score']} level={level.value}"),
        }

    # ---------------------------------------------------------------- #
    def policy_agent(self, state: CaseState) -> CaseState:
        role = Role(state.get("role", "analyst"))
        claim = state["claim"]
        query = (
            f"payment integrity and fraud policy for procedure {claim['procedure_code']} "
            f"amount {claim['amount']} risk {state.get('risk', {}).get('risk_level', '')}"
        )
        citations: list[dict[str, Any]] = []
        if self.retriever is not None:
            for c in self.retriever.retrieve(query, role=role, k=4):
                citations.append(c.model_dump())
        return {
            "citations": citations,
            "audit": _audit(state, "policy_agent",
                            f"retrieved {len(citations)} policy passage(s)"),
        }

    # ---------------------------------------------------------------- #
    def summarizer_agent(self, state: CaseState) -> CaseState:
        claim = state.get("safe_claim", state["claim"])
        risk = state["risk"]
        citations = state.get("citations", [])
        evidence = list(risk.get("top_factors", []))
        if state.get("retrieved", {}).get("vendor", {}).get("prior_fraud_flag"):
            evidence.append("Vendor has a prior confirmed-fraud flag.")

        schema = CaseSummary.model_json_schema()
        raw = self.llm.structured(
            system=(
                "You are a financial-crimes case summarizer for healthcare payments. "
                "Only use provided evidence and cited policies. Never invent facts."
            ),
            prompt="Summarize this case as structured JSON.",
            schema=schema,
            context={"claim": claim, "risk": risk,
                     "citations": citations, "evidence": evidence},
        )
        # validate / coerce through the schema
        summary = CaseSummary.model_validate(raw).model_dump()
        return {
            "summary": summary,
            "audit": _audit(state, "summarizer_agent",
                            f"grounded={summary['grounded']} conf={summary['confidence']}"),
        }

    # ---------------------------------------------------------------- #
    def planner_agent(self, state: CaseState) -> CaseState:
        level = state["risk"]["risk_level"]
        path = {"low": "auto_clear", "medium": "analyst_review",
                "high": "siu_escalation"}[level]
        return {
            "review_path": path,
            "audit": _audit(state, "planner_agent", f"review_path={path}"),
        }

    # ---------------------------------------------------------------- #
    def escalation_agent(self, state: CaseState) -> CaseState:
        summary = state.get("summary", {})
        reason = summary.get("escalation_reason") or "High-risk case flagged for SIU."
        return {
            "audit": _audit(state, "escalation_agent", f"escalated: {reason}"),
        }
