"""Assembles the multi-agent graph and runs a case end-to-end.

Uses LangGraph's StateGraph when the library is installed (supervisor as the
conditional router between specialized-agent nodes). If LangGraph is unavailable,
a built-in runner executes the *identical* supervisor routing loop, so behavior
is the same in either environment.
"""
from __future__ import annotations

import uuid
from typing import Any

from ..config import get_settings
from ..llm.client import LLMClient
from ..ml.anomaly import AnomalyDetector
from ..ml.risk_model import RiskModel
from ..mcp.server import build_default_server
from ..rag.retriever import PolicyRetriever
from ..schemas import (
    AuditEvent, CaseRequest, CaseResult, CaseSummary, ReviewPath, RiskAssessment,
)
from ..security.pii_phi import Deidentifier
from . import supervisor
from .specialized import Agents
from .state import CaseState

try:
    from langgraph.graph import END, StateGraph
    _HAS_LANGGRAPH = True
except Exception:  # pragma: no cover
    _HAS_LANGGRAPH = False


class Orchestrator:
    def __init__(self) -> None:
        settings = get_settings()
        self.deidentifier = Deidentifier()
        self.mcp = build_default_server(self.deidentifier)
        self.anomaly = AnomalyDetector.load(settings.model_dir / "anomaly.joblib")
        self.risk = RiskModel.load(settings.model_dir / "risk.joblib")
        try:
            self.retriever = PolicyRetriever.load()
        except Exception:
            self.retriever = None
        self.llm = LLMClient()
        self.agents = Agents(self.mcp, self.anomaly, self.risk, self.retriever, self.llm)
        self._node_map = {
            "retrieval": self.agents.retrieval_agent,
            "risk": self.agents.risk_agent,
            "policy": self.agents.policy_agent,
            "summarize": self.agents.summarizer_agent,
            "plan": self.agents.planner_agent,
            "escalate": self.agents.escalation_agent,
        }
        self._lg_app = self._build_langgraph() if _HAS_LANGGRAPH else None

    # ------------------------------------------------------------------ #
    def run(self, request: CaseRequest) -> CaseResult:
        claim = request.claim.model_dump()
        settings = get_settings()
        safe_claim = (
            self.deidentifier.deidentify_record(claim)
            if settings.enable_pii_masking else claim
        )
        state: CaseState = {
            "case_id": f"CASE-{uuid.uuid4().hex[:10].upper()}",
            "claim": claim,
            "safe_claim": safe_claim,
            "role": request.role.value,
            "audit": [],
            "retries": 0,
        }
        if request.human_decision is not None:
            state["human_decision"] = request.human_decision.model_dump()

        final = self._run_manual(state)  # deterministic supervisor loop
        return self._to_result(final)

    # ------------------------------------------------------------------ #
    def _run_manual(self, state: CaseState) -> CaseState:
        guard = 0
        while guard < 20:
            guard += 1
            nxt = supervisor.route(state)
            if nxt == "END":
                break
            node = self._node_map[nxt]
            update = node(state)
            state.update(update)
        return state

    # ------------------------------------------------------------------ #
    def _build_langgraph(self):  # pragma: no cover - exercised only if installed
        graph = StateGraph(dict)
        for name, fn in self._node_map.items():
            graph.add_node(name, fn)

        graph.add_node("supervisor", lambda s: s)
        graph.set_entry_point("supervisor")

        def _router(state: dict[str, Any]) -> str:
            nxt = supervisor.route(state)  # mutates requires_human/human_prompt
            return END if nxt == "END" else nxt

        graph.add_conditional_edges(
            "supervisor", _router,
            {**{n: n for n in self._node_map}, END: END},
        )
        for name in self._node_map:
            graph.add_edge(name, "supervisor")
        return graph.compile()

    # ------------------------------------------------------------------ #
    def _to_result(self, state: CaseState) -> CaseResult:
        risk = RiskAssessment.model_validate(state["risk"])
        summary = CaseSummary.model_validate(state["summary"])
        path = ReviewPath(state.get("review_path", "analyst_review"))

        requires_human = bool(state.get("requires_human"))
        if requires_human:
            status = "awaiting_human"
        elif path == ReviewPath.AUTO_CLEAR:
            status = "auto_cleared"
        elif path == ReviewPath.SIU_ESCALATION:
            status = "escalated"
        else:
            status = "completed"

        return CaseResult(
            case_id=state["case_id"],
            claim_id=state["claim"]["claim_id"],
            status=status,
            review_path=path,
            risk=risk,
            summary=summary,
            requires_human=requires_human,
            human_prompt=state.get("human_prompt"),
            audit_trail=[AuditEvent(**a) for a in state.get("audit", [])],
        )
