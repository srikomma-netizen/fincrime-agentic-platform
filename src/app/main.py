"""FastAPI microservice — the platform's entrypoint.

Endpoints
  GET  /health              liveness + backend/provider report
  GET  /tools               list registered MCP tools (typed schemas)
  POST /policies/search     RAG retrieval demo (hybrid search + RBAC)
  POST /cases               run an agentic investigation for one claim
  POST /cases/resume        resume a paused case with a human decision
"""
from __future__ import annotations

from functools import lru_cache

from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .config import get_settings
from .schemas import CaseRequest, CaseResult, PolicyCitation, Role

app = FastAPI(
    title="Healthcare FinCrime Agentic Platform",
    version="1.0.0",
    description="Agentic AI + analytics for financial-crimes & fraud controls "
                "across healthcare payments and vendor transactions.",
)

_WEB_DIR = Path(__file__).resolve().parents[2] / "web"


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the Case Triage Console UI."""
    return FileResponse(_WEB_DIR / "index.html")


@lru_cache
def _orchestrator():
    # imported lazily so the app can boot even before models are trained
    from .agents.graph import Orchestrator
    return Orchestrator()


@app.get("/health")
def health() -> dict:
    from .llm.client import LLMClient
    from .rag.embeddings import Embedder
    settings = get_settings()
    try:
        emb_backend = Embedder().backend
    except Exception:
        emb_backend = "unavailable"
    return {
        "status": "ok",
        "llm_provider": LLMClient().provider,
        "rag_embedding_backend": emb_backend,
        "pii_masking": settings.enable_pii_masking,
        "risk_thresholds": {"low_max": settings.risk_low_max,
                            "high_min": settings.risk_high_min},
    }


@app.get("/tools")
def tools() -> dict:
    return {"tools": _orchestrator().mcp.list_tools()}


class PolicyQuery(BaseModel):
    query: str
    role: Role = Role.ANALYST
    k: int = 4


@app.post("/policies/search", response_model=list[PolicyCitation])
def policy_search(q: PolicyQuery) -> list[PolicyCitation]:
    retriever = _orchestrator().retriever
    if retriever is None:
        raise HTTPException(503, "Vector index not built. Run scripts/build_vector_index.py")
    return retriever.retrieve(q.query, role=q.role, k=q.k)


@app.post("/cases", response_model=CaseResult)
def create_case(request: CaseRequest) -> CaseResult:
    try:
        return _orchestrator().run(request)
    except FileNotFoundError as exc:
        raise HTTPException(503, f"Models not trained: {exc}. Run scripts/train_models.py")


@app.post("/cases/resume", response_model=CaseResult)
def resume_case(request: CaseRequest = Body(...)) -> CaseResult:
    if request.human_decision is None:
        raise HTTPException(400, "human_decision is required to resume a case")
    return _orchestrator().run(request)
