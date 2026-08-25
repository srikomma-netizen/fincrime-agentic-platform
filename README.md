# Healthcare FinCrime Agentic Platform

[![CI](https://github.com/srikomma-netizen/fincrime-agentic-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/srikomma-netizen/fincrime-agentic-platform/actions/workflows/ci.yml)

**▶ Live demo:** https://srikomma-netizen.github.io/fincrime-agentic-platform/ — an
interactive, in-browser version of the Case Triage Console (no install needed).

An **agentic AI + analytics platform to strengthen financial-crimes and fraud
controls across healthcare payments and vendor transactions** — it detects
anomalous claims and payments, retrieves approved compliance policies, generates
structured case summaries, and coordinates investigator follow-up.

It combines **multi-agent orchestration, RAG, predictive ML, secure tool access,
human approval, and PHI/PII-aligned controls** — and runs **fully offline** out
of the box (deterministic mock LLM + TF-IDF fallback), while being wired for real
cloud LLMs (Anthropic / OpenAI) and FAISS + sentence-transformers via env vars.

---

## Architecture

```
                         ┌─────────────────────────────────────────────┐
   POST /cases  ──────▶  │            FastAPI  microservice             │
                         └───────────────────────┬─────────────────────┘
                                                 ▼
                                   ┌───────────────────────────┐
                                   │      SUPERVISOR AGENT      │  conditional routing,
                                   │  (state mgmt, retries,     │  retries, fallbacks,
                                   │   fallbacks, HITL pause)   │  human-in-the-loop
                                   └────────────┬──────────────┘
        ┌───────────────┬────────────┬─────────┼───────────┬───────────────┐
        ▼               ▼            ▼          ▼           ▼               ▼
   retrieval        risk         policy    summarizer    planner       escalation
    agent           agent        agent       agent        agent          agent
      │               │            │           │
      ▼               ▼            ▼           ▼
  MCP tools     IsolationForest  RAG        LLM (structured
 (RBAC + PHI     + XGBoost /    retriever   Pydantic/JSON
  masking +      GBM risk       (hybrid      output; grounded
  audit log)     scoring        search,      summary)
                                rerank,
                                metadata
                                filter, ACL)
```

### Components (all from the project spec)

| Capability | Where |
|---|---|
| LangGraph multi-agent architecture, supervisor + specialized agents | `src/app/agents/` (`graph.py`, `supervisor.py`, `specialized.py`) |
| Conditional routing, state management, retries, fallbacks | `agents/supervisor.py`, `agents/state.py` |
| Secure **MCP servers/tools** (claim history, payments, vendor, prior cases) | `src/app/mcp/server.py` |
| Minimum-necessary access, **PHI/PII masking, de-identification, RBAC** | `src/app/security/` |
| **Anomaly detection + risk scoring** (Isolation Forest, XGBoost) | `src/app/ml/` |
| Business rules + predictive scores → case severity & review path | `agents/specialized.py::risk_agent`, `planner_agent` |
| **RAG pipeline**: embeddings, vector search, hybrid search, metadata filter, re-rank, source validation, document-level ACL | `src/app/rag/` |
| Enterprise LLM for **grounded summarization** with **Pydantic/JSON-Schema structured output** | `src/app/llm/client.py`, `schemas.py` |
| Conditional routing: low → auto, medium → analyst, high → SIU | `agents/planner_agent`, `supervisor.route` |
| **Human-in-the-loop** checkpoints | `agents/hitl.py` |
| **FastAPI microservices** (start workflow, case status, human decisions) | `src/app/main.py` |
| **Docker / Cloud Run / GKE** deployment, health checks, audit logging | `Dockerfile`, `docker-compose.yml`, MCP audit log |

---

## Quickstart (offline, no API keys)

```bash
cd fincrime-agentic-platform
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# generate synthetic data, train models, build the policy vector index
python scripts/generate_synthetic_data.py
python scripts/train_models.py
python scripts/build_vector_index.py

# run the end-to-end demo (low-risk auto-clear vs high-risk escalation)
python scripts/demo.py
```

> **Optional heavy libraries** (LangGraph, FAISS, sentence-transformers, xgboost,
> and even scikit-learn) are detected at runtime, with graceful fallbacks so the
> platform runs anywhere:
>
> | If unavailable | Fallback used |
> |---|---|
> | LangGraph | built-in supervisor runner (identical routing) |
> | xgboost / scikit-learn | numpy logistic-regression risk model |
> | scikit-learn IsolationForest | numpy standardized-distance detector |
> | FAISS / sentence-transformers | NumPy cosine search + feature-hashing embeddings |
> | Anthropic / OpenAI key | deterministic, grounded mock summarizer |
>
> This means the **entire platform runs with just numpy + pydantic + fastapi** —
> verified on a locked-down Windows host where Application Control blocks scipy's
> compiled extensions (so scikit-learn/xgboost can't load at all).

### Run the API + web UI

```bash
uvicorn app.main:app --app-dir src --reload --port 8000
```

- **Case Triage Console (web UI):** http://localhost:8000/ — enter or load a
  claim, press **Run triage**, and watch the agent pipeline score risk, cite
  policy, and route the case (with a human-review pause on high-risk cases).
- **Interactive API docs:** http://localhost:8000/docs

### Example call

```bash
curl -s http://localhost:8000/cases -H "content-type: application/json" -d '{
  "role": "senior_investigator",
  "claim": {
    "claim_id": "CLM-1", "member_id": "MBR-1", "member_name": "Sam Chen",
    "member_ssn": "987-65-4321", "provider_id": "PROV-205",
    "provider_name": "Chen Clinic", "vendor_id": "VEND-1029", "amount": 48500,
    "procedure_code": "27447", "diagnosis_code": "M54.5",
    "claim_date": "2026-08-12", "place_of_service": "home",
    "prior_claims_30d": 7, "avg_provider_amount": 2200, "duplicate_flag": true
  }
}' | python -m json.tool
```

Returns a structured `CaseResult`: risk assessment, grounded summary with policy
citations, recommended actions, review path, a `requires_human` pause for
high-risk cases, and a full audit trail.

### Use a real LLM

```bash
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
export LLM_MODEL=claude-sonnet-4-5
```

The summarizer agent then produces the case summary via the model's structured
tool-output (validated against the same Pydantic `CaseSummary` schema).

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness + which LLM/RAG backends are active |
| GET | `/tools` | list registered MCP tools with typed schemas |
| POST | `/policies/search` | RAG retrieval (hybrid search + role-based ACL) |
| POST | `/cases` | run an agentic investigation for one claim |
| POST | `/cases/resume` | resume a paused case with a human decision |

## Testing

```bash
PYTHONPATH=src pytest -q
```

## Security & governance notes

- **No real PHI/PII** — all data under `data/synthetic/` is randomly generated.
- Direct identifiers are masked/tokenized **before** any payload reaches the LLM
  or a tool; re-identification is RBAC-gated and stays inside the trusted boundary.
- Every MCP tool call is written to an **append-only audit log** for model-risk
  and regulatory review.
- Risk thresholds (`RISK_LOW_MAX`, `RISK_HIGH_MIN`) are configurable via `.env`.

## See also

**[fincrime-mini](https://github.com/srikomma-netizen/fincrime-mini)** — a tiny,
single-file, zero-dependency version of the same idea, small enough to read in
one sitting. A good starting point before diving into this full platform.

## Disclaimer

"Baxter Health" is referenced only as the résumé project this repo re-creates.
This is an independent, synthetic reference implementation — it ships no
proprietary code, data, or models from any employer.
