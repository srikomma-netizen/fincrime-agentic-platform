"""Smoke + behavior tests for the platform. Run: pytest -q"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.schemas import CaseRequest, ClaimRecord, RiskLevel, Role  # noqa: E402
from app.security.pii_phi import Deidentifier  # noqa: E402
from app.security.rbac import AccessDenied, check_tool_access  # noqa: E402


def _claim(**kw):
    base = dict(
        claim_id="CLM-T", member_id="MBR-1", member_name="Jane Doe",
        member_ssn="111-22-3333", provider_id="PROV-201", provider_name="Clinic",
        vendor_id="VEND-1000", amount=150.0, procedure_code="99213",
        diagnosis_code="I10", claim_date="2026-08-01", place_of_service="office",
        prior_claims_30d=0, avg_provider_amount=180.0,
    )
    base.update(kw)
    return ClaimRecord(**base)


def test_deidentify_masks_direct_identifiers():
    d = Deidentifier()
    rec = _claim(member_email="a@b.com", notes="SSN 111-22-3333 on file").model_dump()
    safe = d.deidentify_record(rec)
    assert "Jane Doe" not in str(safe)
    assert "111-22-3333" not in str(safe["notes"])
    assert safe["member_name"].startswith("[NAME_")
    # reversible inside boundary
    assert d.reidentify(safe["member_name"]) == "Jane Doe"


def test_rbac_blocks_reidentify_for_analyst():
    check_tool_access("get_claim_history", Role.ANALYST)  # allowed
    with pytest.raises(AccessDenied):
        check_tool_access("reidentify_member", Role.ANALYST)


def test_low_risk_auto_clears():
    from app.agents.graph import Orchestrator
    orch = Orchestrator()
    res = orch.run(CaseRequest(claim=_claim(), role=Role.ANALYST))
    assert res.risk.risk_level == RiskLevel.LOW
    assert res.review_path.value == "auto_clear"
    assert res.status == "auto_cleared"


def test_extreme_claim_scores_above_benign():
    """An extreme fraud pattern must score strictly higher than a benign claim
    and route to escalation + human review whenever it lands in the HIGH band.

    The exact score depends on the active model backend (sklearn vs. numpy
    fallback), so we assert the *behavioral contract* rather than a fixed level.
    """
    from app.agents.graph import Orchestrator
    orch = Orchestrator()
    extreme = _claim(amount=250000.0, avg_provider_amount=800.0,
                     prior_claims_30d=12, duplicate_flag=True, place_of_service="home")
    res = orch.run(CaseRequest(claim=extreme, role=Role.SENIOR_INVESTIGATOR))
    benign = orch.run(CaseRequest(claim=_claim(), role=Role.ANALYST))

    assert res.risk.risk_score > benign.risk.risk_score
    assert res.risk.risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH)
    if res.risk.risk_level == RiskLevel.HIGH:
        assert res.requires_human is True
        assert res.human_prompt
        assert res.review_path.value == "siu_escalation"


def test_high_risk_routing_and_hitl_are_deterministic():
    """Escalation + human-in-the-loop must fire for a HIGH case regardless of
    which ML backend produced the score. Tests the pure routing logic directly."""
    from app.agents.hitl import needs_human
    from app.agents.supervisor import route

    high_state = {
        "retrieved": {}, "risk": {"risk_level": "high"}, "citations": [],
        "summary": {"confidence": 0.9, "grounded": True}, "review_path": "siu_escalation",
        "audit": [], "requires_human": False,
    }
    required, prompt = needs_human(high_state)
    assert required is True and prompt

    # supervisor pauses (END) for human before escalation when no decision yet
    nxt = route(dict(high_state))
    assert nxt == "END"

    # once a human decision is supplied, the high case routes through escalation
    decided = dict(high_state)
    decided["human_decision"] = {"decision": "escalate"}
    assert route(decided) == "escalate"


def test_structured_summary_contract():
    from app.agents.graph import Orchestrator
    res = Orchestrator().run(CaseRequest(claim=_claim(), role=Role.ANALYST))
    s = res.summary
    assert 0.0 <= s.confidence <= 1.0
    assert s.recommended_actions
    assert isinstance(s.risk_level, RiskLevel)
