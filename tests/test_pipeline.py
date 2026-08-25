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


def test_high_risk_escalates_and_requests_human():
    from app.agents.graph import Orchestrator
    orch = Orchestrator()
    claim = _claim(amount=60000.0, avg_provider_amount=2000.0,
                   prior_claims_30d=8, duplicate_flag=True, place_of_service="home")
    res = orch.run(CaseRequest(claim=claim, role=Role.SENIOR_INVESTIGATOR))
    assert res.risk.risk_level == RiskLevel.HIGH
    assert res.requires_human is True
    assert res.human_prompt


def test_structured_summary_contract():
    from app.agents.graph import Orchestrator
    res = Orchestrator().run(CaseRequest(claim=_claim(), role=Role.ANALYST))
    s = res.summary
    assert 0.0 <= s.confidence <= 1.0
    assert s.recommended_actions
    assert isinstance(s.risk_level, RiskLevel)
