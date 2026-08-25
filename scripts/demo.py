"""End-to-end demo: run a low-risk and a high-risk claim through the platform
and print the structured results (no server needed)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from app.agents.graph import Orchestrator  # noqa: E402
from app.schemas import CaseRequest, ClaimRecord, Role  # noqa: E402

LOW = ClaimRecord(
    claim_id="CLM-DEMO-LOW", member_id="MBR-11111", member_name="Jordan Rivera",
    member_ssn="123-45-6789", provider_id="PROV-201", provider_name="Rivera Clinic",
    vendor_id="VEND-1000", amount=180.0, procedure_code="99213", diagnosis_code="I10",
    claim_date="2026-08-10", place_of_service="office", notes="Routine visit.",
    prior_claims_30d=1, avg_provider_amount=200.0, duplicate_flag=False,
)
HIGH = ClaimRecord(
    claim_id="CLM-DEMO-HIGH", member_id="MBR-22222", member_name="Sam Chen",
    member_ssn="987-65-4321", provider_id="PROV-205", provider_name="Chen Clinic",
    vendor_id="VEND-1029", amount=48500.0, procedure_code="27447", diagnosis_code="M54.5",
    claim_date="2026-08-12", place_of_service="home", notes="Resubmitted after denial.",
    prior_claims_30d=7, avg_provider_amount=2200.0, duplicate_flag=True,
)


def show(title: str, claim: ClaimRecord, role: Role) -> None:
    orch = Orchestrator()
    result = orch.run(CaseRequest(claim=claim, role=role))
    print(f"\n{'='*70}\n{title}\n{'='*70}")
    d = result.model_dump()
    print(f"case_id      : {d['case_id']}")
    print(f"status       : {d['status']}")
    print(f"review_path  : {d['review_path']}")
    print(f"risk         : score={d['risk']['risk_score']} "
          f"anomaly={d['risk']['anomaly_score']} level={d['risk']['risk_level']}")
    print(f"requires_human: {d['requires_human']}  {d.get('human_prompt') or ''}")
    print(f"headline     : {d['summary']['headline']}")
    print(f"actions      : {json.dumps(d['summary']['recommended_actions'], indent=0)}")
    print(f"citations    : {[c['policy_id'] for c in d['summary']['policy_citations']]}")
    print(f"audit steps  : {[a['step'] for a in d['audit_trail']]}")


if __name__ == "__main__":
    show("LOW-RISK CLAIM", LOW, Role.ANALYST)
    show("HIGH-RISK CLAIM", HIGH, Role.SENIOR_INVESTIGATOR)
