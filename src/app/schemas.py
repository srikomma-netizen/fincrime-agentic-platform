"""Pydantic v2 schemas — the structured-output contracts for the whole platform.

These mirror the JSON Schema used to constrain LLM outputs (risk levels,
supporting evidence, confidence scores, recommended actions, escalation reasons).
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ReviewPath(str, Enum):
    AUTO_CLEAR = "auto_clear"          # low risk -> automated handling
    ANALYST_REVIEW = "analyst_review"  # medium risk -> analyst queue
    SIU_ESCALATION = "siu_escalation"  # high/uncertain -> senior investigator


class Role(str, Enum):
    ANALYST = "analyst"
    SENIOR_INVESTIGATOR = "senior_investigator"
    COMPLIANCE = "compliance"
    SERVICE = "service"  # machine-to-machine


# --------------------------------------------------------------------------- #
# Domain / input records
# --------------------------------------------------------------------------- #
class ClaimRecord(BaseModel):
    """A healthcare payment / claim submitted for review."""
    claim_id: str
    member_id: str
    member_name: str
    member_ssn: Optional[str] = None
    provider_id: str
    provider_name: str
    vendor_id: Optional[str] = None
    amount: float
    procedure_code: str
    diagnosis_code: str
    claim_date: str
    place_of_service: str
    notes: str = ""
    # engineered/observed signals
    prior_claims_30d: int = 0
    avg_provider_amount: float = 0.0
    duplicate_flag: bool = False
    member_email: Optional[str] = None
    member_phone: Optional[str] = None


class CaseRequest(BaseModel):
    claim: ClaimRecord
    requested_by: str = "system"
    role: Role = Role.ANALYST
    # a human decision supplied to resume a paused (HITL) workflow
    human_decision: Optional["HumanDecision"] = None


# --------------------------------------------------------------------------- #
# Model / agent outputs
# --------------------------------------------------------------------------- #
class RiskAssessment(BaseModel):
    risk_score: float = Field(ge=0.0, le=1.0)
    anomaly_score: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    top_factors: list[str] = Field(default_factory=list)
    model_versions: dict[str, str] = Field(default_factory=dict)


class PolicyCitation(BaseModel):
    policy_id: str
    title: str
    snippet: str
    score: float


class CaseSummary(BaseModel):
    """The grounded, structured LLM output for a case."""
    risk_level: RiskLevel
    headline: str
    narrative: str
    supporting_evidence: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    policy_citations: list[PolicyCitation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    escalation_reason: Optional[str] = None
    grounded: bool = True


class HumanDecision(BaseModel):
    decision: Literal["approve", "reject", "request_info", "escalate"]
    decided_by: str
    role: Role
    rationale: str = ""
    decided_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AuditEvent(BaseModel):
    step: str
    detail: str
    at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CaseResult(BaseModel):
    case_id: str
    claim_id: str
    status: Literal["completed", "awaiting_human", "auto_cleared", "escalated"]
    review_path: ReviewPath
    risk: RiskAssessment
    summary: CaseSummary
    requires_human: bool = False
    human_prompt: Optional[str] = None
    audit_trail: list[AuditEvent] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


CaseRequest.model_rebuild()
