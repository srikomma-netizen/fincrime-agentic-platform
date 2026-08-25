"""Role-based access control for MCP tools and case actions.

Enforces the minimum-necessary access model: each MCP tool declares the roles
allowed to invoke it, and each field-level scope declares who may see raw values.
"""
from __future__ import annotations

from ..schemas import Role

# tool name -> allowed roles
_TOOL_ACL: dict[str, set[Role]] = {
    "get_claim_history": {Role.ANALYST, Role.SENIOR_INVESTIGATOR, Role.COMPLIANCE, Role.SERVICE},
    "get_payment_records": {Role.ANALYST, Role.SENIOR_INVESTIGATOR, Role.COMPLIANCE, Role.SERVICE},
    "get_vendor_details": {Role.ANALYST, Role.SENIOR_INVESTIGATOR, Role.COMPLIANCE, Role.SERVICE},
    "get_prior_case_status": {Role.SENIOR_INVESTIGATOR, Role.COMPLIANCE, Role.SERVICE},
    # raw re-identification is restricted
    "reidentify_member": {Role.SENIOR_INVESTIGATOR, Role.COMPLIANCE},
}


class AccessDenied(PermissionError):
    pass


def check_tool_access(tool_name: str, role: Role) -> None:
    allowed = _TOOL_ACL.get(tool_name)
    if allowed is None:
        raise AccessDenied(f"Unknown tool '{tool_name}'")
    if role not in allowed:
        raise AccessDenied(
            f"Role '{role.value}' is not permitted to call '{tool_name}'"
        )


def can_reidentify(role: Role) -> bool:
    return role in _TOOL_ACL["reidentify_member"]
