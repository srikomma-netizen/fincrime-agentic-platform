"""A secure, MCP-style tool server.

Agents never touch backend systems directly. They call registered tools through
this gateway, which enforces:
  * role-based access control (RBAC) per tool,
  * minimum-necessary de-identification of returned records,
  * an append-only audit log of every tool invocation.

This mirrors the MCP contract (typed tool schema + controlled invocation) while
staying dependency-free so it runs anywhere. A real deployment would expose the
same registry over the MCP protocol.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..config import get_settings
from ..schemas import Role
from ..security.pii_phi import Deidentifier
from ..security.rbac import check_tool_access

ToolFn = Callable[[dict[str, Any]], Any]


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: ToolFn


@dataclass
class MCPServer:
    deidentifier: Deidentifier
    tools: dict[str, Tool] = field(default_factory=dict)
    audit_log: list[dict[str, Any]] = field(default_factory=list)

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in self.tools.values()
        ]

    def call(self, name: str, args: dict[str, Any], role: Role) -> Any:
        check_tool_access(name, role)  # raises AccessDenied
        tool = self.tools.get(name)
        if tool is None:
            raise KeyError(f"Unknown tool '{name}'")
        result = tool.fn(args)
        # de-identify any record-shaped result before it leaves the boundary
        safe = self._deidentify(result)
        self.audit_log.append({
            "at": datetime.now(timezone.utc).isoformat(),
            "tool": name,
            "role": role.value,
            "args": args,
        })
        return safe

    def _deidentify(self, result: Any) -> Any:
        settings = get_settings()
        if not settings.enable_pii_masking:
            return result
        if isinstance(result, dict):
            return self.deidentifier.deidentify_record(result)
        if isinstance(result, list):
            return [self._deidentify(r) for r in result]
        return result


# --------------------------------------------------------------------------- #
# Backing "data warehouse" (synthetic). In production these would be secure
# service calls; here they read from local synthetic data.
# --------------------------------------------------------------------------- #
def _load_synthetic(name: str) -> list[dict[str, Any]]:
    path = get_settings().synthetic_dir / f"{name}.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_default_server(deidentifier: Deidentifier | None = None) -> MCPServer:
    server = MCPServer(deidentifier=deidentifier or Deidentifier())

    claims = _load_synthetic("claims")
    payments = _load_synthetic("payments")
    vendors = _load_synthetic("vendors")
    cases = _load_synthetic("prior_cases")

    def get_claim_history(args: dict[str, Any]) -> list[dict[str, Any]]:
        member_id = args.get("member_id")
        return [c for c in claims if c.get("member_id") == member_id][:25]

    def get_payment_records(args: dict[str, Any]) -> list[dict[str, Any]]:
        provider_id = args.get("provider_id")
        return [p for p in payments if p.get("provider_id") == provider_id][:25]

    def get_vendor_details(args: dict[str, Any]) -> dict[str, Any]:
        vendor_id = args.get("vendor_id")
        for v in vendors:
            if v.get("vendor_id") == vendor_id:
                return v
        return {}

    def get_prior_case_status(args: dict[str, Any]) -> list[dict[str, Any]]:
        member_id = args.get("member_id")
        provider_id = args.get("provider_id")
        return [
            c for c in cases
            if c.get("member_id") == member_id or c.get("provider_id") == provider_id
        ][:10]

    server.register(Tool(
        "get_claim_history",
        "Return prior claims for a de-identified member.",
        {"type": "object", "properties": {"member_id": {"type": "string"}},
         "required": ["member_id"]},
        get_claim_history,
    ))
    server.register(Tool(
        "get_payment_records",
        "Return recent payments to a provider.",
        {"type": "object", "properties": {"provider_id": {"type": "string"}},
         "required": ["provider_id"]},
        get_payment_records,
    ))
    server.register(Tool(
        "get_vendor_details",
        "Return risk metadata for a vendor.",
        {"type": "object", "properties": {"vendor_id": {"type": "string"}},
         "required": ["vendor_id"]},
        get_vendor_details,
    ))
    server.register(Tool(
        "get_prior_case_status",
        "Return prior investigation case outcomes for a member/provider.",
        {"type": "object", "properties": {
            "member_id": {"type": "string"}, "provider_id": {"type": "string"}}},
        get_prior_case_status,
    ))
    return server
