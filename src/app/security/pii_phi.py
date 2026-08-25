"""PHI / PII masking and de-identification.

Applies the *minimum-necessary* principle: before any claim payload is passed to
an LLM or a downstream tool, direct identifiers are masked / tokenized so the
model reasons over de-identified data. A reversible token map is kept in-process
so re-identification is possible only inside the trusted boundary (never sent out).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

# Order matters: most specific patterns first.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("EMAIL", re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")),
    ("PHONE", re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]*?){13,16}\b")),
    ("MRN", re.compile(r"\bMRN[-:\s]?\d{6,}\b", re.IGNORECASE)),
]

# Fields we always tokenize on structured records regardless of regex hits.
_DIRECT_IDENTIFIER_FIELDS = {
    "member_name": "NAME",
    "member_ssn": "SSN",
    "member_email": "EMAIL",
    "member_phone": "PHONE",
    "member_id": "MEMBERID",
}


def _token(kind: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8].upper()
    return f"[{kind}_{digest}]"


@dataclass
class Deidentifier:
    """Stateful de-identifier that remembers a reversible token map."""

    token_map: dict[str, str] = field(default_factory=dict)  # token -> original

    def mask_text(self, text: str) -> str:
        if not text:
            return text
        out = text
        for kind, pattern in _PATTERNS:
            def repl(m: re.Match[str], _kind=kind) -> str:
                original = m.group(0)
                tok = _token(_kind, original)
                self.token_map[tok] = original
                return tok
            out = pattern.sub(repl, out)
        return out

    def mask_value(self, kind: str, value: Any) -> str:
        if value is None or value == "":
            return value
        tok = _token(kind, str(value))
        self.token_map[tok] = str(value)
        return tok

    def deidentify_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Return a de-identified copy of a claim dict."""
        safe = dict(record)
        for field_name, kind in _DIRECT_IDENTIFIER_FIELDS.items():
            if field_name in safe and safe[field_name]:
                safe[field_name] = self.mask_value(kind, safe[field_name])
        # free-text fields still get regex scrubbing
        for text_field in ("notes",):
            if safe.get(text_field):
                safe[text_field] = self.mask_text(str(safe[text_field]))
        return safe

    def reidentify(self, text: str) -> str:
        """Only ever called inside the trusted boundary (e.g. investigator UI)."""
        out = text
        for tok, original in self.token_map.items():
            out = out.replace(tok, original)
        return out
