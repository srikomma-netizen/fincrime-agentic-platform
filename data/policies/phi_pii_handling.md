---
policy_id: SEC-PHI-003
title: PHI/PII Handling and Minimum-Necessary Access
access: restricted
---

# PHI/PII Handling and Minimum-Necessary Access

## Minimum necessary
Only the minimum information necessary to make a decision may be shared with any
model, tool, or downstream service. Direct identifiers — member name, SSN,
email, phone, and member ID — must be masked or tokenized before a payload is
passed to a language model.

## De-identification
Free-text notes must be scrubbed for identifiers. Re-identification is permitted
only inside the trusted boundary and only for roles with senior-investigator or
compliance permissions.

## Access control
Access to raw records is governed by role. Analysts may see de-identified data
and analyst-tagged policies. Restricted documents and re-identification are
limited to senior investigators and compliance.

## Auditability
Every tool invocation and every re-identification must be recorded in an
append-only audit log to support model-risk and regulatory review.
