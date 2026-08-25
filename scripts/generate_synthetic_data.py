"""Generate synthetic healthcare claims / payments / vendors / prior cases.

No real PHI/PII — everything here is randomly generated. Produces:
  data/synthetic/claims.json       (labeled: is_fraud) for model training
  data/synthetic/payments.json
  data/synthetic/vendors.json
  data/synthetic/prior_cases.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from app.config import get_settings  # noqa: E402

random.seed(42)

FIRST = ["Alex", "Jordan", "Sam", "Taylor", "Morgan", "Casey", "Riley", "Jamie"]
LAST = ["Rivera", "Chen", "Patel", "Okafor", "Nguyen", "Garcia", "Kim", "Brooks"]
PROCS = ["99213", "99285", "70450", "80053", "93000", "27447", "J1885"]
DIAGS = ["E11.9", "I10", "M54.5", "J45.909", "R07.9", "Z00.00"]
POS = ["office", "hospital", "home", "telehealth", "urgent_care"]


def _name() -> str:
    return f"{random.choice(FIRST)} {random.choice(LAST)}"


def _ssn() -> str:
    return f"{random.randint(100,899)}-{random.randint(10,99)}-{random.randint(1000,9999)}"


def gen_vendors(n: int = 30) -> list[dict]:
    vendors = []
    for i in range(n):
        tier = random.choices(["low", "medium", "high"], [0.6, 0.3, 0.1])[0]
        vendors.append({
            "vendor_id": f"VEND-{1000+i}",
            "vendor_name": f"{random.choice(LAST)} Medical Supplies",
            "risk_tier": tier,
            "prior_fraud_flag": tier == "high" and random.random() < 0.5,
            "country": random.choice(["US", "US", "US", "MX", "CA"]),
        })
    return vendors


def gen_claims(vendors: list[dict], n: int = 1200) -> list[dict]:
    provider_avg = {f"PROV-{200+i}": random.uniform(150, 4000) for i in range(40)}
    claims = []
    for i in range(n):
        provider_id = random.choice(list(provider_avg))
        avg = provider_avg[provider_id]
        vendor = random.choice(vendors)
        is_fraud = 0
        amount = max(20.0, random.gauss(avg, avg * 0.35))

        # inject fraud patterns into ~12% of records
        if random.random() < 0.12:
            is_fraud = 1
            pattern = random.random()
            if pattern < 0.4:              # outlier high-dollar
                amount = avg * random.uniform(4, 9)
            elif pattern < 0.7:            # duplicate
                pass
            # bias fraud toward high-risk vendors
            vendor = random.choice([v for v in vendors if v["risk_tier"] != "low"] or vendors)

        member_id = f"MBR-{random.randint(10000, 99999)}"
        prior = random.randint(0, 3) if not is_fraud else random.randint(2, 8)
        duplicate = bool(is_fraud and random.random() < 0.5)
        claims.append({
            "claim_id": f"CLM-{100000+i}",
            "member_id": member_id,
            "member_name": _name(),
            "member_ssn": _ssn(),
            "member_email": f"user{i}@example.com",
            "member_phone": f"({random.randint(200,899)}) {random.randint(200,899)}-{random.randint(1000,9999)}",
            "provider_id": provider_id,
            "provider_name": f"{random.choice(LAST)} Clinic",
            "vendor_id": vendor["vendor_id"],
            "amount": round(amount, 2),
            "procedure_code": random.choice(PROCS),
            "diagnosis_code": random.choice(DIAGS),
            "claim_date": f"2026-{random.randint(1,8):02d}-{random.randint(1,28):02d}",
            "place_of_service": random.choice(POS),
            "notes": "Routine submission." if not is_fraud else
                     "Resubmitted after prior denial; contact provider billing dept.",
            "prior_claims_30d": prior,
            "avg_provider_amount": round(avg, 2),
            "duplicate_flag": duplicate,
            "is_fraud": is_fraud,
        })
    return claims


def gen_payments(claims: list[dict]) -> list[dict]:
    payments = []
    for c in random.sample(claims, k=min(400, len(claims))):
        payments.append({
            "payment_id": f"PMT-{random.randint(500000, 599999)}",
            "provider_id": c["provider_id"],
            "vendor_id": c["vendor_id"],
            "amount": c["amount"],
            "paid_date": c["claim_date"],
            "status": random.choice(["paid", "paid", "held", "denied"]),
        })
    return payments


def gen_prior_cases(claims: list[dict]) -> list[dict]:
    cases = []
    for c in random.sample([x for x in claims if x["is_fraud"]], k=40):
        cases.append({
            "case_id": f"PRIOR-{random.randint(7000,7999)}",
            "member_id": c["member_id"],
            "provider_id": c["provider_id"],
            "outcome": random.choice(["confirmed_fraud", "cleared", "recovered"]),
        })
    return cases


def main() -> None:
    settings = get_settings()
    out = settings.synthetic_dir
    out.mkdir(parents=True, exist_ok=True)

    vendors = gen_vendors()
    claims = gen_claims(vendors)
    payments = gen_payments(claims)
    prior = gen_prior_cases(claims)

    for name, data in [("vendors", vendors), ("claims", claims),
                       ("payments", payments), ("prior_cases", prior)]:
        with open(out / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"wrote {len(data):>5} -> {out / (name + '.json')}")

    frauds = sum(c["is_fraud"] for c in claims)
    print(f"fraud rate: {frauds}/{len(claims)} = {frauds/len(claims):.1%}")


if __name__ == "__main__":
    main()
