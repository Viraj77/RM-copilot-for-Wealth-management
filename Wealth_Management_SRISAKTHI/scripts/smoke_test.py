"""Quick smoke tests for the RM Copilot."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.guardrails.compliance import detect_licensed_advice_request
from src.ingestion.loader import load_documents
from src.models import RiskProfile
from src.tools.portfolio import load_client_portfolio, get_portfolio_summary
from src.tools.suitability import check_suitability_logic

docs = load_documents()
print(f"Loaded {len(docs)} documents")
for doc in docs:
    print(f"  - {doc.metadata['doc_id']} ({doc.metadata['type']}, {doc.metadata['sensitivity']})")

client = load_client_portfolio("C-204")
print(get_portfolio_summary(client))

r = check_suitability_logic("PG-003", RiskProfile.CONSERVATIVE)
print(f"PG-003/Conservative: suitable={r.suitable}, blocked={r.blocked_reason}")

r2 = check_suitability_logic("PG-002", RiskProfile.CONSERVATIVE)
print(f"PG-002/Conservative: suitable={r2.suitable}")

esc = detect_licensed_advice_request(
    "Recommend a personalized 4% withdrawal rate for client C-204 retirement"
)
print(f"Escalation detected: {esc}")

print("All smoke tests passed.")
