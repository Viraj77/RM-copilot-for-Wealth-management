# RM Copilot — Capstone Presentation

## 1. Problem Statement

Relationship managers need fast, compliant decision support when preparing for client meetings: portfolio context, product/policy research, suitability checks, and grounded talking points — without crossing into unlicensed personalized advice.

## 2. Architecture

```
RM Query (Streamlit)
       │
       ▼
┌──────────────────────────────────────┐
│         LangGraph Agent              │
│  plan → react_tools → gather_*       │
│  → check_suitability → synthesize    │
│  → review_gate (HITL interrupt)      │
└──────────────────────────────────────┘
       │                    │
       ▼                    ▼
  Tools (portfolio,     Hybrid RAG
  market, suitability)  (BM25 + ChromaDB)
       │                    │
       ▼                    ▼
  Client JSON           11 source docs
                        (PG/CMP/RN)
```

## 3. Knowledge Base

| Category | Documents | Count |
|----------|-----------|-------|
| Product guides | PG-001 – PG-004 | 4 |
| Compliance | CMP-001 – CMP-004 | 4 |
| Research | RN-001 – RN-003 | 3 |
| **Total** | | **11** |

Metadata: `doc_id`, `type`, `date`, `source`, `sensitivity` (public / internal / restricted).

## 4. Agent Pipeline

1. **Plan** — decompose RM request; detect licensed-advice escalation triggers
2. **ReAct tools** — bounded tool call (portfolio, market, suitability)
3. **Gather portfolio** — load `C-204` / `C-301` holdings + risk profile
4. **Gather research** — hybrid RAG with entitlement filtering
5. **Gather market** — symbol-level context
6. **Check suitability** — PG/CMP policy validation
7. **Synthesize** — Pydantic `ClientBrief` with citations
8. **Review gate** — compliance status + human approve/reject interrupt

## 5. Safety & Guardrails

- Citations required on every recommendation
- Suitability gate blocks mismatched risk profiles (e.g. PG-003 / Conservative)
- Licensed advice patterns → `Needs Review` + HITL
- Restricted documents (CMP-002) hidden from unentitled RMs
- Disclaimer on every brief: decision support only

## 6. Evaluation Results

Run: `python -m src.evaluation.metrics` or `notebooks/evaluation.ipynb`

| Metric | Target | Result |
|--------|--------|--------|
| Retrieval hit rate (golden set) | High | See notebook |
| Citation coverage | 100% | ✅ |
| Suitability gate precision | 100% | ✅ |
| Escalation detection | 100% | ✅ |
| Entitlement filter | No leaks | ✅ |
| Multi-hop vs single-shot | Comparison | See notebook |
| RAGAS faithfulness | Grounded answers | See notebook |

## 7. Demo Walkthrough

### Setup
```powershell
pip install -r requirements.txt
copy .env.example .env   # set OPENAI_API_KEY
python scripts/ingest.py
streamlit run app/streamlit_app.py
```

### Scenario 1 — Quarterly Review (C-204)
- Query: *Prepare talking points for client C-204's quarterly review*
- Output: `ClientBrief` with PG-002, RN-001, CMP-003 citations

### Scenario 2 — Suitability
- Query: *Is PG-003 suitable for a conservative client?*
- Output: Not suitable + product/policy citations

### Scenario 3 — Portfolio Risk
- Query: *Summarize the portfolio risk for client C-204*
- Output: Conservative allocation summary

### Scenario 4 — Licensed Advice Escalation
- Query: *Recommend a personalized 4% withdrawal rate for client C-204*
- Output: HITL interrupt — Approve or Escalate

### Scenario 5 — Entitlement Control
- RM-001 (no restricted): CMP-002 hidden
- RM-002 (restricted): CMP-002 visible in RAG tab

## 8. Screenshots (capture during demo)

1. Streamlit dashboard — Generate Brief tab with ClientBrief output
2. Compliance status badge (Cleared / Needs Review)
3. HITL Approve/Reject buttons on escalation query
4. Suitability tab — PG-003 blocked for Conservative
5. RAG tab — entitlement-filtered CMP-002 results
6. Evaluation notebook — summary metrics table + bar chart

## 9. Tech Stack

- **LLM:** OpenAI GPT-4o
- **Embeddings:** text-embedding-3-small
- **Vector DB:** ChromaDB
- **Orchestration:** LangGraph
- **UI:** Streamlit
- **Observability:** LangSmith (optional)
- **Eval:** RAGAS faithfulness + golden set

## 10. Disclaimer

This copilot provides decision support for relationship managers. It does not replace licensed investment advice.
