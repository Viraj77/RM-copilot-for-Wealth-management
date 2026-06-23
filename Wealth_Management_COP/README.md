# 🏦 Relationship Manager Copilot for Wealth Management

> **Capstone Project 6** | Python · LangGraph · OpenAI · ChromaDB · Streamlit

A multi-tool agentic assistant that helps wealth-management Relationship Managers (RMs) prepare for client meetings by automatically generating grounded, compliant `ClientBrief` documents with citations, suitability checks, and talking points.

---

## 🗂️ Project Structure

```
WealthManagerCopilot/
├── config/             # Centralized settings (pydantic-settings)
├── data/
│   ├── raw/            # Source documents (product/policy/research)
│   ├── clients/        # Synthetic client portfolio database
│   └── golden_set/     # Evaluation queries
├── src/
│   ├── models/         # Pydantic data models
│   ├── ingestion/      # Document loading, chunking, embedding, indexing
│   ├── tools/          # Agent tools (portfolio, market data, RAG, suitability)
│   ├── agent/          # LangGraph StateGraph (nodes, graph, prompts, state)
│   ├── guardrails/     # Compliance gate, entitlement filter, disclaimers
│   └── utils/          # Logging, LangSmith tracing
├── app/                # Streamlit RM dashboard
├── notebooks/          # Ingestion demo, agent playground, evaluation
└── tests/              # Unit + integration tests
```

---

## ⚙️ Setup

### 1. Clone & Install

```bash
git clone <repo-url>
cd WealthManagerCopilot
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Ingest Knowledge Documents

```bash
python -m src.ingestion.indexer
```

### 4. Run the Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## 🤖 Agent Pipeline

```
Plan → Gather Portfolio → Gather Research (RAG) → Check Suitability → Synthesize Brief → Review Gate
```

- **LLM**: GPT-4o via OpenAI
- **Embeddings**: `text-embedding-3-small`
- **Vector Store**: ChromaDB (primary) + FAISS (evaluation)
- **Orchestration**: LangGraph `StateGraph` with conditional edges + human-in-the-loop
- **Output**: Validated `ClientBrief` (Pydantic v2) with citations

---

## 🛡️ Safety & Guardrails

| Guardrail | Description |
|---|---|
| Grounding + Citations | Every recommendation must cite ≥1 retrieved source |
| Compliance Gate | Suitability checker validates all recommendations |
| No Licensed Advice | Auto-escalation if personalized investment advice detected |
| Entitlement Filtering | Sensitivity-based access control on knowledge chunks |
| Bounded Agent Steps | max 10 steps per node, 30 total |
| Audit Logging | Full LangSmith trace for every run |

---

## 📊 Demo Scenarios

| # | Query | Expected Outcome |
|---|---|---|
| 1 | "Prepare talking points for client C-204's quarterly review" | Grounded `ClientBrief` with citations |
| 2 | "Is fund XYZ suitable for a conservative client?" | Suitability check + policy citation |
| 3 | "Summarize the portfolio risk for client C-204" | Portfolio risk summary |
| 4 | Recommendation requiring licensed advice | Escalation triggered |
| 5 | Restricted research for unentitled RM | Filtered out (access control) |

---

## 🚀 Enhancement: Client Sentiment Time-Machine

Analyzes the emotional trajectory of client interactions over time, predicts meeting mood, and generates conversation strategy suggestions. See `src/tools/sentiment_analyzer_tool.py`.

---

## 📋 Requirements

See `requirements.txt` for full dependency list. Key packages:
- `langgraph`, `langchain`, `langchain-openai`
- `openai`, `chromadb`, `faiss-cpu`
- `pydantic>=2.0`, `pydantic-settings`
- `streamlit`, `ragas`, `langsmith`

---

> ⚠️ **Disclaimer**: This system is decision-support tooling for Relationship Managers and does not constitute personalized investment advice.
