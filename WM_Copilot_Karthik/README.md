# Horizon Wealth Management — Relationship Manager (RM) Copilot

> **Capstone P6 — LangGraph-based agentic copilot for Wealth Management Relationship Managers**

A production-style AI copilot that helps Relationship Managers (RMs) prepare compliant, grounded client briefs, evaluate investment suitability, and retrieve entitlement-filtered research — all within a human-in-the-loop (HITL) compliance workflow.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Component Descriptions](#3-component-descriptions)
   - [agent.py — LangGraph State Graph](#agentpy--langgraph-state-graph)
   - [tools.py — Tool Functions](#toolspy--tool-functions)
   - [ingest.py — Document Ingestion Pipeline](#ingestpy--document-ingestion-pipeline)
   - [client_db.py — Client Database](#client_dbpy--client-database)
   - [app.py — Streamlit Application](#apppy--streamlit-application)
   - [analytics.py — Evaluation & Benchmarking](#analyticspy--evaluation--benchmarking)
   - [eval_dataset.json — Evaluation Dataset](#eval_datasetjson--evaluation-dataset)
4. [Knowledge Base (docs/)](#4-knowledge-base-docs)
5. [Prerequisites & Setup](#5-prerequisites--setup)
6. [Running the Application](#6-running-the-application)
7. [Running the Ingestion Pipeline](#7-running-the-ingestion-pipeline)
8. [Running Tests](#8-running-tests)
9. [What the Application Produces](#9-what-the-application-produces)
10. [Evaluation Dataset Structure](#10-evaluation-dataset-structure)
11. [Key Design Decisions](#11-key-design-decisions)

---

## 1. System Overview

The **RM Copilot** is built on a **deterministic LangGraph state graph** — not a ReAct agent. Instead of giving the LLM free rein to call tools in an open loop, the workflow follows a compliance-safe pipeline with adaptive routing based on query intent classification.

When a query is received, the agent first classifies it to detect:
1. **Out-of-Context Queries**: If the query is unrelated to wealth management or client advisory (e.g. general trivia, greetings, sports), the graph routes **directly** to the output node to return a standardized message, bypassing the portfolio database, RAG retriever, and compliance checking nodes entirely.
2. **In-Context Queries**: Classified as either **Structured** (requires a formatted `ClientBrief`) or **Free-form** (general analytical, research, fund-to-fund, or portfolio-to-portfolio comparison). These flow through the full compliance and data gathering pipelines:

```
                                 classify_intent
                                        │
                ┌───────────────────────┴───────────────────────┐
         (In-Context)                                    (Out-of-Context)
                │                                               │
                ▼                                               │
         gather_portfolio                                       │
                │                                               │
         gather_research (RAG)                                  │
                │                                               │
        check_suitability (Compliance)                          │
                │                                               │
         ┌──────┴───────────────────────┐                       │
(Structured)                       (Free-form)                  │
         │                              │                       │
         ▼                              ▼                       ▼
    synthesize                  free_form_answer ◄──────────────┘
  (ClientBrief)                (Prose or Bypass)
         │                              │
         ▼                              ▼
        END                            END
```

This design ensures that **valid wealth management queries are always checked by compliance** while irrelevant general trivia queries are safely rejected early, without wasteful database lookups or RAG retrieval runs.

---

## 2. Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                       Streamlit UI (app.py)                      │
│   Client Selector │ Query Input │ HITL Override Panel            │
└──────────────┬───────────────────────────────────────────────────┘
               │ invoke(initial_state, config)
               ▼
┌──────────────────────────────────────────────────────────────────┐
│                  LangGraph State Graph (agent.py)                │
│                                                                  │
│       │          │                      │                        │
│       └──────────┴─────────────────────►END                      │
└──────────────────────────────────────────────────────────────────┘
               │
        Tools called by nodes
               ▼
┌──────────────────────────────────────────────────────────────────┐
│                        tools.py                                  │
│  portfolio_lookup │ market_data_tool │ suitability_checker       │
│  rag_retriever (ChromaDB + metadata entitlement filter)          │
└──────────────┬───────────────────────────────────────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
  ChromaDB          client_db.py
  (chroma_db/)     (Mock Client DB)
       ▲
       │ ingested by
  ingest.py ◄── docs/ (10 documents)
```

---

## 3. Component Descriptions

### `agent.py` — LangGraph State Graph

**Purpose:** The core of the system. Defines the `AgentState` (TypedDict), all graph nodes, conditional routing logic, and compiles the LangGraph `StateGraph` with a `MemorySaver` checkpointer for HITL interrupts.

**Key Classes:**
- `AgentState` — TypedDict defining all state fields (query, client_id, product_code, response_mode, is_out_of_context, free_form_response, client_profile, retrieved_evidence, suitability_report, compliance_status, final_brief, etc.)
- `Reco` — Pydantic model for a single investment recommendation (idea, rationale, suitability, citations)
- `ClientBrief` — Pydantic model for the full structured output (portfolio summary, recommendations, compliance status, talking points, disclaimer)

**Graph Nodes:**

| Node | Function | What it Does |
|---|---|---|
| `classify_intent` | `classify_intent_node()` | Identifies response mode (structured vs freeform), extracts entities (client_id, product_code, allocation_amount), detects out-of-context queries, and outlines plans. |
| `gather_portfolio` | `gather_portfolio_node()` | Fetches the client profile from `client_db.py`. |
| `gather_research` | `gather_research_node()` | Calls `rag_retriever()` with the query + client risk profile. Also calls `market_data_tool()` if a product code is present. |
| `check_suitability` | `check_suitability_node()` | Runs programmatic `suitability_checker()` + an LLM compliance review (licensing checks, qualitative violations). Merges both into a final `compliance_status`. |
| `synthesize` | `synthesize_node()` | Generates the full structured `ClientBrief` via structured LLM output, grounded in retrieved evidence (Structured response mode). |
| `free_form_answer` | `free_form_answer_node()` | Generates a rich, cited analytical markdown answer (Free-form mode) OR returns a standardized bypass warning (if out of context). |
| `human_review` | `human_review_node()` | LangGraph pauses **before** this node. If the RM approves, the override clears the status and appends RM notes. |

**Routing Logic:**
- `route_classify_intent()` — After `classify_intent`: if `is_out_of_context` is `True`, routes directly to `free_form_answer` (bypassing portfolio lookup, RAG, and compliance checking nodes). Otherwise, routes to `gather_portfolio`.
- `route_compliance_or_freeform()` — After `check_suitability`: if not Cleared → `human_review`; if Cleared and mode is freeform → `free_form_answer`; if Cleared and mode is structured → `synthesize`.
- `interrupt_before=["human_review"]` — The graph pauses automatically before the human review node.

---

### `tools.py` — Tool Functions

**Purpose:** All deterministic tool logic used by the graph nodes. No LLM calls here — pure Python functions.

| Function | Description |
|---|---|
| `portfolio_lookup(client_id)` | Wraps `get_client_profile()`. Returns full client record or error dict. |
| `market_data_tool(product_code)` | Resolves a product code (including ticker aliases like `HBGF` → `PG-001`) to a structured product details dict. Checks hard-coded mutual funds first, then falls back to PG-004 CSV lookup. |
| `suitability_checker(client_id, product_code, allocation_amount)` | Programmatic compliance engine. Checks: (1) risk profile match against `suitable_risk_profiles`, (2) RULE-016 (SCN restriction for Conservative/Balanced), (3) RULE-003 (15% structured note concentration cap). Returns `{status, violations, reasons, citations}`. |
| `rag_retriever(query, rm_research_tier)` | Queries the ChromaDB vector store with a sensitivity metadata filter. Tier-1 → `{"sensitivity": "Public"}`. Tier-2 → no filter (Public + Restricted). Returns top-5 relevant document chunks with metadata. |

---

### `ingest.py` — Document Ingestion Pipeline

**Purpose:** One-time pipeline to read all 10 source documents from `docs/`, extract metadata, chunk them, embed them, and store them in ChromaDB.

**Supported formats:** `.txt`, `.docx`, `.pdf`, `.csv`

**Pipeline Steps:**
1. **Walk** `docs/` directory recursively
2. **Load** each file using format-specific loaders (`load_txt`, `load_docx`, `load_pdf`, `load_csv_rows`)
3. **Extract metadata** dynamically from file content and path — extracts `doc_id`, `type`, `date`, `source`, `sensitivity`
4. **Chunk** all documents using `RecursiveCharacterTextSplitter` (chunk_size=500 tokens, overlap=50 tokens, tiktoken `cl100k_base` encoding)
5. **Embed** using OpenAI `text-embedding-3-small`
6. **Persist** to ChromaDB at `chroma_db/` with collection `wealth_mgmt_knowledge`

> ⚠️ **Must be run once before starting the app.** The `chroma_db/` directory must exist and be populated for retrieval to work.

---

### `client_db.py` — Client Database

**Purpose:** In-memory mock client database. Returns client profiles including holdings, risk profile, RM assignment, and RM research entitlement tier.

**Clients:**

| Client ID | Name | Risk Profile | RM | RM Tier |
|---|---|---|---|---|
| C-101 | Arthur Pendleton | Conservative | Jane Smith | 1 (Public only) |
| C-204 | Eleanor Vance | Balanced | John Doe | 2 (Public + Restricted) |
| C-302 | Marcus Vance | Aggressive | John Doe | 2 (Public + Restricted) |

**Key Functions:**
- `get_client_profile(client_id)` — Returns client dict or `None`
- `get_all_clients()` — Returns list of all client dicts (used in app sidebar)

---

### `app.py` — Streamlit Application

**Purpose:** The main UI. Provides two tabs:

**Tab 1 — RM Workspace:**
- Sidebar: Client selector, RM info display, trade simulation parameters (product code, allocation amount)
- Main area: Client info, current holdings, query input with templates
- Execution: Classifies query intent, runs the shared pipeline, branches output rendering dynamically based on response mode:
  - **Structured Brief**: Calls `render_client_brief()` to print Pydantic ClientBrief details.
  - **Analytical (Free-form)**: Calls `render_freeform_response()` to print cited markdown prose and comparison tables.
- HITL Panel: If the agent pauses at `human_review` (for compliance review/block), shows an override form where the RM can justify and approve or terminate.

**Tab 2 — Analytics & Benchmarks:**
Calls functions from `analytics.py` to display evaluation metrics:
1. Retrieval Relevance & Citation Coverage
2. Suitability Gate Precision & Recall
3. Faithfulness & Groundedness (LLM-as-Judge)
4. Single-Shot vs Multi-Hop Retrieval Analysis

**Session State variables:**
- `graph_state` — The current LangGraph state dict after invocation
- `thread_id` — Unique thread ID for checkpointing (format: `thread_{client_id}_{timestamp}`)
- `agent_run` — Boolean: whether the agent has been run in this session
- `interrupt_state` — Boolean: whether the graph is currently paused at `human_review`

---

### `analytics.py` — Evaluation & Benchmarking

**Purpose:** Implements 4 evaluation dimensions against a hard-coded `GOLDEN_SET`. Renders results in the Streamlit analytics tab.

| Function | What it Evaluates |
|---|---|
| `evaluate_retrieval()` | Tests the RAG retriever against 5 golden queries. Measures document coverage (% of expected doc_ids retrieved). Reports overall retrieval accuracy. |
| `evaluate_suitability_gate()` | Runs 6 suitability test cases through `suitability_checker()`. Computes Precision, Recall, and F1 score for the compliance gate. |
| `evaluate_faithfulness()` | Generates a sample recommendation and runs an LLM-as-judge to score Groundedness (0–1) and Citation Accuracy (0–1). |
| `compare_single_vs_multihop()` | Compares single-shot vector retrieval vs a 2-step multi-hop approach (retrieve → LLM identifies cross-references → second retrieval). |

---

### `eval_dataset.json` — Evaluation Dataset

**Purpose:** A comprehensive, structured evaluation dataset with **44 test cases** across 8 categories, derived from the full knowledge base. Used for systematic validation outside of the Streamlit app.

| Section | Cases | Tests |
|---|---|---|
| `retrieval` | 10 | RAG document retrieval accuracy per query & RM tier |
| `suitability_gate` | 10 | Programmatic suitability_checker — Cleared/Needs Review/Blocked + violation codes |
| `compliance_block` | 5 | LLM compliance layer — licensing/discretionary restrictions |
| `entitlement_filtering` | 5 | Metadata sensitivity filter — Tier-1 vs Tier-2 access |
| `faithfulness` | 4 | LLM-as-judge groundedness and citation accuracy |
| `multihop_retrieval` | 4 | Cross-document multi-hop retrieval chains |
| `edge_cases` | 8 | Boundary values, unknown clients, ticker aliases, zero allocations |
| `hitl_escalation` | 5 | Human-in-the-loop interrupt and resume flow |

---

## 4. Knowledge Base (`docs/`)

All 10 source documents are synthetic and created for this capstone. They are cross-referenced to support multi-hop retrieval testing.

| Doc ID | Category | Title | Format | Sensitivity |
|---|---|---|---|---|
| PG-001 | Product Guide | Horizon Balanced Growth Fund (HBGF) | DOCX | Public |
| PG-002 | Product Guide | Horizon Conservative Income Fund (HCIF) | TXT | Public |
| PG-003 | Product Guide | Horizon Aggressive Equity Fund (HAEF) | PDF | Public |
| PG-004 | Product Guide | Fixed Income & Structured Products (160 rows) | CSV | Public |
| CMP-001 | Compliance | Suitability & Compliance Policy | DOCX | Public |
| CMP-002 | Compliance | Restricted List & Entitlement Rules (83 rules) | CSV | Public |
| CMP-003 | Compliance | Client Risk Profiling Methodology | PDF | Public |
| RN-001 | Research | Q2 2026 Global Market Outlook | PDF | Public |
| RN-002 | Research | Sector Deep Dive — Technology & Applied AI | TXT | **Restricted** |
| RN-003 | Research | Fixed Income Outlook — Mid-Year 2026 | DOCX | Public |

> 📌 **RN-002** is tagged `sensitivity=Restricted`. It is only retrievable by Tier-2 RMs (assigned to clients C-204 and C-302). Tier-1 RMs (assigned to C-101) are filtered out by the entitlement gate in `rag_retriever()`.

---

## 5. Prerequisites & Setup

### Requirements

- Python 3.10+
- An OpenAI API key (for GPT-4o-mini inference and text-embedding-3-small embeddings)

### Installation

```bash
# Clone or navigate to the project directory
cd my_capstone_workspace

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...your-key-here...
```

---

## 6. Running the Application

### Step 1 — Ingest documents (one-time setup)

```bash
python ingest.py
```

This populates the `chroma_db/` vector store. Expected output:

```
Scanning and loading docs/compliance/CMP-001...
...
Total raw document objects loaded: 167
Split raw documents into 412 token-bounded chunks.
Successfully persisted ChromaDB to chroma_db
```

### Step 2 — Launch the Streamlit app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### Using the RM Workspace (Tab 1)

1. **Select a client** from the sidebar dropdown (C-101, C-204, or C-302)
2. *(Optional)* Toggle **"Simulate New Trade Recommendation"** and enter a product code and allocation amount
3. **Choose a query template** or write a custom request
4. Click **"Execute Copilot Agent"**
5. The agent runs the full pipeline. If a compliance violation is detected, the **HITL Override Panel** appears
6. RM can approve (with justification notes) or reject. Final brief is displayed after approval.

### Using the Analytics Tab (Tab 2)

Click the **"📊 Analytics & Benchmarks (Part B)"** tab. The page automatically runs all 4 evaluation functions against live ChromaDB and displays results in tables and metrics.

---

## 7. Running the Ingestion Pipeline

```bash
python ingest.py
```

**What it does:**
- Scans `docs/compliance/`, `docs/product_guides/`, `docs/research/`
- Loads `.txt`, `.docx`, `.pdf` files as single text blocks
- Loads `.csv` files as individual row documents (each CSV row becomes a separate LangChain `Document`)
- Extracts metadata from content and filename (doc_id, type, date, source, sensitivity)
- Chunks with 500-token chunks / 50-token overlap (tiktoken `cl100k_base`)
- Embeds with OpenAI `text-embedding-3-small`
- Persists to `chroma_db/` (ChromaDB)

> ⚠️ Re-running `ingest.py` will add duplicate chunks to ChromaDB. If you need a fresh ingest, delete the `chroma_db/` folder first.

---

## 8. Running Tests

```bash
# From the project root
pytest tests/test_copilot.py -v
```

**Test coverage in `tests/test_copilot.py`:**

| Test | What it Validates |
|---|---|
| `test_client_db` | `get_client_profile("C-204")` returns Eleanor Vance with Balanced risk profile |
| `test_portfolio_lookup` | `portfolio_lookup("C-204")` returns valid dict without error |
| `test_market_data_tool` | `market_data_tool("PG-001")` resolves HBGF product correctly |
| `test_suitability_checker` | Cleared for C-204/PG-001; Blocked with RULE-016 for C-101/SCN-US-24 |
| `test_rag_retriever` | Returns ≥1 document for "Balanced Growth Fund"; PG-001 or compliance docs in results |
| `test_agent_graph` | Full graph invocation for C-204/HBGF talking points; verifies final_brief is not None and compliance_status=Cleared |
| `test_classify_intent_structured` | Verifies that client talking points and suitability queries default to structured response mode |
| `test_classify_intent_freeform` | Verifies that comparison and informational queries default to freeform response mode |
| `test_freeform_answer_node` | Verifies that freeform node returns clean markdown content with citations |
| `test_out_of_context_query` | Verifies that out-of-context queries route directly to the answer node and return the standardized bypass message. |

> ⚠️ `test_rag_retriever` and `test_agent_graph` require a populated ChromaDB and a valid `OPENAI_API_KEY`. Run `python ingest.py` first.

---

## 9. What the Application Produces

### Agent Outputs

Depending on the response mode, the agent produces one of two output formats:

#### 1. Structured Brief Output (`ClientBrief`)
Used for client-specific suitability assessments. Populates a structured `ClientBrief` object with:
- `client_id` (str): Client ID (e.g. `"C-204"`).
- `risk_profile` (str): Risk tolerance level.
- `portfolio_summary` (str): Current holdings description.
- `recommendations` (List[Reco]): Grounded investment ideas with rationale, suitability, and citations.
- `compliance_status` (str): `"Cleared"`, `"Needs Review"`, or `"Blocked"`.
- `talking_points` (List[str]): Compliance-grounded discussion starters.
- `disclaimer` (str): Standard decision-support notice.

#### 2. Analytical Response Output (`free_form_response`)
Used for comparisons, research, and general QA. Populates `free_form_response` with a rich markdown text body containing:
- Structured side-by-side comparison tables.
- Entitlement-filtered market insights.
- Bulleted methodology explanations.
- Citations inline (e.g., `[PG-001]`, `[RN-002]`).

### Intermediate State Fields (visible in UI under "View Execution Steps")

- **Plan** — Planning analysis showing response mode classification ("STRUCTURED" or "FREEFORM") and extraction details.
- **Compliance Gate Status** — Programmatic + LLM suitability results.
- **Retrieved Knowledge (RAG)** — Entitlement-filtered document passages with doc_id, type, sensitivity, and content.

### HITL Override

When `compliance_status` is `"Blocked"` or `"Needs Review"`, the app pauses and shows the override panel. After RM approval:
- `compliance_status` → `"Cleared"`
- `escalated` → `True`
- `talking_points` → appended with `"RM Note: Approved with review notes: {notes}"` (or `final_brief` generated with override metadata).

---

## 10. Evaluation Dataset Structure

The `eval_dataset.json` file contains **44 curated test cases** organized in 8 sections:

```
eval_dataset.json
├── _meta              — Dataset metadata, client/product reference tables
├── retrieval          — 10 RAG retrieval test cases (query → expected doc_ids)
├── suitability_gate   — 10 programmatic compliance test cases (client + product → expected status)
├── compliance_block   — 5 LLM compliance layer test cases (licensing violations)
├── entitlement_filtering — 5 Tier-1 vs Tier-2 access control tests
├── faithfulness       — 4 LLM-as-judge groundedness test cases
├── multihop_retrieval — 4 multi-hop cross-document retrieval chains
├── edge_cases         — 8 boundary/robustness tests
└── hitl_escalation    — 5 HITL interrupt and resume flow tests
```

Each test case includes:
- `id` — Unique identifier (e.g., `SUIT-004`, `RET-007`)
- `category` — Human-readable category label
- `rationale` — Why this test case exists and what it proves
- `reference_rule` or `reference_docs` — The specific policy rule or document being tested
- `expected_*` fields — The expected system behavior or output

---

## 11. Key Design Decisions

| Decision | Rationale |
|---|---|
| **Deterministic graph, not ReAct** | Compliance gates cannot be bypassed by an LLM deciding to skip a step. The graph enforces order. |
| **Structured output (Pydantic)** | Entity extraction, intent classification, and structured briefs use `.with_structured_output()` to guarantee type-safe responses. The free-form node uses unstructured generation for rich, markdown-formatted comparisons. |
| **Metadata-based entitlement filtering** | ChromaDB's metadata filter on `sensitivity` enforces document access control at retrieval time, not at prompt time. |
| **Merging programmatic + LLM compliance** | The programmatic checker catches known rules (RULE-016, RULE-003) with 100% reliability. The LLM layer catches qualitative issues (tax advice, discretionary management) that rules can't express. The most restrictive status wins. |
| **MemorySaver + interrupt_before** | LangGraph's built-in checkpointing enables stateful HITL pauses. The graph state is fully persisted and resumable across Streamlit re-renders. |
| **CSV row-level ingestion** | PG-004 (160 products) and CMP-002 (83 rules) are ingested row-by-row, making individual product and rule retrieval far more precise than ingesting the CSV as a monolithic blob. |

---

*All client names, fund names, AUM figures, performance numbers, and policy rules are fictional and created for this capstone exercise.*
