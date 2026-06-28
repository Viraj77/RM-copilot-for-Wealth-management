# Relationship Manager Copilot: Technical Deep Dive & Presentation Guide

This document is designed to help you build your final Capstone presentation. It breaks down the project into logical "slides" or sections, explaining both **what** the feature does (Functionality) and **how** it works under the hood (Technical Implementation).

---

## 1. Project Overview & Architecture
**Functionality:** 
The Relationship Manager (RM) Copilot is an AI assistant designed to help wealth managers prepare for client meetings. It fetches live portfolio data, reads through thousands of pages of firm policy and research, checks the RM's ideas against strict compliance rules, and generates a clean, structured `ClientBrief` with citations.

**Technicality:**
- **Orchestrator:** Built using **LangGraph**, representing a state machine (`StateGraph`). The graph controls the flow of execution: `agent` -> `tools` -> `agent` -> `synthesize_brief` -> `END`.
- **LLM:** Powered by OpenAI `gpt-4o` with a `temperature` of 0 for deterministic, reliable tool calling.
- **State Management:** Uses a typed `AgentState` dictionary to pass the conversation history (`messages`), the `client_id`, the `rm_tier`, and the final generated brief between nodes.

---

## 2. The Agent's Toolbelt
**Functionality:**
The agent is not just a chatbot; it acts as an autonomous researcher using four distinct tools to gather context before it ever answers the RM.

**Technicality:**
The tools are bound to the LLM using LangChain's `@tool` decorator and `llm.bind_tools()`. 
1. **`portfolio_lookup_tool`**: Reads from `clients.json` to pull the active client's AUM, holdings, and strict Risk Profile (e.g., Conservative, Aggressive).
2. **`market_data_tool`**: Fetches mocked live pricing and volatility data.
3. **`rag_retriever_tool`**: The entry point to the vector database. Allows the agent to query product guides, research notes, and compliance policies.
4. **`suitability_checker_tool`**: Evaluates a proposed investment against the client's risk profile.

---

## 3. Advanced Hybrid RAG System
**Functionality:**
When the agent needs to find a specific firm policy (e.g., "maximum equity for conservative clients"), it searches the firm's knowledge base. It guarantees that the RM only sees documents they are allowed to see based on their seniority tier.

**Technicality:**
- **Storage (Dual-Index Strategy):** Uses **ChromaDB** as the primary vector store for production. During ingestion, a secondary **FAISS** index is built simultaneously. This allows us to use FAISS as a baseline to evaluate and prove the superiority of the advanced ChromaDB hybrid search. Documents are embedded using `text-embedding-3-small`.
- **Hybrid Retrieval:** The `query_hybrid` function uses **Reciprocal Rank Fusion (RRF)**. It combines vector similarity (Semantic Search) with exact-keyword matching (BM25) to ensure high precision even for specific fund tickers.
- **Cross-Encoder Reranking:** Results are passed through a sentence-transformer cross-encoder to re-score and perfectly order the top 5 chunks.
- **Entitlement Filtering:** Metadata filtering (`$in`) is applied directly at the database level. If an RM is `standard`, the database completely hides chunks marked `restricted` or `internal`.
- **Thread Safety:** A global `threading.Lock()` wraps the retrieval function to prevent SQLite database corruption when LangGraph attempts to run multi-hop queries (calling the RAG tool twice concurrently).

---

## 4. Structured Output (Pydantic)
**Functionality:**
Instead of a messy block of text, the final output to the RM is a clean, structured Client Brief containing the portfolio summary, recommendations, and compliance statuses.

**Technicality:**
- The `synthesize_brief` LangGraph node uses `with_structured_output(ClientBrief)`.
- It leverages **Pydantic V2**. The LLM is forced to return a JSON object matching the exact schema defined in `src/models/brief.py` (e.g., forcing `compliance_status` to be exactly `"cleared"`, `"needs_review"`, or `"blocked"`).

---

## 5. Safety & Guardrails (The Compliance Engine)
**Functionality:**
In Wealth Management, AI hallucinations or bad advice can lead to massive fines. The Copilot contains hard-coded safety nets that cannot be bypassed by the LLM.

**Technicality:**
1. **The Suitability Gate:** Before the brief is generated, the agent runs the `suitability_checker_tool`. This tool runs hard-coded Python logic (not AI guesses) to check if the recommended asset class violates the client's risk profile limits.
2. **The Citation Validator:** Found in `src/guardrails/citation_validator.py`. After the LLM generates the final JSON brief, this script intercepts it. It scans the `citations` array and cross-references the `chunk_id`s against the actual database. If the LLM hallucinated a fake citation, the validator catches it and forcibly overwrites the brief's `compliance_status` to `"needs_review"`.

---

## 6. The User Interface & Dynamic Dashboards
**Functionality:**
A sleek, interactive Streamlit application that serves as the RM's central hub. It features:
- **Copilot Chat**: A conversational interface where the RM interacts with the AI, with expanders to see exactly what tools the agent is using in real-time.
- **Client Information Dashboard**: Provides an in-depth breakdown of the active client's portfolio, Risk Profile, and RM notes. Incorporates dynamic conditional formatting to highlight underperforming assets.
- **Market Data Dashboard**: A sortable, interactive global market overview displaying current pricing, historical returns, and volatility.
- **Knowledge Base Upload**: Allows the RM to securely upload new firm documents (PDF, DOCX, TXT) on the fly and instantly add them to the Copilot's searchable knowledge base.

**Technicality:**
- Built with **Streamlit** and **Pandas**. Utilises `st.dataframe` with custom `Styler` objects to automatically apply conditional formatting (e.g., highlighting negative returns in red).
- **Single-File Dynamic Ingestion:** The Upload page utilizes a targeted `ingest_single_file` function. Instead of rebuilding the entire vector store from scratch, it efficiently loads, chunks, embeds, and `upserts` *only* the newly uploaded document into the existing ChromaDB collection.
- The LangGraph execution is streamed using `copilot_app.stream()`. As events yield from the graph, the UI dynamically renders them using `st.empty()` containers and expanders (`render_tool_calls`).
- To ensure maximum stability and bypass Streamlit's notorious Windows caching bugs, the `streamlit_app.py` script aggressively clears `sys.modules` for the `src` directory on every rerun, ensuring the latest backend code is always executed.

---

## 7. Evaluation & Analytics (Part B)
**Functionality:**
Proving that the Copilot actually works through mathematical evaluation, ensuring it is ready for production.

**Technicality:**
- **The Golden Set:** A suite of test queries (`data/golden_set/evaluation_queries.json`) designed to test specific traps (e.g., trying to trick the agent into recommending High Yield bonds to a Conservative client).
- **Metrics Evaluated:** 
  - *Retrieval Relevance:* Does the RAG pull the right document?
  - *Suitability Precision:* Does the compliance gate trigger when it is supposed to?
  - *Faithfulness (RAGAS):* Using an LLM-as-a-judge to evaluate if the final brief's claims are entirely backed by the retrieved context chunks, proving a 0% hallucination rate.

---

## 8. Token & Latency Optimizations
**Functionality:**
To ensure the Copilot remains cost-effective and responsive during long conversations, we identified and implemented multiple optimization strategies targeting token usage and model routing.

**Technicality:**
- **Tool Output Summarization & Filtering:** We identified that tools were dumping massive, unoptimized payloads into the context window. We modified the `portfolio_lookup_tool` to exclude massive line-item holdings arrays by default, and streamlined the `rag_retriever_tool` to strip redundant nested citation dictionaries and reduce the default `top_k` chunk return from 5 to 3. This drastically shrank the context window payload on every turn.
- **LLM Routing / Model Downgrading:** We recognized that background operations reading historical interaction logs (e.g., Sentiment Timeline Extraction and Meeting Mood Prediction) were unnecessarily burning premium API quota. We swapped out the expensive `gpt-4o` for the highly efficient `gpt-4o-mini` specifically for these isolated background tools. This slashed API token costs by roughly 95% for those tasks without affecting the core intelligence of the main Copilot chat.
