# Comprehensive Technical Presentation Guide
**Project:** Relationship Manager Copilot for Wealth Management

This guide provides a deep, granular look into every technical aspect of the RM Copilot Capstone project. Use this document to build your presentation slides, talk tracks, and technical defense.

---

## 1. High-Level Architecture & Orchestration
**Overview:**
The core brain of the Copilot is built using **LangGraph**, which models the agent as a cyclic State Machine (`StateGraph`). 

**Technical Deep-Dive:**
- **State Management:** The graph passes a strictly typed `AgentState` dictionary between nodes. This state holds the `messages` array (conversation history), `client_id`, `rm_tier` (for entitlement filtering), and the final `ClientBrief`.
- **Node Flow:** The agent loops between a `reasoning_node` (where the LLM decides what to do) and a `tool_node` (where Python functions execute). Once reasoning is complete, the graph transitions to the `synthesize_brief` node to generate the final structured output.
- **LLM Engine:** We use OpenAI's `gpt-4o` for the core reasoning agent.
- **Temperature Control:** The core agent's temperature is set to **0.0** to ensure highly deterministic, reproducible tool calling and logical reasoning.

---

## 2. Ingestion Pipeline & Chunking Strategy
**Overview:**
The system transforms raw unstructured documents (PDFs, DOCX, TXT, MD) into a searchable semantic database.

**Technical Deep-Dive:**
- **Document Loading:** The pipeline processes three categories of documents: Product Guides, Compliance Policies, and Research Notes.
- **Structure-Aware Chunking:** Using `RecursiveCharacterTextSplitter`, documents are broken down into ~1000-character chunks with 200-character overlaps. This overlap prevents critical context (like the end of a sentence) from being severed.
- **Metadata Tagging:** During chunking, every slice of text is permanently tagged with `doc_id`, `doc_type`, and `sensitivity` (e.g., `public`, `internal`, `restricted`).
- **Embedding:** Chunks are vectorized using OpenAI's `text-embedding-3-small` model, which generates dense 1536-dimensional vectors optimized for financial terminology.

---

## 3. Hybrid RAG & Knowledge Retrieval
**Overview:**
The Copilot uses an advanced Retrieval-Augmented Generation (RAG) system to ensure it never hallucinates firm policies.

**Technical Deep-Dive:**
- **Vector Store (Dual-Index Strategy):** Embedded chunks are primarily stored locally in **ChromaDB**. However, the ingestion pipeline simultaneously builds a **FAISS** index. FAISS acts as a baseline pure-similarity dense vector store for Capstone analytics, allowing us to mathematically compare and prove the superiority of the advanced ChromaDB hybrid search.
- **Hybrid Search Engine:** 
  - *Semantic Search:* Finds concepts with similar meanings using vector cosine similarity.
  - *Keyword Search (BM25):* Finds exact word matches (crucial for specific fund tickers like 'HYCBF').
- **Reciprocal Rank Fusion (RRF):** The results of the semantic search and keyword search are mathematically fused together to create a single, highly robust candidate list.
- **Cross-Encoder Reranking:** The initial fused candidates are passed through a HuggingFace `sentence-transformer` cross-encoder model. This model scores the exact relationship between the query and the chunk, perfectly re-ordering the top results.
- **Access Control (Entitlement Filtering):** Before the search even begins, a metadata filter (`$in`) is applied to ChromaDB based on the RM's tier. A `standard` RM physically cannot retrieve a `restricted` document.

---

## 4. The Agent's Toolbelt
**Overview:**
The LangGraph agent is granted autonomous access to four specific Python functions, bound using LangChain's `@tool` decorator.

**Technical Deep-Dive:**
1. **`portfolio_lookup_tool`**: Queries `clients.json` to retrieve the client's total AUM, asset allocation, and strict Risk Profile (e.g., Conservative, Aggressive).
2. **`market_data_tool`**: Fetches simulated live pricing, 3-year annualized returns, and volatility for requested tickers.
3. **`rag_retriever_tool`**: The bridge to the ChromaDB index.
4. **`suitability_checker_tool`**: A hard-coded compliance engine.

---

## 5. Token Optimization & LLM Routing
**Overview:**
To ensure the Copilot remains cost-effective and doesn't crash due to context-window bloat, we implemented aggressive optimization strategies.

**Technical Deep-Dive:**
- **Tool Output Filtering:** The `portfolio_lookup_tool` was modified to exclude massive line-item holdings arrays by default, only fetching them if explicitly requested (`include_holdings=True`). The `rag_retriever_tool` was stripped of redundant citation dictionaries, and the default `top_k` was reduced from 5 to 3.
- **LLM Routing (Model Downgrading):** The background tools responsible for generating the client's Sentiment Timeline and Meeting Mood prediction read through massive blocks of historical transcripts. We explicitly downgraded these specific background tools to use `gpt-4o-mini` with a temperature of **0.2**, slashing API token costs by 95% while keeping the core agent on `gpt-4o`.

---

## 6. Guardrails & Compliance Framework
**Overview:**
In Wealth Management, AI hallucinations can lead to severe legal penalties. We built deterministic safety nets that the LLM cannot bypass.

**Technical Deep-Dive:**
- **Pydantic Structured Outputs (V2):** The final brief is not a string of text. The `synthesize_brief` node uses `with_structured_output` to force the LLM to return a strict JSON object matching our `ClientBrief` Pydantic schema.
- **The Suitability Gate:** The `suitability_checker_tool` uses hard-coded Python logic (not LLM guessing) to compare a proposed investment against the client's Risk Profile. If a Conservative client is offered an Aggressive fund, the tool returns `unsuitable`.
- **The Citation Validator:** Found in `src/guardrails/citation_validator.py`. After the LLM generates the JSON brief, this script intercepts it and scans the `citations` array. It cross-references every cited `chunk_id` against the database. If the LLM hallucinated a fake citation, the validator forcibly overwrites the brief's `compliance_status` to `"needs_review"`.

---

## 7. The User Interface (Streamlit)
**Overview:**
The frontend is a dynamic, interactive dashboard built for Relationship Managers.

**Technical Deep-Dive:**
- **Streaming Execution:** The LangGraph execution is streamed using `.stream()`. As the agent thinks and calls tools, the UI dynamically renders these events in real-time using `st.empty()` and expanders.
- **Dynamic CSS Injection:** Overrides Streamlit's default padding and block-container heights to create a seamless, locked side-by-side layout (Chat vs. Portfolio Dashboard).
- **Session State:** Manages the active `client_id`, the `rm_tier` toggle, and the persistent message history to prevent UI refreshes from wiping the chat.

---

## 8. Analytics & Evaluation
**Overview:**
Proving the Copilot is production-ready through mathematical evaluation.

**Technical Deep-Dive:**
- **The Golden Set:** A suite of test queries designed to test specific traps (e.g., tricking the agent into recommending High Yield bonds to a Conservative client).
- **RAGAS (LLM-as-a-Judge):** We evaluate the final output mathematically to ensure:
  - *Retrieval Relevance:* Did the RAG pull the right document?
  - *Suitability Precision:* Did the compliance gate trigger appropriately?
  - *Faithfulness:* Are the final brief's claims 100% backed by the retrieved context chunks (proving a 0% hallucination rate)?
