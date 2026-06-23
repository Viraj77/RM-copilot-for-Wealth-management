# 💼 Wealth Manager Copilot - Run Guide

An AI-powered agentic application designed to assist Relationship Managers (RMs) in answering complex client queries, fetching market data, scanning firm policies, and ensuring all generated recommendations comply with strict firm suitability guardrails.

This project uses **LangGraph** for agent reasoning and **ChromaDB** with **FAISS** for Retrieval-Augmented Generation (RAG).

---

## 🚀 Quick Start Guide

Follow these steps to run the complete end-to-end pipeline.

### 1. Configure the Environment
The application requires an OpenAI API key to function (for both the LLM and the embedding models).
1. Rename `.env.example` to `.env`.
2. Open `.env` and add your OpenAI API key to line 2:
   ```ini
   OPENAI_API_KEY="sk-..."
   ```

### 2. Run the Data Ingestion Pipeline
Before the agent can answer questions, it needs to ingest the knowledge base (product guides, policies, research notes) into the vector database.
1. Open your terminal in the project root.
2. Run the ingestion command:
   ```bash
   python -m src.ingestion.indexer --reset
   ```
   *This will chunk the raw documents, embed them using OpenAI, and save the ChromaDB and FAISS indexes in the `./data/` folder.*

### 3. Launch the Application (Streamlit)
Start the frontend interface to chat with the Copilot. Since we have a handy python launcher to bypass potential PATH issues, run:
```bash
python run_streamlit.py
```
*The app will automatically open in your default browser at `http://localhost:8501`. If prompted for an email address in the terminal during the first run, simply hit `Enter` to skip.*

### 4. Run the Test Suite (Optional)
To verify that the guardrails, tools, and agent nodes are functioning perfectly, run the automated tests:
```bash
python -m pytest -v
```

---

## 🏗️ Architecture Overview

- **Tools Layer (`src/tools/`)**: Modular plugins allowing the agent to fetch client portfolios, live market data, and RAG document chunks.
- **Agent Layer (`src/agent/`)**: A LangGraph state machine that routes queries between the LLM and the tool execution nodes.
- **Guardrails (`src/guardrails/`)**: A strict defense-in-depth system that scans the final LLM output to catch and block prohibited guarantees, unapproved licensed advice, and entitlement leaks.
- **Frontend (`app/`)**: A Streamlit chat interface with an escalation viewer to trace the agent's tool calls and compliance warnings.
