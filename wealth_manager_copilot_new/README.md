# Wealth Manager Copilot for Relationship Management

A multi-tool agentic assistant for wealth-management relationship managers (RMs) using OpenAI LLMs, LangGraph, and RAG to generate grounded, compliant client interaction briefs.

## 🎯 Project Overview

**Goal:** Build an AI-powered copilot that helps RMs prepare for client interactions by:
- Summarizing client portfolio and risk profile
- Retrieving relevant market/research and product information (RAG)
- Checking suitability and compliance constraints
- Producing grounded, compliant talking points with citations
- Escalating anything requiring licensed advice or human sign-off

## 🏗️ Architecture

### Core Components

1. **Knowledge/Vector Store**
   - Product guides, suitability policies, research notes, compliance rules
   - Chroma DB or FAISS for vector storage with metadata filtering
   - OpenAI embeddings (text-embedding-3-small)

2. **Tools**
   - `portfolio_lookup`: Fetch client holdings and risk profile
   - `market_data`: Retrieve current market data and economic indicators
   - `check_suitability`: Validate recommendations against compliance rules
   - `compliance_gate`: Determine if recommendations need escalation

3. **LangGraph Agent**
   - **plan**: Decompose RM request
   - **gather_portfolio**: Fetch client portfolio data
   - **gather_research**: Retrieve relevant documents via RAG
   - **check_suitability**: Validate against policies
   - **synthesize**: Create grounded ClientBrief with citations
   - **review_gate**: Escalate if needed, else return brief

4. **Frontend**
   - Streamlit dashboard for RM interaction
   - Interactive client selection and request processing
   - Real-time brief generation with citations

## 📁 Project Structure

```
wealth_manager_copilot/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── models.py                # Pydantic data models
│   ├── ingestion.py             # Knowledge ingestion pipeline
│   ├── retriever.py             # Vector store and RAG retriever
│   ├── tools.py                 # Agent tools (portfolio, market data, etc)
│   └── agent.py                 # LangGraph agent orchestration
├── notebooks/
│   └── evaluation.ipynb          # Evaluation and analytics
├── data/
│   ├── sample_knowledge/         # Sample documents (PDF, DOCX, TXT, CSV)
│   └── vector_store/             # Persisted vector store
├── app.py                        # Streamlit dashboard
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables
└── README.md                     # This file
```

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=sk-your-api-key-here
```

### 3. Prepare Knowledge Documents

Place your knowledge documents in `data/sample_knowledge/`:
- Product guides (PDF, DOCX, TXT)
- Compliance and suitability policies (TXT)
- Market research (TXT, CSV)

Minimum 10 source documents across types required.

### 4. Run the Application

**Option A: Streamlit Dashboard**
```bash
streamlit run app.py
```

**Option B: Direct Agent Usage**
```python
from src.agent import create_langgraph_agent
from src.retriever import RAGRetriever

# Initialize
retriever = RAGRetriever(vector_store_type="chroma")
agent = create_langgraph_agent(retriever=retriever)

# Run
brief = agent.run_agent(
    client_id="C-204",
    request="Prepare talking points for quarterly review"
)

print(brief)
```

### 5. Run Evaluation

```bash
jupyter notebook notebooks/evaluation.ipynb
```

## 📊 Data Models

### ClientBrief (Main Output)
```python
{
    "client_id": str,
    "brief_id": str,
    "risk_profile": RiskProfile,
    "portfolio_summary": PortfolioSummary,
    "recommendations": List[Recommendation],
    "talking_points": List[str],
    "compliance_status": ComplianceStatus,
    "escalated_items": List[Dict],
    "metadata": Dict
}
```

### Recommendation
```python
{
    "idea": str,
    "rationale": str,
    "suitability": SuitabilityAssessment,
    "citations": List[Citation],
    "confidence_score": float,
    "action_required": bool
}
```

## 🔧 Key Features

### ✅ Safety & Guardrails
- Grounding + citations for every recommendation
- Compliance/suitability gate before surfacing recommendations
- No personalized investment advice without escalation
- Entitlement filtering on sensitive knowledge
- Temperature control and bounded agent steps
- Full trace/audit logging

### 📈 Hybrid RAG
- Semantic similarity search (embeddings)
- Metadata filtering (document type, sensitivity, date)
- Structure-aware chunking with preserved context
- Multi-document citation tracking

### 🔄 Multi-Step Orchestration
- ReAct-style tool calling
- Bounded loops (max steps configuration)
- Conditional edges for compliance gate
- Human-in-the-loop interrupt points
- LangSmith tracing for observability

## 🧪 Evaluation Metrics

The `evaluation.ipynb` notebook measures:

1. **Retrieval Relevance** - Similarity scores, coverage, document diversity
2. **Suitability Gate Precision** - Accuracy of suitability checks
3. **Recommendation Groundedness** - Citation coverage, confidence scores
4. **Agent Performance** - Success rate, recommendation quality on golden set
5. **Single-hop vs Multi-hop** - Retrieval strategy comparison

### Golden Set Test Queries
- 'Prepare talking points for client C-204's quarterly review' → grounded ClientBrief
- 'Is fund XYZ suitable for a conservative client?' → suitability check + citation
- 'Summarize the portfolio risk for client C-204' → portfolio summary
- Recommendation requiring licensed advice → escalation, not auto-advice

## 🎮 Sample Usage

### Query 1: Quarterly Review
```
Input: "Prepare talking points for client C-204's quarterly review"
Output: ClientBrief with:
  - Portfolio summary
  - Recommendations aligned with Growth profile
  - Talking points for discussion
  - Citations to supporting documents
```

### Query 2: Suitability Check
```
Input: "Is fund XYZ suitable for a conservative client?"
Output: Suitability assessment with:
  - Compliance check result
  - Policy violations (if any)
  - Citations to suitability rules
  - Escalation flag if needed
```

### Query 3: Access Control
```
Input: Restricted research for unentitled RM
Output: Filtered results excluding sensitive documents
```

## 📚 Sample Knowledge Documents

The system includes sample documents for:
- **Product Guide**: Growth Equity Fund specifications
- **Compliance Policy**: Suitability rules by risk profile
- **Market Research**: Q2 2024 economic outlook

Create your own by placing files in `data/sample_knowledge/`.

## 🔒 Security & Compliance

- API key management via environment variables
- Document sensitivity levels (public/restricted)
- Access control filters
- Compliance gate enforcement
- Audit logging of all operations

## 🧠 LangGraph Flow

```
START
  ↓
[plan] - Decompose request
  ↓
[gather_portfolio] - Fetch holdings
  ↓
[gather_research] - RAG retrieval
  ↓
[check_suitability] - Policy validation
  ↓
[synthesize] - Create brief
  ↓
[review_gate] - Check escalation
  ↓
  ├→ Escalate (Needs Review/Blocked)
  │
  └→ Return Brief (Cleared)
```

## 📊 Tools & Dependencies

- **LLM**: OpenAI GPT-4o
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector Store**: Chroma / FAISS
- **Framework**: LangGraph, LangChain
- **Data Validation**: Pydantic
- **UI**: Streamlit
- **Evaluation**: RAGAS, LangSmith
- **Notebooks**: Jupyter

## 📝 Submission Deliverables

- [x] Source code repository
- [x] requirements.txt
- [x] Environment configuration (.env template)
- [x] Evaluation notebook with metrics
- [x] Sample input data
- [x] Working demo (Streamlit app)
- [x] Golden set test queries
- [x] Agent trace/audit logging

## 🚦 Testing

Run the evaluation notebook to test all components:

```bash
# Install Jupyter if needed
pip install jupyter

# Run notebook
jupyter notebook notebooks/evaluation.ipynb
```

This will generate:
- Retrieval quality metrics
- Suitability gate accuracy
- Agent performance on golden set
- Recommendation groundedness scores

## 🤝 Contributing

To add new tools or modify the agent flow:

1. Add tool implementation in `src/tools.py`
2. Register in `create_tools_dict()`
3. Update agent flow in `src/agent.py`
4. Add tests in evaluation notebook

## 📖 Documentation

See inline docstrings in each module for detailed API documentation.

Key modules:
- `models.py` - Pydantic schema definitions
- `agent.py` - Agent orchestration and workflow
- `retriever.py` - Vector store and RAG implementation
- `tools.py` - Agent tools and utilities

## ⚠️ Disclaimer

This system is **decision support only**, not automated investment advice. All recommendations require compliance gate checks and potential human escalation. Relationship managers should review all briefs before presenting to clients.

## 📄 License

This is a capstone project for educational purposes.

---

**Built with LangGraph | OpenAI | Chroma**
