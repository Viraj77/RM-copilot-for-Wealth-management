# QUICK START GUIDE

## 📋 Project Summary

Complete implementation of **Wealth Manager Copilot** - an AI-powered assistant for relationship managers to generate grounded, compliant client interaction briefs using LangGraph and RAG.

## 🎯 What Was Generated

### Core Components (src/)
- **models.py** - Pydantic data models for ClientBrief, Recommendations, Portfolios
- **ingestion.py** - Knowledge pipeline: loading, preprocessing, chunking documents
- **retriever.py** - Vector store (Chroma/FAISS) and hybrid RAG implementation
- **tools.py** - Agent tools: portfolio lookup, market data, suitability checking, compliance gate
- **agent.py** - LangGraph orchestration with multi-step workflow
- **__init__.py** - Package initialization

### Application Layer
- **app.py** - Streamlit dashboard for interactive brief generation
- **main.py** - Demo script with golden set test queries
- **test_setup.py** - Validation script to verify all components

### Configuration & Documentation
- **requirements.txt** - All Python dependencies
- **.env.template** - Environment configuration template
- **config.py** - Settings management from environment
- **README.md** - Comprehensive project documentation

### Evaluation & Analysis
- **notebooks/evaluation.ipynb** - Full evaluation notebook with:
  - Retrieval relevance metrics
  - Suitability gate precision analysis
  - Agent performance on golden set
  - Recommendation groundedness evaluation
  - Single-hop vs multi-hop retrieval comparison

### Data Directory (auto-created)
- **data/sample_knowledge/** - Sample documents (product guides, policies, research)
- **data/vector_store/** - Persisted vector store (Chroma/FAISS)

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Copy template
cp .env.template .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-...
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Validate Setup
```bash
python test_setup.py
```

### 4. Run Interactive Dashboard
```bash
streamlit run app.py
```

Access at: http://localhost:8501

### 5. Run Demo with Golden Set Queries
```bash
python main.py
```

### 6. Run Evaluation Notebook
```bash
jupyter notebook notebooks/evaluation.ipynb
```

## 📊 Architecture

```
REQUEST
  ↓
[PLAN] Decompose request
  ↓
[GATHER_PORTFOLIO] Fetch client data
  ↓
[GATHER_RESEARCH] RAG retrieval
  ↓
[CHECK_SUITABILITY] Policy validation
  ↓
[SYNTHESIZE] Create brief with citations
  ↓
[REVIEW_GATE] Compliance check
  ↓
RETURN BRIEF or ESCALATE
```

## 🧪 Test Queries (Golden Set)

The system is tested on:

1. **Query**: "Prepare talking points for client C-204's quarterly review"
   - **Expected**: Grounded brief with portfolio summary & recommendations

2. **Query**: "Is fund XYZ suitable for a conservative client?"
   - **Expected**: Suitability check with compliance assessment

3. **Query**: "Summarize the portfolio risk for client C-204"
   - **Expected**: Portfolio summary with risk analysis

4. **Query**: "What recommendations would you make for client C-201?"
   - **Expected**: Personalized recommendations with citations

## 📈 Key Features Implemented

✅ **Multi-Step Agent Orchestration**
- ReAct-style tool calling
- Bounded loops with max steps
- Human-in-the-loop escalation

✅ **Hybrid RAG System**
- Semantic similarity search
- Metadata filtering (doc type, sensitivity)
- Citation tracking across sources

✅ **Safety & Guardrails**
- Grounding + citations for every recommendation
- Compliance gate before surfacing recommendations
- Escalation for licensed advice

✅ **Structured Output**
- Pydantic models with validation
- JSON serialization
- Type-safe data pipeline

✅ **Comprehensive Evaluation**
- Retrieval quality metrics
- Suitability gate precision
- Recommendation faithfulness
- Agent performance tracking

## 📚 Sample Knowledge Documents

The system includes 3 auto-generated sample documents:
1. **product_guide.txt** - Growth Equity Fund details
2. **compliance_policy.txt** - Suitability rules by risk profile
3. **market_research.txt** - Q2 2024 economic outlook

Add more documents to `data/sample_knowledge/` and re-run ingestion.

## 🔧 Configuration

Key settings in `.env`:
```
OPENAI_API_KEY=sk-...           # Required: OpenAI API key
OPENAI_MODEL=gpt-4o             # LLM model
VECTOR_STORE_TYPE=chroma        # Vector store (chroma/faiss)
MAX_AGENT_STEPS=20              # Max agent iterations
AGENT_TEMPERATURE=0.2           # LLM temperature
```

## 📊 Evaluation Metrics

The evaluation notebook measures:

| Metric | Description |
|--------|-------------|
| **Retrieval Coverage** | Avg documents retrieved per query |
| **Relevance Score** | Average semantic similarity score |
| **Suitability Precision** | % of correct suitability assessments |
| **Citation Coverage** | Avg citations per recommendation |
| **Confidence Score** | LLM confidence in recommendations |
| **Compliance Pass Rate** | % briefs passing compliance gate |

## 🎯 Sample Clients

Available for testing:
- **C-201**: John Smith (Balanced) - $500k portfolio
- **C-202**: Jane Doe (Conservative) - $750k portfolio
- **C-203**: Robert Johnson (Aggressive) - $1M portfolio
- **C-204**: Sarah Wilson (Growth) - $600k portfolio

## 📁 Directory Structure

```
wealth_manager_copilot/
├── src/                         # Core modules
│   ├── models.py               # Data models
│   ├── ingestion.py            # Document ingestion
│   ├── retriever.py            # Vector store & RAG
│   ├── tools.py                # Agent tools
│   ├── agent.py                # LangGraph agent
│   └── __init__.py
├── notebooks/
│   └── evaluation.ipynb         # Full evaluation
├── data/
│   ├── sample_knowledge/        # Auto-generated documents
│   └── vector_store/            # Persisted vector DB
├── app.py                       # Streamlit dashboard
├── main.py                      # Demo script
├── test_setup.py                # Validation tests
├── config.py                    # Settings management
├── requirements.txt             # Dependencies
├── .env.template                # Config template
└── README.md                    # Full documentation
```

## ✨ Key Innovations

1. **Grounded RAG** - Every recommendation includes citations to source documents
2. **Compliance Gate** - Automated suitability checking before recommendation
3. **Multi-Hop Retrieval** - Context-aware document retrieval
4. **Tool Orchestration** - Portfolio, market data, and suitability in one flow
5. **LangGraph Integration** - Production-ready agent orchestration
6. **Streamlit UI** - Interactive RM-facing dashboard

## 🔒 Security Notes

- API keys managed via environment variables (.env)
- Document sensitivity levels (public/restricted)
- Access control filters in retriever
- Full audit logging
- No ungrounded investment advice

## 📝 Deliverables Checklist

- [x] Source code repository
- [x] requirements.txt with all dependencies
- [x] Environment configuration template
- [x] Comprehensive README
- [x] Evaluation notebook with metrics
- [x] Sample knowledge documents
- [x] Streamlit dashboard (live UI)
- [x] Demo script with golden set queries
- [x] Validation test suite
- [x] Configuration management
- [x] Agent trace logging
- [x] Citation system

## 🚀 Next Steps

1. **Add Custom Documents**
   - Place PDFs/Word docs in `data/sample_knowledge/`
   - Re-run ingestion to index

2. **Configure LangSmith Tracing**
   - Set LANGSMITH_API_KEY in .env
   - View agent traces in LangSmith dashboard

3. **Customize Suitability Rules**
   - Edit `SUITABILITY_RULES` in src/tools.py
   - Add your firm's compliance rules

4. **Deploy to Production**
   - Use Streamlit Cloud, AWS, or your platform
   - Update .env with production secrets

## 📞 Support

- Check `README.md` for detailed architecture
- Review `src/agent.py` for workflow logic
- See `notebooks/evaluation.ipynb` for metrics
- Run `python test_setup.py` to debug issues

---

**Built with LangGraph | OpenAI | Chroma | Streamlit**

Ready to help RMs make compliant, grounded recommendations! 🎯
