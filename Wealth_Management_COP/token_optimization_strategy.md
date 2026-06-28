# Token Optimization Strategy for Wealth Manager Copilot

As a LangGraph/RAG application scales, token usage can balloon quickly, leading to high latency and API costs. Based on an audit of the current codebase, here are the top 4 strategies we identified and implemented to drastically improve token efficiency.

## 1. Chat History Pruning (Context Window Management)
**Status:** *Pending / Optional for long sessions*
**Current State:** 
In `src/agent/nodes.py`, the `call_model` node passes the entire `messages` array from the LangGraph state to the LLM on every turn. This includes all past user questions, massive tool outputs (like RAG document chunks), and AI responses.
**Optimization:**
- **Rolling Window:** Keep only the last $N$ turns of conversation (e.g., the last 5 user-assistant pairs).
- **Tool Output Scrubbing:** Once an AI has used a tool's output to generate a final answer, we can remove or summarize the massive `ToolMessage` from the history before the next turn. The AI only needs to remember its *answer*, not the raw 3,000-word source document it read to get there.

## 2. Tool Output Summarization & Filtering
**Status:** ✅ *Implemented*
**Current State:**
When tools like `portfolio_lookup` are called, they used to return the entire raw JSON object for the client, including hundreds of stock holdings. When `rag_retriever_tool` fired, it dumped raw document chunks alongside duplicate citation dictionaries.
**Optimization:**
- **Selective JSON:** In `portfolio_lookup`, we added an `include_holdings` parameter defaulting to `False`. The tool now omits the massive line-item holdings array unless specifically requested by the AI.
- **RAG Chunk Truncation:** We stripped out redundant nested citation dictionaries from the retriever tool output and reduced the default `top_k` documents from 5 down to 3.

## 3. Optimizing Background Tools (Sentiment & Mood)
**Status:** *Pending*
**Current State:**
The `predict_meeting_mood` tool and `analyze_client_sentiment` tool load the entire `interactions.json` array for a client and dump all historical interaction transcripts directly into the LLM.
**Optimization:**
- **Incremental Summarization:** Instead of processing all raw transcripts every time, we should maintain a "Running Client Summary" in the database. When a new interaction happens, we use a small LLM call to update the summary, and pass *only the summary* to the mood predictor.

## 4. LLM Routing (Model Downgrading)
**Status:** ✅ *Implemented*
**Current State:**
We were previously using `settings.openai_model` (currently `gpt-4o`) for everything: casual chat, RAG synthesis, sentiment extraction, and mood prediction. This was needlessly burning premium token quota on backend tasks.
**Optimization:**
- **Task-Specific Models:** We downgraded the isolated background tools (`analyze_client_sentiment` and `predict_meeting_mood`) to use `gpt-4o-mini`. This slashed API token costs for those specific tools by ~95%, while reserving the heavy `gpt-4o` model exclusively for complex financial synthesis and compliance checks in the main chat.
