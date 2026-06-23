"""
Streamlit dashboard for Wealth Manager Copilot
"""
import streamlit as st
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
import os
import shutil

from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import models and retriever
from src.retriever import RAGRetriever
from src.ingestion import create_sample_knowledge_documents
from src.models import ClientBrief, ComplianceStatus
from langchain_openai import ChatOpenAI


def initialize_session_state():
    """Initialize Streamlit session state."""
    if "agent" not in st.session_state:
        st.session_state.agent = None
    
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    
    if "brief" not in st.session_state:
        st.session_state.brief = None
    
    if "knowledge_loaded" not in st.session_state:
        st.session_state.knowledge_loaded = False
    
    if "selected_sources" not in st.session_state:
        st.session_state.selected_sources = []
    
    if "uploaded_doc_count" not in st.session_state:
        st.session_state.uploaded_doc_count = 0

    if "request_prompt" not in st.session_state:
        st.session_state.request_prompt = "Retrieve relevant documents for this prompt"

    if "selected_client_id" not in st.session_state:
        st.session_state.selected_client_id = None


def load_knowledge_base():
    """Load and initialize the knowledge base."""
    try:
        with st.spinner("Loading knowledge base..."):
            # Initialize retriever
            retriever = RAGRetriever(
                embedding_model="text-embedding-3-small",
                vector_store_type="faiss",
                persist_dir="./data/vector_store",
                openai_api_key=settings.openai_api_key
            )
            retriever.top_k = 5
            
            # Try to load existing store
            retriever.load_existing_store()
            stats = retriever.get_stats()
            
            if stats.get("total_documents", 0) == 0:
                st.warning("Vector store empty. Creating sample documents...")
                create_sample_knowledge_documents("./data/sample_knowledge")
                
                # Ingest sample documents
                from src.ingestion import KnowledgeIngestionPipeline
                pipeline = KnowledgeIngestionPipeline()
                docs = pipeline.run_ingestion_pipeline("./data/sample_knowledge")
                retriever.index_documents(docs, recreate=True)
            
            st.session_state.retriever = retriever
            st.session_state.knowledge_loaded = True
            st.success(f"Knowledge base loaded! ({stats.get('total_documents', 0)} documents)")
    
    except Exception as e:
        st.error(f"Error loading knowledge base: {e}")
        logger.error(f"Knowledge base loading failed: {e}")


def save_uploaded_files(uploaded_files, target_dir: str) -> int:
    """Save uploaded files to disk for ingestion."""
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    count = 0
    for uploaded_file in uploaded_files:
        file_path = Path(target_dir) / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        count += 1
    return count


def ingest_uploaded_files(uploaded_files, chunk_size: int, chunk_overlap: int):
    """Ingest uploaded files into the knowledge base."""
    upload_dir = "./data/user_uploads"
    saved_count = save_uploaded_files(uploaded_files, upload_dir)

    from src.ingestion import KnowledgeIngestionPipeline
    pipeline = KnowledgeIngestionPipeline(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    docs = pipeline.run_ingestion_pipeline(upload_dir)

    retriever = RAGRetriever(
        embedding_model="text-embedding-3-small",
        vector_store_type="faiss",
        persist_dir="./data/vector_store",
        openai_api_key=settings.openai_api_key
    )
    retriever.index_documents(docs, recreate=True)

    st.session_state.retriever = retriever
    st.session_state.knowledge_loaded = True
    st.session_state.uploaded_doc_count = len(docs)
    return saved_count, len(docs)


def reload_knowledge_base():
    """Reload the existing vector store from disk."""
    retriever = RAGRetriever(
        embedding_model="text-embedding-3-small",
        vector_store_type="faiss",
        persist_dir="./data/vector_store",
        openai_api_key=settings.openai_api_key
    )
    retriever.load_existing_store()
    stats = retriever.get_stats()
    if stats.get("total_documents", 0) == 0:
        return False, 0

    st.session_state.retriever = retriever
    st.session_state.knowledge_loaded = True
    st.session_state.uploaded_doc_count = stats["total_documents"]
    st.session_state.selected_sources = []
    return True, stats["total_documents"]


def clear_uploaded_knowledge():
    """Clear the current knowledge base and uploaded files."""
    upload_dir = Path("./data/user_uploads")
    vector_dir = Path("./data/vector_store")
    removed_count = 0

    if st.session_state.retriever is not None:
        try:
            removed_count = st.session_state.retriever.clear_store()
        except Exception as e:
            logger.error(f"Failed to clear vector store: {e}")

    if upload_dir.exists():
        shutil.rmtree(upload_dir, ignore_errors=True)
    if vector_dir.exists():
        shutil.rmtree(vector_dir, ignore_errors=True)

    st.session_state.retriever = None
    st.session_state.agent = None
    st.session_state.brief = None
    st.session_state.knowledge_loaded = False
    st.session_state.selected_sources = []
    st.session_state.uploaded_doc_count = 0
    st.session_state.request_prompt = ""
    return removed_count


def rerun_app():
    """Trigger a Streamlit rerun across supported versions."""
    rerun = getattr(st, "rerun", None)
    if rerun is None:
        rerun = getattr(st, "experimental_rerun", None)
    if rerun is None:
        raise RuntimeError("This Streamlit version does not expose a rerun API.")
    rerun()


def initialize_agent():
    """Initialize the LangGraph agent."""
    try:
        agent = create_langgraph_agent(
            llm_model="gpt-4o",
            retriever=st.session_state.retriever,
            openai_api_key=settings.openai_api_key
        )
        st.session_state.agent = agent
        return agent
    except Exception as e:
        st.error(f"Error initializing agent: {e}")
        logger.error(f"Agent initialization failed: {e}")
        return None


def render_brief(brief: ClientBrief):
    """Render the generated brief."""
    st.divider()
    st.subheader("📊 Generated Client Brief")
    
    # Header info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Client ID", brief.client_id)
    with col2:
        st.metric("Risk Profile", brief.risk_profile.value)
    with col3:
        st.metric("Compliance Status", brief.compliance_status.value)
    with col4:
        st.metric("Recommendations", len(brief.recommendations))
    
    # Portfolio summary
    st.subheader("💼 Portfolio Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Value", f"${brief.portfolio_summary.total_value:,.0f}")
    with col2:
        st.metric("Risk Score", f"{brief.portfolio_summary.risk_score:.1f}/10")
    with col3:
        st.metric("Equities", f"{brief.portfolio_summary.allocation.get('equities', 0)*100:.0f}%")
    with col4:
        st.metric("Fixed Income", f"{brief.portfolio_summary.allocation.get('fixed_income', 0)*100:.0f}%")
    
    # Allocation breakdown
    st.write("**Asset Allocation:**")
    allocation_data = {
        k.replace("_", " ").title(): v * 100
        for k, v in brief.portfolio_summary.allocation.items()
    }
    st.bar_chart(allocation_data)
    
    # Talking points
    st.subheader("💬 Talking Points")
    for i, point in enumerate(brief.talking_points, 1):
        st.write(f"{i}. {point}")
    
    # Recommendations
    if brief.recommendations:
        st.subheader("🎯 Recommendations")
        
        for i, rec in enumerate(brief.recommendations, 1):
            with st.expander(f"Recommendation {i}: {rec.idea}"):
                st.write(f"**Idea:** {rec.idea}")
                st.write(f"**Rationale:** {rec.rationale}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Confidence", f"{rec.confidence_score*100:.0f}%")
                with col2:
                    suitable = "✅ Yes" if rec.suitability.suitable_for_profile else "❌ No"
                    st.metric("Suitable", suitable)
                with col3:
                    st.metric("Needs Review", "Yes" if rec.action_required else "No")
                
                if rec.suitability.compliance_notes:
                    st.warning(f"Compliance Note: {rec.suitability.compliance_notes}")
                
                # Citations
                if rec.citations:
                    st.write("**Sources:**")
                    for citation in rec.citations:
                        st.caption(
                            f"📄 {citation.source} ({citation.doc_type}) - {citation.chunk_text[:100]}..."
                        )
    
    # Escalations
    if brief.escalated_items:
        st.subheader("⚠️ Escalated Items")
        for item in brief.escalated_items:
            st.warning(
                f"**{item.get('recommendation')}** - {item.get('escalation_type')}: {item.get('reason')}"
            )
    
    # Metadata
    with st.expander("📋 Metadata"):
        st.json(brief.metadata)
    
    # Disclaimer
    st.info(f"ℹ️ {brief.disclaimer}")


def summarize_retrieved_content(results, query):
    """Summarize retrieved chunks using LLM."""
    if not results:
        return None
    
    # Combine retrieved chunks
    combined_content = "\n\n---\n\n".join([f"[{doc['metadata'].get('source', 'unknown')}]\n{doc['content']}" for doc in results])
    
    # Create LLM prompt for summarization
    prompt = f"""Based on the following retrieved documents, provide a concise summary that answers the user's query.

**User Query:** {query}

**Retrieved Documents:**
{combined_content}

**Instructions:**
- Summarize the key information relevant to the query
- Keep it concise (3-5 sentences)
- Mention the sources where information comes from
- Focus on actionable insights"""
    
    try:
        from langchain_core.messages import HumanMessage
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,
            max_tokens=500,
            api_key=settings.openai_api_key
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        logger.error(f"Error summarizing content: {e}")
        return None


def render_retrieval_results(results, query):
    """Render document retrieval results with LLM summary."""
    st.divider()
    st.subheader("📄 Retrieval Results")
    if not results:
        st.info("No documents retrieved for this query.")
        return

    # Show LLM-generated summary first
    st.subheader("🎯 Summary")
    with st.spinner("Generating summary..."):
        summary = summarize_retrieved_content(results, query)
        if summary:
            st.success(summary)
        else:
            st.info("Could not generate summary. Review retrieved documents below.")
    
    st.divider()
    st.subheader("📚 Retrieved Chunks")
    st.markdown(f"**{len(results)} documents retrieved**")
    
    for i, doc in enumerate(results, 1):
        with st.expander(f"Chunk {i}: {doc['metadata'].get('source', 'unknown')} (Relevance: {doc['score']:.2%})"):
            st.write(doc["content"])
            st.markdown(
                f"**Source:** {doc['metadata'].get('source', 'unknown')} | "
                f"**Type:** {doc['metadata'].get('doc_type', 'unknown')} | "
                f"**ID:** {doc['metadata'].get('chunk_id', 'unknown')}"
            )


def auto_load_knowledge_base():
    """Auto-load existing knowledge base on startup if available."""
    if not st.session_state.knowledge_loaded and st.session_state.retriever is None:
        try:
            # Try to load from disk
            retriever = RAGRetriever(
                embedding_model="text-embedding-3-small",
                vector_store_type="faiss",
                persist_dir="./data/vector_store",
                openai_api_key=settings.openai_api_key
            )
            retriever.load_existing_store()
            stats = retriever.get_stats()
            
            if stats.get("total_documents", 0) > 0:
                st.session_state.retriever = retriever
                st.session_state.knowledge_loaded = True
                st.session_state.uploaded_doc_count = stats.get("total_documents", 0)
                logger.info(f"Auto-loaded knowledge base: {stats.get('total_documents', 0)} documents")
        except Exception as e:
            logger.debug(f"No existing knowledge base to auto-load: {e}")


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Wealth Manager Copilot",
        page_icon="💰",
        layout="wide"
    )
    
    st.title("💰 Wealth Manager Copilot")
    st.markdown("**Prepare grounded, compliant client interaction briefs powered by LangGraph**")
    
    # Initialize session
    initialize_session_state()
    
    # Auto-load existing knowledge base if available
    auto_load_knowledge_base()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.write("Use the main panel for prompt entry and retrieval results.")
        st.divider()

        st.header("👤 Client Selection")
        MOCK_CLIENTS = {
            "— None —": None,
            "C-201 · John Smith (Balanced, $500K)": "C-201",
            "C-202 · Jane Doe (Conservative, $750K)": "C-202",
            "C-203 · Robert Johnson (Aggressive, $1M)": "C-203",
            "C-204 · Sarah Wilson (Growth, $600K)": "C-204",
        }
        client_label = st.selectbox(
            "Select client",
            list(MOCK_CLIENTS.keys()),
            index=0,
            key="client_selector"
        )
        st.session_state.selected_client_id = MOCK_CLIENTS[client_label]

        if st.session_state.selected_client_id:
            from src.tools import PortfolioLookupTool
            _pt = PortfolioLookupTool()
            _result = _pt(st.session_state.selected_client_id)
            if _result["success"]:
                _p = _result["data"]
                st.markdown(f"**Name:** {_p['client_name']}")
                st.markdown(f"**Risk Profile:** {_p['risk_profile']}")
                st.markdown(f"**Portfolio Value:** ${_p['total_value']:,.0f}")
                st.markdown(f"**Risk Score:** {_p['risk_score']}/10")
                st.markdown("**Holdings:**")
                for h in _p["holdings"]:
                    st.caption(f"• {h['ticker']} — {h['name']} ({h['allocation']*100:.0f}%)")
        st.divider()

        st.header("🧠 Prompt-only retrieval")
        st.write("Select a client above for a full brief, or just enter a prompt below.")
        st.divider()

        if st.session_state.knowledge_loaded:
            st.success(f"Knowledge base loaded: {st.session_state.uploaded_doc_count} chunks")
            col1, col2 = st.columns([1, 1])
            if col1.button("🔄 Reload Knowledge Base"):
                success, total = reload_knowledge_base()
                if success:
                    st.success(f"Reloaded knowledge base with {total} chunks.")
                else:
                    st.error("No vector store available to reload.")
            if col2.button("🧹 Clear Knowledge Base"):
                removed_count = clear_uploaded_knowledge()
                st.success(f"Cleared uploaded files and removed {removed_count} embeddings from the knowledge base.")
                rerun_app()
        else:
            st.info("Ingest documents in the sidebar or reload an existing knowledge base.")

        st.divider()
        ingest_enabled = st.checkbox(
            "Enable knowledge ingestion",
            value=False,
            help="Show upload and ingestion controls in the sidebar."
        )

        if ingest_enabled:
            st.subheader("📥 Upload documents")
            uploaded_files = st.file_uploader(
                "Upload knowledge documents",
                type=["pdf", "docx", "txt", "csv"],
                accept_multiple_files=True,
                key="upload_files"
            )

            chunk_size = st.number_input(
                "Chunk size",
                min_value=100,
                max_value=2000,
                value=settings.chunk_size,
                step=50,
                key="upload_chunk_size"
            )
            chunk_overlap = st.number_input(
                "Chunk overlap",
                min_value=0,
                max_value=max(0, chunk_size - 50),
                value=settings.chunk_overlap,
                step=10,
                key="upload_chunk_overlap"
            )

            if st.button("Ingest uploaded files", key="ingest_uploaded_files"):
                if not uploaded_files:
                    st.warning("Please upload at least one document before ingestion.")
                else:
                    progress = st.progress(0)
                    status = st.empty()
                    status.text("Saving uploaded files...")
                    saved_count = save_uploaded_files(uploaded_files, "./data/user_uploads")
                    progress.progress(20)

                    status.text("Running ingestion pipeline...")
                    from src.ingestion import KnowledgeIngestionPipeline
                    pipeline = KnowledgeIngestionPipeline(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    docs = pipeline.run_ingestion_pipeline("./data/user_uploads")
                    progress.progress(60)

                    status.text("Indexing documents...")
                    retriever = RAGRetriever(
                        embedding_model="text-embedding-3-small",
                        vector_store_type="faiss",
                        persist_dir="./data/vector_store",
                        openai_api_key=settings.openai_api_key
                    )
                    retriever.index_documents(docs, recreate=True)
                    progress.progress(100)

                    st.session_state.retriever = retriever
                    st.session_state.knowledge_loaded = True
                    st.session_state.uploaded_doc_count = len(docs)
                    st.session_state.selected_sources = []

                    status.text(f"Ingestion complete: {saved_count} files, {len(docs)} chunks.")
                    st.success(f"Uploaded {saved_count} files and created {len(docs)} chunks.")

        st.divider()
        st.subheader("📄 Document selection")
        if st.session_state.retriever:
            doc_list = st.session_state.retriever.list_documents()
            source_options = sorted({doc["source"] for doc in doc_list if doc.get("source")})
            if source_options:
                current_default = st.session_state.get("selected_sources", source_options)
                selected_sources = st.multiselect(
                    "Choose source documents",
                    source_options,
                    default=current_default,
                    key="sidebar_selected_sources"
                )
                st.session_state.selected_sources = selected_sources
                st.markdown(f"**{len(doc_list)} document chunks available.**")
                with st.expander("View loaded source documents"):
                    sources_df = [
                        {
                            "Source": doc["source"],
                            "Type": doc["doc_type"],
                            "Chunk": doc["chunk_id"],
                            "Preview": doc["content_preview"]
                        }
                        for doc in doc_list
                    ]
                    st.dataframe(sources_df)
            else:
                st.info("No document metadata found in the loaded knowledge base.")
        else:
            st.info("Retriever not initialized yet.")

    # Main content
    if st.session_state.knowledge_loaded:
        st.success(f"✅ Knowledge base ready with {st.session_state.uploaded_doc_count} chunks")
    else:
        st.info("💡 No knowledge base loaded. Enable ingestion in the sidebar to upload documents.")

    st.markdown("---")

    st.subheader("✍️ Prompt")
    request = st.text_area(
        "Prompt",
        key="request_prompt",
        height=120,
        placeholder="Enter your query prompt here"
    )

    col_retrieve, col_brief = st.columns([1, 1])
    with col_retrieve:
        submit_button = st.button("Retrieve Documents", type="primary", use_container_width=True)
    with col_brief:
        generate_brief_button = st.button(
            "Generate Client Brief",
            type="secondary",
            use_container_width=True,
            disabled=not st.session_state.selected_client_id
        )

    if generate_brief_button and st.session_state.selected_client_id:
        if not st.session_state.retriever:
            st.warning("No knowledge base loaded — brief will be generated without RAG context.")
        with st.spinner(f"Generating brief for {st.session_state.selected_client_id}..."):
            try:
                from src.agent import create_langgraph_agent
                agent = create_langgraph_agent(
                    llm_model="gpt-4o",
                    retriever=st.session_state.retriever,
                    openai_api_key=settings.openai_api_key
                )
                brief = agent.run_agent(
                    client_id=st.session_state.selected_client_id,
                    request=request or "Prepare a comprehensive wealth management brief for this client."
                )
                st.session_state.brief = brief
                rerun_app()
            except Exception as e:
                st.error(f"Failed to generate brief: {e}")
                logger.error(f"Brief generation failed: {e}")

    if st.button("Clear Prompt"):
        st.session_state.request_prompt = ""
        rerun_app()
    
    
    # Query controls
    st.subheader("🔍 Query settings")
    top_k = st.number_input(
        "Top K results",
        min_value=1,
        max_value=20,
        value=5,
        step=1
    )

    if submit_button and request:
        if not st.session_state.retriever:
            st.error("📋 No knowledge base available. Please enable ingestion in the sidebar and upload documents to enable retrieval.")
        else:
            try:
                with st.spinner("Retrieving documents..."):
                    st.session_state.retriever.top_k = top_k
                    selected_sources = st.session_state.get("selected_sources", None)
                    results = st.session_state.retriever.hybrid_retrieve(
                        request,
                        k=top_k,
                        sources=selected_sources
                    )
                    render_retrieval_results(results, request)
            except Exception as e:
                st.error(f"Error retrieving documents: {e}")
                logger.error(f"Document retrieval failed: {e}")

    elif st.session_state.brief:
        # Display previously generated brief
        render_brief(st.session_state.brief)


if __name__ == "__main__":
    main()
