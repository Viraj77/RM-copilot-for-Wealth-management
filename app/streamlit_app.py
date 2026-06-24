import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.ingestion import build_vector_store
from src.agent import build_graph, extract_client_id
from src.config import USER_DOCS


@st.cache_resource
def load_agent():

    return build_graph()

# ====================================================
# LOAD AGENT
# ====================================================

agent = load_agent()

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Wealth RM Copilot",
    page_icon="💼",
    layout="wide"
)

# =====================================================
# HEADER
# =====================================================

st.markdown("""
# 💼 Wealth Relationship Manager Copilot

AI-powered wealth management assistant for:
- Portfolio analysis
- Market research
- Suitability checks
- Product insights
- RM meeting preparation
- Compliance-aware recommendations
""")

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("📂 Knowledge Center")

    st.markdown("""
Upload:
- Research reports
- Product documents
- Compliance policies
- Portfolio files
- Client documents
""")

    uploaded = st.file_uploader(
        "Upload Documents",
        accept_multiple_files=True,
        type=[
            "pdf",
            "csv",
            "txt",
            "docx"
        ]
    )

    # =========================================
    # Save + Embed
    # =========================================

    if uploaded:

        for file in uploaded:

            save_path = USER_DOCS / file.name

            with open(save_path, "wb") as f:
                f.write(file.read())

        with st.spinner("Creating embeddings..."):

            chunks = build_vector_store()

        st.success(
            f"""
Documents uploaded successfully.

Chunks indexed: {chunks}
"""
        )

# =====================================================
# QUERY INPUT
# =====================================================

st.markdown("## RM Request")

query = st.text_area(
    "",
    height=100,
    placeholder="""
Examples:

• Prepare quarterly review for client C-204
• Compare equity and fixed income outlook
• Best performing bond funds
• Is Fund XYZ suitable for conservative investors?
• Summarize portfolio risk for C-101
"""
)

# =====================================================
# CLIENT DETECTION
# =====================================================

if query:

    client_id = extract_client_id(query)

    if client_id:

        st.info(
            f"Detected Client ID: {client_id}"
        )

    else:

        st.info(
            "No client ID detected. Using generic wealth-management retrieval."
        )

# =====================================================
# GENERATE RESPONSE
# =====================================================

if st.button("Generate Insights"):

    # =========================================
    # Empty Query Check
    # =========================================

    if not query.strip():

        st.warning(
            "Please enter a request."
        )

    else:

        with st.spinner(
            "Analyzing portfolios, research, products, and compliance..."
        ):

            try:

                # =========================================
                # Invoke Agent
                # =========================================

                result = agent.invoke({
                    "query": query
                })

                response = result.get(
                    "response",
                    "No response generated."
                )

                # =========================================
                # OUTPUT SECTION
                # =========================================

                st.text_area(
                    "AI Wealth Insights",
                    value=response,
                    height=500
                )

            except Exception as e:

                st.error(
                    f"""
                        Application Error

                        {str(e)}
                    """
                )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption("""
RM Copilot provides decision-support assistance for wealth-management workflows.
Responses are generated using uploaded research, product, policy,
and portfolio documents with compliance-aware reasoning.
""")


# ====================================================
# CACHE AGENT
# ====================================================


