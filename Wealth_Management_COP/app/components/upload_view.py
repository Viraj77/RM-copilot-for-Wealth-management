import os
import tempfile
from pathlib import Path
import streamlit as st
from config.settings import settings
from src.ingestion.indexer import ingest_single_file

# Map nice UI names to the DocType enum values and target subdirectories
CATEGORY_MAP = {
    "Product Guide": {"type": "product", "folder": "product_guides"},
    "Compliance Policy": {"type": "policy", "folder": "compliance_policies"},
    "Research Note": {"type": "research", "folder": "research_notes"},
}

def render_upload_view():
    st.title("📂 Knowledge Base Upload")
    st.markdown("Upload new documents to expand the Copilot's knowledge base. Files are automatically processed, chunked, and indexed for immediate retrieval.")

    with st.container():
        st.markdown("### Upload Document")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Select a file (PDF, DOCX, TXT, MD)",
            type=["pdf", "docx", "txt", "md"],
            help="Maximum file size is 200MB."
        )

        # Document category selection
        doc_category = st.selectbox(
            "Document Category",
            options=list(CATEGORY_MAP.keys()),
            help="Select the category that best describes this document."
        )

        if st.button("Upload & Index", type="primary"):
            if uploaded_file is None:
                st.error("Please select a file to upload.")
            else:
                with st.spinner(f"Uploading and indexing '{uploaded_file.name}'..."):
                    try:
                        # Determine target path
                        target_info = CATEGORY_MAP[doc_category]
                        doc_type = target_info["type"]
                        target_folder = target_info["folder"]
                        
                        target_dir = Path(settings.raw_docs_dir) / target_folder
                        target_dir.mkdir(parents=True, exist_ok=True)
                        
                        target_path = target_dir / uploaded_file.name
                        
                        # Save the file
                        with open(target_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                            
                        # Run ingestion for this single file
                        ingest_single_file(target_path, doc_type)
                        
                        st.success(f"✅ Successfully uploaded and indexed '{uploaded_file.name}' as a {doc_category}!")
                        
                    except Exception as e:
                        st.error(f"Failed to process document: {str(e)}")
                        # Optionally remove the file if indexing failed
                        if target_path.exists():
                            try:
                                target_path.unlink()
                            except:
                                pass
