"""
Knowledge ingestion pipeline for Wealth Manager Copilot
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
)
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)


class KnowledgeIngestionPipeline:
    """
    Handles loading, preprocessing, and chunking of knowledge documents
    (product guides, policies, research notes, compliance rules)
    """
    
    def __init__(
        self, 
        chunk_size: int = 500, 
        chunk_overlap: int = 100,
        min_chunk_size: int = 50
    ):
        """
        Initialize the ingestion pipeline.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            min_chunk_size: Minimum chunk size to keep
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len,
        )
    
    def load_documents(self, data_dir: str) -> Dict[str, List[Document]]:
        """
        Load documents from data directory.
        
        Supports: PDF, DOCX, TXT, CSV
        
        Args:
            data_dir: Directory containing knowledge documents
            
        Returns:
            Dict with document type as key and list of Documents as value
        """
        documents = {}
        data_path = Path(data_dir)
        
        if not data_path.exists():
            logger.warning(f"Data directory {data_dir} does not exist")
            return documents
        
        # Load PDF files
        pdf_docs = []
        for pdf_file in data_path.glob("*.pdf"):
            try:
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = str(pdf_file)
                    doc.metadata["doc_type"] = "product"  # Default, can be overridden
                pdf_docs.extend(docs)
                logger.info(f"Loaded PDF: {pdf_file.name} ({len(docs)} pages)")
            except Exception as e:
                logger.error(f"Error loading PDF {pdf_file}: {e}")
        
        if pdf_docs:
            documents["pdf"] = pdf_docs
        
        # Load DOCX files
        docx_docs = []
        for docx_file in data_path.glob("*.docx"):
            try:
                loader = UnstructuredWordDocumentLoader(str(docx_file))
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = str(docx_file)
                    doc.metadata["doc_type"] = "product"
                docx_docs.extend(docs)
                logger.info(f"Loaded DOCX: {docx_file.name}")
            except Exception as e:
                logger.error(f"Error loading DOCX {docx_file}: {e}")
        
        if docx_docs:
            documents["docx"] = docx_docs
        
        # Load TXT files
        txt_docs = []
        for txt_file in data_path.glob("*.txt"):
            try:
                loader = TextLoader(str(txt_file), encoding='utf-8')
                doc = loader.load()
                for d in doc:
                    d.metadata["source"] = str(txt_file)
                    d.metadata["doc_type"] = "policy"  # Default for TXT
                txt_docs.extend(doc)
                logger.info(f"Loaded TXT: {txt_file.name}")
            except Exception as e:
                logger.error(f"Error loading TXT {txt_file}: {e}")
        
        if txt_docs:
            documents["txt"] = txt_docs
        
        # Load CSV files
        csv_docs = []
        for csv_file in data_path.glob("*.csv"):
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(csv_file),
                        "doc_type": "research"
                    }
                )
                csv_docs.append(doc)
                logger.info(f"Loaded CSV: {csv_file.name}")
            except Exception as e:
                logger.error(f"Error loading CSV {csv_file}: {e}")
        
        if csv_docs:
            documents["csv"] = csv_docs
        
        return documents
    
    def preprocess_documents(
        self, 
        documents: Dict[str, List[Document]],
        doc_type_mapping: Optional[Dict[str, str]] = None
    ) -> List[Document]:
        """
        Preprocess and combine documents.
        
        Args:
            documents: Dict of documents by type
            doc_type_mapping: Override doc_type for specific file patterns
            
        Returns:
            List of preprocessed documents
        """
        all_docs = []
        
        for doc_type, docs in documents.items():
            for doc in docs:
                # Clean content
                content = doc.page_content.strip()
                if len(content) < self.min_chunk_size:
                    continue
                
                # Enhance metadata
                if "date" not in doc.metadata:
                    doc.metadata["date"] = datetime.utcnow().isoformat()
                
                if "sensitivity" not in doc.metadata:
                    doc.metadata["sensitivity"] = "public"
                
                all_docs.append(doc)
        
        logger.info(f"Preprocessed {len(all_docs)} documents")
        return all_docs
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks with preserved metadata.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        chunked_docs = []
        
        for i, doc in enumerate(documents):
            chunks = self.text_splitter.split_text(doc.page_content)
            
            for j, chunk in enumerate(chunks):
                if len(chunk) < self.min_chunk_size:
                    continue
                
                chunk_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_id": f"{i}_{j}",
                        "original_doc_index": i,
                    }
                )
                chunked_docs.append(chunk_doc)
        
        logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
        return chunked_docs
    
    def run_ingestion_pipeline(
        self, 
        data_dir: str,
        doc_type_mapping: Optional[Dict[str, str]] = None
    ) -> List[Document]:
        """
        Run the complete ingestion pipeline.
        
        Step 1: Load documents
        Step 2: Preprocess and filter
        Step 3: Chunk for embedding
        
        Args:
            data_dir: Directory containing knowledge documents
            doc_type_mapping: Optional mapping to override doc types
            
        Returns:
            List of processed, chunked documents ready for embedding
        """
        logger.info("Starting knowledge ingestion pipeline...")
        
        # Step 1: Load
        documents = self.load_documents(data_dir)
        total_loaded = sum(len(docs) for docs in documents.values())
        logger.info(f"Step 1 complete: Loaded {total_loaded} documents")
        
        # Step 2: Preprocess
        preprocessed = self.preprocess_documents(documents, doc_type_mapping)
        logger.info(f"Step 2 complete: Preprocessed {len(preprocessed)} documents")
        
        # Step 3: Chunk
        chunked = self.chunk_documents(preprocessed)
        logger.info(f"Step 3 complete: Created {len(chunked)} chunks")
        
        logger.info("Knowledge ingestion pipeline completed successfully")
        return chunked


def create_sample_knowledge_documents(output_dir: str) -> None:
    """
    Create sample knowledge documents for demonstration.
    
    Args:
        output_dir: Directory to save sample documents
    """
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Sample product guide
    product_guide = """
PRODUCT GUIDE: Growth Equity Fund

Overview:
The Growth Equity Fund is designed for investors seeking long-term capital appreciation 
through a diversified portfolio of large-cap and mid-cap equities.

Investment Objectives:
- Achieve above-market returns over 5+ year periods
- Maintain diversification across sectors
- Minimize downside risk through quality screening

Target Allocation:
- Large-Cap Equities: 60%
- Mid-Cap Equities: 30%
- Cash/Defensive: 10%

Risk Profile: Growth (Higher volatility, suited for aggressive investors)
Minimum Investment: $10,000
Management Fee: 0.75% annually

Suitable For:
- Aggressive to Growth risk profiles
- Long-term investment horizons (5+ years)
- Investors comfortable with market volatility

Not Suitable For:
- Conservative investors
- Short-term liquidity needs
- Investors with low risk tolerance
"""
    
    with open(os.path.join(output_dir, "product_guide.txt"), "w") as f:
        f.write(product_guide)
    
    # Sample compliance policy
    compliance_policy = """
COMPLIANCE AND SUITABILITY POLICY

1. SUITABILITY RULES
1.1 Conservative Risk Profile
- Maximum equity allocation: 40%
- Minimum bond allocation: 50%
- Maximum volatility (std dev): 8% annually
- Restricted products: Emerging Markets, Derivatives

1.2 Balanced Risk Profile
- Equity allocation: 50-60%
- Bond allocation: 30-40%
- Maximum volatility: 12% annually
- Allowed products: All standard funds

1.3 Growth Risk Profile
- Equity allocation: 70-80%
- Bond allocation: 10-20%
- Maximum volatility: 15% annually
- Allowed products: All including sector-specific

1.4 Aggressive Risk Profile
- Equity allocation: 85-100%
- Bond allocation: 0-15%
- No volatility limits
- All products allowed

2. COMPLIANCE GATES
2.1 Portfolio Concentration
- Single holding: Maximum 10% of portfolio
- Single sector: Maximum 25% of portfolio

2.2 Liquidity Requirements
- Maintain minimum 5% in liquid assets for conservative profiles
- Maintain minimum 2% in liquid assets for aggressive profiles

2.3 Rebalancing Policy
- Quarterly rebalancing for conservative to balanced
- Bi-annual for growth to aggressive

3. ESCALATION RULES
- Recommendations violating suitability rules require manager review
- Deviations from policy require compliance sign-off
- Licensed advice required for complex products
"""
    
    with open(os.path.join(output_dir, "compliance_policy.txt"), "w") as f:
        f.write(compliance_policy)
    
    # Sample research
    research = """
MARKET RESEARCH: Q2 2024 Economic Outlook

Executive Summary:
Economic growth remains moderate with inflation gradually cooling. 
Key themes: interest rates, technology sector valuations, geopolitical risks.

Key Points:
1. GDP Growth: Expected 2-2.5% annually
2. Inflation: Trending toward 2.5-3% target
3. Employment: Strong labor market, unemployment near 4%
4. Fed Policy: Rate cuts likely in H2 2024

Sector Recommendations:
- Technology: HOLD (valuations elevated, but growth strong)
- Healthcare: OVERWEIGHT (defensive, stable dividends)
- Financials: BUY (benefiting from higher rates)
- Energy: HOLD (volatile, geopolitical exposure)
- Consumer Discretionary: NEUTRAL (sensitive to economic slowdown)

Investment Implications:
- Diversification remains critical
- Quality over quantity bias warranted
- International exposure provides diversification
- Fixed income allocation justified by yields

Risk Factors:
- Geopolitical tensions
- Inflation re-acceleration
- Credit market stress
- Technology concentration risk
"""
    
    with open(os.path.join(output_dir, "market_research.txt"), "w") as f:
        f.write(research)
    
    logger.info(f"Created sample knowledge documents in {output_dir}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample documents
    sample_dir = "./data/sample_knowledge"
    create_sample_knowledge_documents(sample_dir)
    
    # Run pipeline
    pipeline = KnowledgeIngestionPipeline()
    processed_docs = pipeline.run_ingestion_pipeline(sample_dir)
    
    print(f"\nProcessed {len(processed_docs)} chunks ready for embedding")
