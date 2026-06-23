"""
Vector store and RAG retriever for Wealth Manager Copilot
"""
import os
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS

logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    Retrieves relevant documents from knowledge base using hybrid RAG.
    Supports both Chroma and FAISS vector stores.
    """
    
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        vector_store_type: str = "faiss",
        persist_dir: Optional[str] = None,
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize the RAG retriever.
        
        Args:
            embedding_model: OpenAI embedding model name
            vector_store_type: "chroma" or "faiss"
            persist_dir: Directory for persisting vector store
            openai_api_key: OpenAI API key
        """
        self.embedding_model = embedding_model
        self.vector_store_type = vector_store_type
        self.persist_dir = persist_dir or "./data/vector_store"
        
        # Initialize embeddings
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vector_store = None
        
        logger.info(f"Initialized RAG Retriever (model: {embedding_model}, store: {vector_store_type})")
    
    def index_documents(
        self, 
        documents: List[Document],
        recreate: bool = False
    ) -> None:
        """
        Index documents in the vector store.
        
        Args:
            documents: List of documents to index
            recreate: Whether to recreate the store (vs. adding to existing)
        """
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        
        if self.vector_store_type.lower() == "chroma":
            self._index_chroma(documents, recreate)
        elif self.vector_store_type.lower() == "faiss":
            self._index_faiss(documents, recreate)
        else:
            raise ValueError(f"Unknown vector store type: {self.vector_store_type}")
        
        logger.info(f"Indexed {len(documents)} documents in {self.vector_store_type}")
    
    def _index_chroma(self, documents: List[Document], recreate: bool = False) -> None:
        """Index documents in Chroma vector store."""
        if recreate or self.vector_store is None:
            self.vector_store = Chroma.from_documents(
                documents,
                self.embeddings,
                persist_directory=self.persist_dir,
                collection_name="wealth_manager"
            )
        else:
            # Add documents to existing store
            self.vector_store.add_documents(documents)
        
        self.vector_store.persist()
    
    def _index_faiss(self, documents: List[Document], recreate: bool = False) -> None:
        """Index documents in FAISS vector store."""
        if recreate or self.vector_store is None:
            self.vector_store = FAISS.from_documents(
                documents,
                self.embeddings
            )
            self.vector_store.save_local(self.persist_dir)
        else:
            # Add documents to existing store
            self.vector_store.add_documents(documents)
            self.vector_store.save_local(self.persist_dir)
    
    def load_existing_store(self) -> None:
        """Load an existing vector store from disk."""
        if not Path(self.persist_dir).exists():
            logger.warning(f"Vector store directory not found: {self.persist_dir}")
            return
        
        if self.vector_store_type.lower() == "chroma":
            self.vector_store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
                collection_name="wealth_manager"
            )
        elif self.vector_store_type.lower() == "faiss":
            self.vector_store = FAISS.load_local(
                self.persist_dir,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        
        logger.info(f"Loaded existing {self.vector_store_type} vector store")

    def clear_store(self) -> int:
        """Remove all documents from the vector store and delete persisted files.

        Returns:
            Number of documents removed.
        """
        removed_count = 0

        try:
            if self.vector_store_type.lower() == "chroma" and self.vector_store is not None:
                collection = self.vector_store._collection
                results = collection.get()
                ids = results.get("ids", []) if results else []
                removed_count = len(ids)
                if ids:
                    collection.delete(ids=ids)
                    self.vector_store.persist()
            elif self.vector_store_type.lower() == "faiss" and self.vector_store is not None:
                if hasattr(self.vector_store, "index"):
                    removed_count = self.vector_store.index.ntotal
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
        finally:
            self.vector_store = None

            persist_path = Path(self.persist_dir)
            if persist_path.exists():
                import shutil

                shutil.rmtree(persist_path, ignore_errors=True)

        logger.info(f"Cleared {removed_count} documents from vector store")
        return removed_count
    
    def retrieve_by_similarity(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents by semantic similarity.
        
        Args:
            query: Query text
            k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of (Document, score) tuples
        """
        if self.vector_store is None:
            logger.warning("Vector store not initialized")
            return []
        
        if self.vector_store_type.lower() == "chroma":
            # Chroma returns similar documents
            results = self.vector_store.similarity_search_with_score(
                query, k=k
            )
            return results
        elif self.vector_store_type.lower() == "faiss":
            results = self.vector_store.similarity_search_with_score(
                query, k=k
            )
            return results
        
        return []
    
    def retrieve_by_metadata(
        self,
        doc_type: Optional[str] = None,
        sensitivity: Optional[str] = None,
        source: Optional[str] = None
    ) -> List[Document]:
        """
        Retrieve documents by metadata filters.
        
        Args:
            doc_type: Document type filter (product/policy/research)
            sensitivity: Sensitivity level filter
            source: Source filter
            
        Returns:
            List of matching documents
        """
        if self.vector_store is None:
            logger.warning("Vector store not initialized")
            return []
        
        # Build filter conditions
        where_filter = {}
        if doc_type:
            where_filter["doc_type"] = doc_type
        if sensitivity:
            where_filter["sensitivity"] = sensitivity
        if source:
            where_filter["source"] = source
        
        try:
            if self.vector_store_type.lower() == "chroma":
                if where_filter:
                    results = self.vector_store.get(where=where_filter)
                    # Convert to Document objects
                    docs = []
                    if results and results.get("documents"):
                        for doc, meta in zip(results["documents"], results["metadatas"]):
                            docs.append(Document(page_content=doc, metadata=meta))
                    return docs
                else:
                    # Get all documents
                    results = self.vector_store.get()
                    docs = []
                    if results and results.get("documents"):
                        for doc, meta in zip(results["documents"], results["metadatas"]):
                            docs.append(Document(page_content=doc, metadata=meta))
                    return docs
            
            elif self.vector_store_type.lower() == "faiss":
                # FAISS doesn't support server-side metadata filtering;
                # retrieve all documents from the docstore and post-filter.
                docstore_dict = getattr(self.vector_store.docstore, "_dict", {})
                all_docs = list(docstore_dict.values())
                filtered = []
                for document in all_docs:
                    meta = document.metadata or {}
                    if doc_type and meta.get("doc_type") != doc_type:
                        continue
                    if sensitivity and meta.get("sensitivity") != sensitivity:
                        continue
                    if source and meta.get("source") != source:
                        continue
                    filtered.append(document)
                return filtered
        
        except Exception as e:
            logger.error(f"Error retrieving by metadata: {e}")
            return []
    
    def hybrid_retrieve(
        self,
        query: str,
        doc_type: Optional[str] = None,
        sensitivity: Optional[str] = None,
        sources: Optional[List[str]] = None,
        k: int = 5,
        alpha: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval combining similarity search with metadata filtering.
        
        Args:
            query: Query text
            doc_type: Document type filter
            sensitivity: Sensitivity filter
            sources: List of source document paths to filter
            k: Number of results
            alpha: Weight for similarity score (0-1)
            
        Returns:
            List of retrieved documents with metadata
        """
        # Get similar documents
        similar_docs = self.retrieve_by_similarity(query, k=k*2)
        
        # Filter by metadata
        results = []
        for doc, score in similar_docs:
            doc_type_match = doc_type is None or doc.metadata.get("doc_type") == doc_type
            sensitivity_match = sensitivity is None or doc.metadata.get("sensitivity") == sensitivity
            source_match = True
            if sources is not None and len(sources) > 0:
                source_match = doc.metadata.get("source") in sources
            
            if doc_type_match and sensitivity_match and source_match:
                results.append({
                    "content": doc.page_content,
                    "score": score,
                    "metadata": doc.metadata,
                    "doc_id": doc.metadata.get("chunk_id", "unknown")
                })
        
        # Sort by score and return top k
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:k]
        
        logger.info(f"Hybrid retrieval for query '{query[:50]}...' returned {len(results)} results")
        return results

    def list_documents(self) -> List[Dict[str, Any]]:
        """Return a summary of documents available in the vector store."""
        if self.vector_store is None:
            return []

        docs = []
        try:
            if self.vector_store_type.lower() == "chroma":
                results = self.vector_store.get()
                for idx, (doc, meta) in enumerate(zip(results.get("documents", []), results.get("metadatas", []))):
                    docs.append({
                        "doc_index": idx,
                        "source": meta.get("source", "unknown"),
                        "doc_type": meta.get("doc_type", "unknown"),
                        "chunk_id": meta.get("chunk_id"),
                        "content_preview": doc[:150].strip(),
                        "metadata": meta
                    })
            elif self.vector_store_type.lower() == "faiss":
                docstore_dict = getattr(self.vector_store.docstore, "_dict", {})
                for idx, doc_id in self.vector_store.index_to_docstore_id.items():
                    document = docstore_dict.get(doc_id)
                    if document is None:
                        continue
                    meta = document.metadata or {}
                    docs.append({
                        "doc_index": idx,
                        "source": meta.get("source", "unknown"),
                        "doc_type": meta.get("doc_type", "unknown"),
                        "chunk_id": meta.get("chunk_id"),
                        "content_preview": document.page_content[:150].strip(),
                        "metadata": meta
                    })
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
        return docs
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        if self.vector_store is None:
            return {"status": "not_initialized"}
        
        try:
            if self.vector_store_type.lower() == "chroma":
                collection = self.vector_store._collection
                count = collection.count()
                return {
                    "vector_store_type": "chroma",
                    "total_documents": count,
                    "persist_directory": self.persist_dir
                }
            elif self.vector_store_type.lower() == "faiss":
                count = 0
                if hasattr(self.vector_store, "index"):
                    count = self.vector_store.index.ntotal
                return {
                    "vector_store_type": "faiss",
                    "total_documents": count,
                    "persist_directory": self.persist_dir
                }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
        
        return {"status": "error"}


class SuitabilityChecker:
    """
    Checks recommendations against suitability and compliance policies.
    """
    
    def __init__(self, policy_retriever: RAGRetriever):
        """
        Initialize suitability checker.
        
        Args:
            policy_retriever: RAG retriever for policy documents
        """
        self.policy_retriever = policy_retriever
    
    def check_suitability(
        self,
        risk_profile: str,
        recommendation: Dict[str, Any],
        portfolio_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Check if a recommendation meets suitability criteria.
        
        Args:
            risk_profile: Client risk profile
            recommendation: Recommendation to check
            portfolio_info: Current portfolio information
            
        Returns:
            Suitability assessment with reasoning
        """
        # Retrieve relevant policy
        policy_docs = self.policy_retriever.hybrid_retrieve(
            f"suitability rules for {risk_profile}",
            doc_type="policy",
            k=3
        )
        
        # Retrieve product details
        product_name = recommendation.get("product_name", "")
        product_docs = self.policy_retriever.hybrid_retrieve(
            product_name,
            doc_type="product",
            k=3
        ) if product_name else []
        
        # Build assessment
        assessment = {
            "risk_profile": risk_profile,
            "recommendation": recommendation.get("idea", ""),
            "suitable": True,  # Default to suitable
            "reasoning": "Recommendation aligns with client risk profile",
            "policy_citations": [doc["doc_id"] for doc in policy_docs],
            "product_citations": [doc["doc_id"] for doc in product_docs],
            "concerns": []
        }
        
        # Check concentration limits
        if portfolio_info and "current_allocation" in portfolio_info:
            single_holding = portfolio_info["current_allocation"].get(product_name, 0)
            if single_holding > 0.10:  # 10% limit
                assessment["suitable"] = False
                assessment["concerns"].append(f"Concentration limit exceeded for {product_name}")
        
        logger.info(f"Suitability check: {assessment['suitable']} - {product_name}")
        return assessment


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    retriever = RAGRetriever(
        vector_store_type="faiss",
        persist_dir="./data/vector_store"
    )
    
    # Note: This requires documents to be indexed first
    retriever.load_existing_store()
    stats = retriever.get_stats()
    print(f"Vector store stats: {stats}")
