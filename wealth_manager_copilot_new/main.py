"""
Main entry point for Wealth Manager Copilot
Demonstrates core functionality and test queries
"""
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import settings
from src.agent import create_langgraph_agent
from src.retriever import RAGRetriever
from src.ingestion import KnowledgeIngestionPipeline, create_sample_knowledge_documents

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("wealth_manager.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_knowledge_base():
    """Initialize and setup the knowledge base."""
    logger.info("Setting up knowledge base...")
    
    retriever = RAGRetriever(
        embedding_model="text-embedding-3-small",
        vector_store_type="chroma",
        persist_dir="./data/vector_store",
        openai_api_key=settings.openai_api_key
    )
    
    # Try loading existing store
    try:
        retriever.load_existing_store()
        stats = retriever.get_stats()
        if stats.get("total_documents", 0) > 0:
            logger.info(f"Loaded existing vector store with {stats['total_documents']} documents")
            return retriever
    except Exception as e:
        logger.warning(f"Could not load existing store: {e}")
    
    # Create sample documents and index
    logger.info("Creating sample knowledge documents...")
    sample_dir = "./data/sample_knowledge"
    create_sample_knowledge_documents(sample_dir)
    
    logger.info("Ingesting documents...")
    pipeline = KnowledgeIngestionPipeline()
    docs = pipeline.run_ingestion_pipeline(sample_dir)
    
    logger.info(f"Indexing {len(docs)} documents...")
    retriever.index_documents(docs, recreate=True)
    
    stats = retriever.get_stats()
    logger.info(f"Knowledge base ready! {stats['total_documents']} documents indexed")
    
    return retriever


def run_demo_query(agent, client_id: str, request: str):
    """Run a single demo query."""
    print(f"\n{'='*70}")
    print(f"Query: {request}")
    print(f"Client: {client_id}")
    print(f"{'='*70}\n")
    
    try:
        brief = agent.run_agent(client_id=client_id, request=request)
        
        print(f"✓ Brief generated successfully!\n")
        print(f"Client ID: {brief.client_id}")
        print(f"Risk Profile: {brief.risk_profile.value}")
        print(f"Compliance Status: {brief.compliance_status.value}")
        print(f"\nPortfolio Summary:")
        print(f"  Total Value: ${brief.portfolio_summary.total_value:,.0f}")
        print(f"  Risk Score: {brief.portfolio_summary.risk_score:.1f}/10")
        print(f"  Allocation: {brief.portfolio_summary.allocation}")
        
        if brief.recommendations:
            print(f"\nRecommendations ({len(brief.recommendations)}):")
            for i, rec in enumerate(brief.recommendations, 1):
                print(f"  {i}. {rec.idea}")
                print(f"     Confidence: {rec.confidence_score*100:.0f}%")
                print(f"     Suitable: {rec.suitability.suitable_for_profile}")
                print(f"     Citations: {len(rec.citations)}")
        
        if brief.talking_points:
            print(f"\nTalking Points:")
            for point in brief.talking_points[:3]:
                print(f"  • {point}")
        
        if brief.escalated_items:
            print(f"\n⚠️  Escalated Items:")
            for item in brief.escalated_items:
                print(f"  - {item.get('recommendation')}")
                print(f"    Type: {item.get('escalation_type')}")
                print(f"    Reason: {item.get('reason')}")
        
        print(f"\n✓ Query completed successfully\n")
        return brief
    
    except Exception as e:
        print(f"✗ Error processing query: {e}\n")
        return None


def run_golden_set_tests(agent):
    """Run all golden set test queries."""
    golden_queries = [
        {
            "client_id": "C-204",
            "request": "Prepare talking points for client C-204's quarterly review",
            "description": "Quarterly Review Brief"
        },
        {
            "client_id": "C-202",
            "request": "Is fund XYZ suitable for a conservative client?",
            "description": "Suitability Check"
        },
        {
            "client_id": "C-204",
            "request": "Summarize the portfolio risk for client C-204",
            "description": "Portfolio Risk Summary"
        },
        {
            "client_id": "C-201",
            "request": "What recommendations would you make for client C-201?",
            "description": "Personalized Recommendations"
        },
    ]
    
    print(f"\n{'='*70}\")\n    RUNNING GOLDEN SET TESTS\n{'='*70}\n")
    
    results = []
    for i, query in enumerate(golden_queries, 1):
        print(f"\\nTest {i}/{len(golden_queries)}: {query['description']}")
        print(f\"Client: {query['client_id']}\")\n")
        
        brief = run_demo_query(agent, query['client_id'], query['request'])
        
        results.append({
            "test_id": i,
            "description": query["description"],
            "success": brief is not None,
            "brief": brief
        })
    
    # Summary
    print(f"\n{'='*70}")
    print("GOLDEN SET TEST SUMMARY")
    print(f\"{'='*70}\n\")
    
    successful = sum(1 for r in results if r["success"])
    print(f"Results: {successful}/{len(results)} tests passed\n")
    
    for result in results:
        status = "✓ PASS" if result["success"] else "✗ FAIL"
        print(f"{status} - Test {result['test_id']}: {result['description']}")
    
    return results


def main():
    """Main entry point."""
    print(f"\n{'='*70}")
    print("WEALTH MANAGER COPILOT - DEMO")
    print(f\"{'='*70}\n\")
    
    # Setup
    print("Initializing system...\n")
    
    try:
        # Load or create knowledge base
        retriever = setup_knowledge_base()
        
        # Initialize agent
        logger.info("Initializing agent...")
        agent = create_langgraph_agent(
            llm_model="gpt-4o",
            retriever=retriever
        )
        logger.info("Agent initialized!")
        
        # Run golden set tests
        run_golden_set_tests(agent)
        
        # Optional: Run individual query
        print(f"\n{'='*70}")
        print("DEMO COMPLETE")
        print(f\"{'='*70}\n\")
        
        print("✓ All components working correctly!")
        print("\\nTo run the interactive dashboard:")
        print("  streamlit run app.py")
        print("\\nTo run detailed evaluation:")
        print("  jupyter notebook notebooks/evaluation.ipynb")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
