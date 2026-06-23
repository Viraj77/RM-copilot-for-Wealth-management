"""
Test script to validate Wealth Manager Copilot setup
"""
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        from src.models import ClientBrief, Recommendation, PortfolioSummary
        logger.info("✓ Models imported")
        
        from src.ingestion import KnowledgeIngestionPipeline
        logger.info("✓ Ingestion pipeline imported")
        
        from src.retriever import RAGRetriever, SuitabilityChecker
        logger.info("✓ Retriever imported")
        
        from src.tools import create_tools_dict
        logger.info("✓ Tools imported")
        
        from src.agent import create_langgraph_agent
        logger.info("✓ Agent imported")
        
        return True
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration():
    """Test configuration."""
    logger.info("Testing configuration...")
    
    try:
        from config import settings
        settings.validate_config()
        logger.info(f"✓ Configuration valid")
        logger.info(f"  - Vector store: {settings.vector_store_type}")
        logger.info(f"  - Model: {settings.openai_model}")
        return True
    except Exception as e:
        logger.error(f"✗ Configuration failed: {e}")
        return False


def test_tools():
    """Test agent tools."""
    logger.info("Testing tools...")
    
    try:
        from src.tools import (
            PortfolioLookupTool,
            MarketDataTool,
            SuitabilityCheckerTool,
            ComplianceGateTool
        )
        
        # Test portfolio lookup
        portfolio_tool = PortfolioLookupTool()
        result = portfolio_tool("C-204")
        assert result["success"], "Portfolio lookup failed"
        logger.info("✓ Portfolio lookup works")
        
        # Test market data
        market_tool = MarketDataTool()
        result = market_tool()
        assert result["success"], "Market data lookup failed"
        logger.info("✓ Market data tool works")
        
        # Test suitability checker
        suitability_tool = SuitabilityCheckerTool()
        result = suitability_tool(
            "Conservative",
            {"idea": "Test", "product_type": "equity"}
        )
        assert "suitable" in result, "Suitability check failed"
        logger.info("✓ Suitability checker works")
        
        # Test compliance gate
        compliance_tool = ComplianceGateTool()
        result = compliance_tool(
            {"idea": "Test"},
            {"suitable": True}
        )
        assert "gate_status" in result, "Compliance gate failed"
        logger.info("✓ Compliance gate works")
        
        return True
    except Exception as e:
        logger.error(f"✗ Tools test failed: {e}")
        return False


def test_models():
    """Test Pydantic models."""
    logger.info("Testing data models...")
    
    try:
        from src.models import (
            ClientBrief, Recommendation, PortfolioSummary,
            Citation, SuitabilityAssessment, RiskProfile, ComplianceStatus
        )
        from datetime import datetime
        
        # Create citation
        citation = Citation(
            doc_id="doc-1",
            doc_type="product",
            chunk_text="Sample text",
            source="sample.pdf"
        )
        logger.info("✓ Citation model works")
        
        # Create suitability assessment
        assessment = SuitabilityAssessment(
            suitable_for_profile=True,
            reasoning="Meets criteria",
            compliance_check_passed=True
        )
        logger.info("✓ SuitabilityAssessment model works")
        
        # Create recommendation
        rec = Recommendation(
            idea="Buy SPY",
            rationale="Diversified exposure",
            suitability=assessment,
            citations=[citation],
            confidence_score=0.85
        )
        logger.info("✓ Recommendation model works")
        
        # Create portfolio summary
        portfolio = PortfolioSummary(
            client_id="C-204",
            total_value=600000,
            allocation={"equities": 0.7, "bonds": 0.3},
            holdings=[],
            risk_score=7.0
        )
        logger.info("✓ PortfolioSummary model works")
        
        # Create brief
        brief = ClientBrief(
            client_id="C-204",
            brief_id="brief-1",
            risk_profile=RiskProfile.GROWTH,
            portfolio_summary=portfolio,
            recommendations=[rec],
            talking_points=["Point 1", "Point 2"],
            compliance_status=ComplianceStatus.CLEARED
        )
        logger.info("✓ ClientBrief model works")
        
        # Test serialization
        brief_json = brief.model_dump_json()
        assert len(brief_json) > 0, "Serialization failed"
        logger.info("✓ Model serialization works")
        
        return True
    except Exception as e:
        logger.error(f"✗ Models test failed: {e}")
        return False


def test_knowledge_pipeline():
    """Test knowledge ingestion."""
    logger.info("Testing knowledge ingestion pipeline...")
    
    try:
        from src.ingestion import KnowledgeIngestionPipeline
        
        pipeline = KnowledgeIngestionPipeline()
        logger.info("✓ Pipeline initialized")
        
        # Test chunking
        from langchain_core.documents import Document
        test_docs = [
            Document(
                page_content="Sample document content " * 50,
                metadata={"source": "test.txt"}
            )
        ]
        
        chunked = pipeline.chunk_documents(test_docs)
        assert len(chunked) > 0, "Chunking failed"
        logger.info(f"✓ Chunking works ({len(chunked)} chunks created)")
        
        return True
    except Exception as e:
        logger.error(f"✗ Pipeline test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("WEALTH MANAGER COPILOT - VALIDATION TESTS")
    print("="*60 + "\n")
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Data Models", test_models),
        ("Tools", test_tools),
        ("Ingestion Pipeline", test_knowledge_pipeline),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! System is ready to use.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please fix issues before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
