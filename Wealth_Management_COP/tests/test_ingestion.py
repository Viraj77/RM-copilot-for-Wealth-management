"""
Unit tests for the Ingestion Pipeline — Phase 2.
Tests loader, chunker, and indexer (ChromaDB) with local doc fixtures.

Run: pytest tests/test_ingestion.py -v
"""

import json
from pathlib import Path

import pytest

from src.ingestion.chunker import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    _extract_section_heading,
    _inject_context_prefix,
    _simple_split,
    chunk_document,
    chunk_documents,
)
from src.ingestion.loader import (
    extract_date_from_filename,
    infer_sensitivity,
    load_client_data,
    load_documents,
    load_file,
    load_market_data,
    make_doc_id,
)
from src.models.documents import DocType, RawDocument, Sensitivity


# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
CLIENTS_FILE = DATA_DIR / "clients" / "clients.json"
MARKET_FILE = DATA_DIR / "raw" / "market_data.json"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_raw_doc():
    return RawDocument(
        doc_id="test_policy",
        doc_type=DocType.POLICY,
        source="test_policy.md",
        content=(
            "# Section One\n\n"
            "This is the first section with some important compliance text. "
            "It covers the conservative portfolio policy in detail.\n\n"
            "## Section Two\n\n"
            "This section discusses the aggressive risk suitability rules. "
            "RMs should refer to this when working with growth and aggressive clients.\n\n"
            "### Subsection 2.1\n\n"
            "Detailed rules for asset allocation go here. "
            "Maximum equity allocation for balanced clients is 65%."
        ),
        sensitivity=Sensitivity.INTERNAL,
    )


@pytest.fixture
def equity_guide_path():
    return RAW_DIR / "product_guides" / "equity_fund_product_guide.md"


# ── Loader Tests ──────────────────────────────────────────────────────────────

class TestLoader:
    def test_load_markdown_file(self, equity_guide_path):
        doc = load_file(equity_guide_path, DocType.PRODUCT)
        assert doc.doc_id == "equity_fund_product_guide"
        assert doc.doc_type == DocType.PRODUCT
        assert len(doc.content) > 100
        assert doc.sensitivity == Sensitivity.PUBLIC

    def test_load_documents_from_raw_dir(self):
        docs = load_documents(RAW_DIR)
        assert len(docs) >= 10, f"Expected ≥10 documents, got {len(docs)}"

    def test_doc_types_assigned_correctly(self):
        docs = load_documents(RAW_DIR)
        types = {d.doc_type for d in docs}
        assert DocType.PRODUCT in types
        assert DocType.POLICY in types
        assert DocType.RESEARCH in types

    def test_sensitivity_inferred(self):
        docs = load_documents(RAW_DIR)
        doc_map = {d.doc_id: d for d in docs}

        # Equity guide should be PUBLIC
        eq_doc = doc_map.get("equity_fund_product_guide")
        assert eq_doc is not None
        assert eq_doc.sensitivity == Sensitivity.PUBLIC

        # Emerging markets should be RESTRICTED
        em_doc = doc_map.get("emerging_markets_deep_dive")
        assert em_doc is not None
        assert em_doc.sensitivity == Sensitivity.RESTRICTED

        # Conservative policy should be INTERNAL
        pol_doc = doc_map.get("conservative_portfolio_policy")
        assert pol_doc is not None
        assert pol_doc.sensitivity == Sensitivity.INTERNAL

    def test_unsupported_extension_raises(self, tmp_path):
        fake_file = tmp_path / "test.xlsx"
        fake_file.write_text("test")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_file(fake_file, DocType.POLICY)

    def test_date_extraction_from_filename(self):
        assert extract_date_from_filename("q2_2026_equity_outlook") is not None
        assert extract_date_from_filename("report_2026_01_15") is not None
        assert extract_date_from_filename("no_date_here") is None

    def test_make_doc_id_normalisation(self):
        from pathlib import Path
        p = Path("My Policy Document-v2.md")
        assert make_doc_id(p) == "my_policy_document_v2"

    def test_load_client_data(self):
        data = load_client_data(CLIENTS_FILE)
        assert "clients" in data
        assert len(data["clients"]) >= 5

    def test_load_market_data(self):
        data = load_market_data(MARKET_FILE)
        assert "market_data" in data
        assert "VTI" in data["market_data"]

    def test_infer_sensitivity_rules(self):
        assert infer_sensitivity(DocType.RESEARCH, "emerging_markets_deep.md") == Sensitivity.RESTRICTED
        assert infer_sensitivity(DocType.RESEARCH, "crypto_assets.md") == Sensitivity.RESTRICTED
        assert infer_sensitivity(DocType.PRODUCT, "equity_fund_product_guide.md") == Sensitivity.PUBLIC
        assert infer_sensitivity(DocType.POLICY, "general_compliance.md") == Sensitivity.INTERNAL


# ── Chunker Tests ─────────────────────────────────────────────────────────────

class TestChunker:
    def test_chunk_count_reasonable(self, sample_raw_doc):
        chunks = chunk_document(sample_raw_doc)
        # Small doc should produce a small number of chunks
        assert 1 <= len(chunks) <= 5

    def test_chunk_ids_unique(self, sample_raw_doc):
        chunks = chunk_document(sample_raw_doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_chunk_ids_formatted(self, sample_raw_doc):
        chunks = chunk_document(sample_raw_doc)
        for chunk in chunks:
            assert chunk.chunk_id.startswith(sample_raw_doc.doc_id)

    def test_metadata_inherited(self, sample_raw_doc):
        chunks = chunk_document(sample_raw_doc)
        for chunk in chunks:
            assert chunk.doc_id == sample_raw_doc.doc_id
            assert chunk.doc_type == sample_raw_doc.doc_type
            assert chunk.sensitivity == sample_raw_doc.sensitivity
            assert chunk.source == sample_raw_doc.source

    def test_section_heading_extracted(self):
        text = "## Conservative Policy\n\nThis is the content."
        heading = _extract_section_heading(text)
        assert heading == "Conservative Policy"

    def test_section_heading_none_for_plain_text(self):
        text = "This is plain text without a heading."
        heading = _extract_section_heading(text)
        assert heading is None

    def test_context_prefix_injected(self, sample_raw_doc):
        text = "Some policy text without heading."
        result = _inject_context_prefix(text, sample_raw_doc)
        assert "POLICY" in result
        assert sample_raw_doc.source in result

    def test_context_prefix_not_added_to_headings(self, sample_raw_doc):
        text = "# Document Title\n\nContent here."
        result = _inject_context_prefix(text, sample_raw_doc)
        # Should NOT prepend header since text starts with heading
        assert result.startswith("#")

    def test_chunk_all_documents(self):
        docs = load_documents(RAW_DIR)
        chunks = chunk_documents(docs)
        assert len(chunks) > 50, f"Expected many chunks from 12 docs, got {len(chunks)}"

    def test_no_empty_chunks(self, sample_raw_doc):
        chunks = chunk_document(sample_raw_doc)
        for chunk in chunks:
            assert chunk.content.strip(), "Chunks must not be empty"

    def test_simple_split_fallback(self):
        long_text = "Word " * 500  # 2500 chars
        chunks = _simple_split(long_text, chunk_size=800, chunk_overlap=200)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 800 + 50  # Some tolerance

    def test_chroma_metadata_contains_required_keys(self, sample_raw_doc):
        chunks = chunk_document(sample_raw_doc)
        for chunk in chunks:
            meta = chunk.to_chroma_metadata()
            for key in ["doc_id", "chunk_id", "doc_type", "source", "sensitivity"]:
                assert key in meta, f"Missing key '{key}' in chroma metadata"
            # All values must be strings
            for k, v in meta.items():
                assert isinstance(v, str), f"Metadata value for '{k}' must be str, got {type(v)}"
