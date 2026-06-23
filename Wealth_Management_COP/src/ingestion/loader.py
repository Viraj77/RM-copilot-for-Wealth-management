"""
Document Loader — Phase 2: Ingestion Pipeline.

Loads PDF, DOCX, TXT, and Markdown files from the raw/ data directories.
Attaches doc_type, sensitivity, source, and date metadata to each document.
"""

import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import Optional

from src.models.documents import DocType, RawDocument, Sensitivity

# ── Supported extensions ──────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}

# ── Subdirectory → DocType mapping ────────────────────────────────────────────

SUBDIR_DOC_TYPE: dict[str, DocType] = {
    "product_guides": DocType.PRODUCT,
    "compliance_policies": DocType.POLICY,
    "research_notes": DocType.RESEARCH,
}

# ── Sensitivity inference rules ───────────────────────────────────────────────

# Filename fragments that mark a document as restricted
RESTRICTED_KEYWORDS = ["restricted", "crypto", "digital_asset", "emerging_markets_deep"]
# Filename fragments that mark a document as public
PUBLIC_KEYWORDS = [
    "equity_fund", "fixed_income_fund", "esg",
    "q2_2026_equity", "product_guide"
]


def infer_sensitivity(doc_type: DocType, filename: str) -> Sensitivity:
    """
    Infer document sensitivity from doc_type and filename.

    Rules (in priority order):
    1. If filename contains a restricted keyword → RESTRICTED
    2. If filename contains a public keyword → PUBLIC
    3. Research notes → INTERNAL by default
    4. Policy → INTERNAL by default
    5. Product → PUBLIC by default
    """
    name_lower = filename.lower()

    # Check restricted first (highest priority)
    if any(kw in name_lower for kw in RESTRICTED_KEYWORDS):
        return Sensitivity.RESTRICTED

    # Check explicit public keywords
    if any(kw in name_lower for kw in PUBLIC_KEYWORDS):
        return Sensitivity.PUBLIC

    # Default by doc_type
    if doc_type == DocType.PRODUCT:
        return Sensitivity.PUBLIC
    return Sensitivity.INTERNAL


def extract_date_from_filename(filename: str) -> Optional[date]:
    """
    Try to extract a date from the filename.
    Supports patterns like: q2_2026, 2026_01_15, 2026-01-15
    """
    # Pattern: YYYY_MM_DD or YYYY-MM-DD
    match = re.search(r"(\d{4})[_-](\d{2})[_-](\d{2})", filename)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            pass

    # Pattern: q1/q2/q3/q4_YYYY
    match = re.search(r"q[1-4][_-](\d{4})", filename.lower())
    if match:
        year = int(match.group(1))
        return date(year, 1, 1)  # Approximate to start of year

    return None


def make_doc_id(filepath: Path) -> str:
    """Generate a stable doc_id from filepath stem."""
    return filepath.stem.replace(" ", "_").replace("-", "_").lower()


# ── Plain text / Markdown loader ─────────────────────────────────────────────

def _load_text_file(filepath: Path) -> str:
    """Load a plain text or markdown file."""
    return filepath.read_text(encoding="utf-8", errors="replace")


# ── PDF loader ────────────────────────────────────────────────────────────────

def _load_pdf_file(filepath: Path) -> str:
    """Load a PDF file using pypdf."""
    try:
        import pypdf  # type: ignore
        reader = pypdf.PdfReader(str(filepath))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)
    except ImportError:
        raise ImportError(
            "pypdf is required to load PDF files. Install it with: pip install pypdf"
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load PDF '{filepath}': {exc}") from exc


# ── DOCX loader ───────────────────────────────────────────────────────────────

def _load_docx_file(filepath: Path) -> str:
    """Load a DOCX file using docx2txt."""
    try:
        import docx2txt  # type: ignore
        return docx2txt.process(str(filepath))
    except ImportError:
        raise ImportError(
            "docx2txt is required to load DOCX files. Install it with: pip install docx2txt"
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load DOCX '{filepath}': {exc}") from exc


# ── Dispatcher ────────────────────────────────────────────────────────────────

_LOADERS = {
    ".txt": _load_text_file,
    ".md": _load_text_file,
    ".pdf": _load_pdf_file,
    ".docx": _load_docx_file,
}


def load_file(filepath: Path, doc_type: DocType) -> RawDocument:
    """
    Load a single file and return a RawDocument with metadata.

    Args:
        filepath: Absolute path to the file.
        doc_type: Document type (product / policy / research).

    Returns:
        RawDocument with content and metadata attached.
    """
    ext = filepath.suffix.lower()
    if ext not in _LOADERS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {list(_LOADERS.keys())}"
        )

    loader = _LOADERS[ext]
    content = loader(filepath)

    if not content.strip():
        raise ValueError(f"Document '{filepath.name}' is empty after loading.")

    return RawDocument(
        doc_id=make_doc_id(filepath),
        doc_type=doc_type,
        source=filepath.name,
        content=content,
        pub_date=extract_date_from_filename(filepath.stem),
        sensitivity=infer_sensitivity(doc_type, filepath.name),
    )


# ── Batch loader ──────────────────────────────────────────────────────────────

def load_documents(raw_docs_dir: str | Path) -> list[RawDocument]:
    """
    Load all supported documents from the raw/ directory structure.

    Expected layout:
        raw_docs_dir/
        ├── product_guides/    → DocType.PRODUCT
        ├── compliance_policies/ → DocType.POLICY
        └── research_notes/    → DocType.RESEARCH

    Args:
        raw_docs_dir: Path to the raw documents directory.

    Returns:
        List of RawDocument objects with metadata.
    """
    raw_path = Path(raw_docs_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw docs directory not found: {raw_path}")

    documents: list[RawDocument] = []
    errors: list[str] = []

    for subdir_name, doc_type in SUBDIR_DOC_TYPE.items():
        subdir = raw_path / subdir_name
        if not subdir.exists():
            print(f"  [WARN] Subdirectory not found, skipping: {subdir}")
            continue

        files = [
            f for f in subdir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        print(f"  Loading {len(files)} {doc_type.value} documents from {subdir_name}/")

        for filepath in sorted(files):
            try:
                doc = load_file(filepath, doc_type)
                documents.append(doc)
                print(f"    [OK] {filepath.name} [{doc.sensitivity.value}] ({len(doc.content)} chars)")
            except Exception as exc:
                errors.append(f"{filepath.name}: {exc}")
                print(f"    [FAIL] {filepath.name}: {exc}")

    if errors:
        print(f"\n  [WARN] {len(errors)} file(s) failed to load:")
        for err in errors:
            print(f"    - {err}")

    print(f"\n  Total: {len(documents)} documents loaded successfully.")
    return documents


# ── Market data loader (JSON) ─────────────────────────────────────────────────

def load_market_data(market_data_path: str | Path) -> dict:
    """Load the simulated market data JSON file."""
    path = Path(market_data_path)
    if not path.exists():
        raise FileNotFoundError(f"Market data file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Client data loader (JSON) ─────────────────────────────────────────────────

def load_client_data(client_data_path: str | Path) -> dict:
    """Load the client portfolio JSON database."""
    path = Path(client_data_path)
    if not path.exists():
        raise FileNotFoundError(f"Client data file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)
