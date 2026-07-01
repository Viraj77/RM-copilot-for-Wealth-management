"""Load documents from the knowledge base directory."""

import csv
import re
from pathlib import Path

from langchain_core.documents import Document

from src.config import DOCUMENTS_DIR
from src.models import DocumentType, Sensitivity

# Folder name → document type (maps to your data/documents/ layout)
DOC_TYPE_MAP = {
    "product_guides": DocumentType.PRODUCT,
    "compliance": DocumentType.POLICY,
    "research": DocumentType.RESEARCH,
}

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".csv"}

DOC_ID_PATTERN = re.compile(r"^([A-Z]{2,4}-\d{3})")


def _parse_field(line: str, labels: tuple[str, ...]) -> str | None:
    """Extract value after 'Label:' from a header line."""
    stripped = line.strip()
    for label in labels:
        prefix = f"{label}:"
        if stripped.lower().startswith(prefix.lower()):
            return stripped.split(":", 1)[1].strip()
    return None


def _sensitivity_from_text(text: str) -> Sensitivity:
    """Infer sensitivity from document header or body."""
    header = text[:500].upper()
    if "RESTRICTED" in header:
        return Sensitivity.RESTRICTED
    if "INTERNAL" in header:
        return Sensitivity.INTERNAL
    return Sensitivity.PUBLIC


def _doc_id_from_filename(filepath: Path) -> str:
    """Extract doc ID prefix from filenames like PG-002_Conservative_Income_Fund.txt."""
    match = DOC_ID_PATTERN.match(filepath.stem)
    return match.group(1) if match else filepath.stem


def _extract_metadata(text: str, filepath: Path, doc_type: DocumentType) -> dict:
    """Extract metadata from document header lines or filename."""
    doc_id = _doc_id_from_filename(filepath)
    source = filepath.name
    date = "2025-01-01"
    sensitivity = Sensitivity.PUBLIC

    for line in text.split("\n")[:20]:
        if parsed := _parse_field(line, ("Document ID", "Product ID", "doc_id")):
            doc_id = parsed
        if parsed := _parse_field(line, ("Date", "Effective Date", "Publication Date")):
            date = parsed
        if parsed := _parse_field(line, ("Classification", "Sensitivity")):
            value = parsed.lower()
            if "restricted" in value:
                sensitivity = Sensitivity.RESTRICTED
            elif "internal" in value:
                sensitivity = Sensitivity.INTERNAL
            else:
                sensitivity = Sensitivity.PUBLIC
        # Legacy markdown-style headers
        if "**Document ID:**" in line or "**Product ID:**" in line:
            doc_id = line.split("**")[-2].strip() if "**" in line else doc_id
        if "**Date:**" in line or "**Effective Date:**" in line:
            date = line.split("**")[-2].strip() if "**" in line else date
        if "**Classification:**" in line:
            classification = line.split("**")[-2].strip().lower()
            if "restricted" in classification:
                sensitivity = Sensitivity.RESTRICTED
            elif "internal" in classification:
                sensitivity = Sensitivity.INTERNAL
            else:
                sensitivity = Sensitivity.PUBLIC

    if sensitivity == Sensitivity.PUBLIC:
        sensitivity = _sensitivity_from_text(text)

    return {
        "doc_id": doc_id,
        "type": doc_type.value,
        "date": date,
        "source": source,
        "sensitivity": sensitivity.value,
        "filepath": str(filepath),
    }


def _read_txt(filepath: Path) -> str:
    return filepath.read_text(encoding="utf-8")


def _read_pdf(filepath: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "PDF support requires pypdf. Install with: pip install pypdf"
        ) from exc

    reader = PdfReader(str(filepath))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def _read_csv(filepath: Path) -> str:
    """Convert CSV rows into readable text for embedding."""
    rows: list[str] = []
    with open(filepath, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            parts = [f"{k}: {v}" for k, v in row.items() if v]
            rows.append(f"Row {i}: " + " | ".join(parts))
    return "\n".join(rows)


def _read_file(filepath: Path) -> str:
    suffix = filepath.suffix.lower()
    if suffix == ".txt":
        return _read_txt(filepath)
    if suffix == ".pdf":
        return _read_pdf(filepath)
    if suffix == ".csv":
        return _read_csv(filepath)
    raise ValueError(f"Unsupported file type: {suffix}")


def _metadata_from_csv(filepath: Path, doc_type: DocumentType) -> dict:
    """Build metadata from the first row of a structured compliance CSV."""
    doc_id = _doc_id_from_filename(filepath)
    date = "2025-01-01"
    sensitivity = Sensitivity.RESTRICTED
    source = filepath.name

    with open(filepath, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        first = next(reader, None)
        if first:
            doc_id = first.get("doc_id", doc_id)
            date = first.get("date", date)
            sens = first.get("sensitivity", "restricted").lower()
            if "public" in sens:
                sensitivity = Sensitivity.PUBLIC
            elif "internal" in sens:
                sensitivity = Sensitivity.INTERNAL
            else:
                sensitivity = Sensitivity.RESTRICTED
            source = first.get("source", source)

    return {
        "doc_id": doc_id,
        "type": doc_type.value,
        "date": date,
        "source": source,
        "sensitivity": sensitivity.value,
        "filepath": str(filepath),
    }


def load_documents(documents_dir: Path | None = None) -> list[Document]:
    """Load TXT, PDF, and CSV documents from product_guides, compliance, and research."""
    base = documents_dir or DOCUMENTS_DIR
    documents: list[Document] = []

    for subfolder, doc_type in DOC_TYPE_MAP.items():
        folder = base / subfolder
        if not folder.exists():
            continue
        for filepath in sorted(folder.iterdir()):
            if filepath.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            text = _read_file(filepath)
            if not text.strip():
                continue
            if filepath.suffix.lower() == ".csv":
                metadata = _metadata_from_csv(filepath, doc_type)
            else:
                metadata = _extract_metadata(text, filepath, doc_type)
            documents.append(Document(page_content=text, metadata=metadata))

    return documents
