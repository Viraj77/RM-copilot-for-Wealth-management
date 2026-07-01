"""Ingest documents into ChromaDB vector store."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_openai_api_key
from src.ingestion.indexer import build_vector_store
from src.ingestion.loader import load_documents


def main():
    api_key = get_openai_api_key()
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found.")
        print("Create a .env file in the project root with:")
        print('  OPENAI_API_KEY=your-key-here')
        sys.exit(1)

    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.metadata['doc_id']} ({doc.metadata['type']}, "
              f"{doc.metadata['sensitivity']})")

    print("\nBuilding vector store...")
    store = build_vector_store(docs)
    count = store._collection.count()
    print(f"Indexed {count} chunks in ChromaDB")
    print("Done.")


if __name__ == "__main__":
    main()
