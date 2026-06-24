from pathlib import Path

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    Docx2txtLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from src.config import USER_DOCS, VECTOR_DIR


def load_documents():

    docs = []

    for file in USER_DOCS.glob("*"):

        suffix = file.suffix.lower()

        try:

            if suffix == ".txt":
                loader = TextLoader(str(file), encoding="utf-8")

            elif suffix == ".pdf":
                loader = PyPDFLoader(str(file))

            elif suffix == ".csv":
                loader = CSVLoader(str(file))

            elif suffix in [".docx", ".doc"]:
                loader = Docx2txtLoader(str(file))

            else:
                continue

            loaded_docs = loader.load()

            for d in loaded_docs:
                d.metadata["source"] = file.name

            docs.extend(loaded_docs)

        except Exception as e:
            print(f"Error loading {file.name}: {e}")

    return docs


def build_vector_store():

    docs = load_documents()

    if not docs:
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()

    db = FAISS.from_documents(chunks, embeddings)

    VECTOR_DIR.mkdir(exist_ok=True)

    db.save_local(str(VECTOR_DIR))

    return len(chunks)