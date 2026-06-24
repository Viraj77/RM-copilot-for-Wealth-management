from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from src.config import VECTOR_DIR
from functools import lru_cache

@lru_cache(maxsize=1)
def get_retriever():

    db = FAISS.load_local(
        str(VECTOR_DIR),
        OpenAIEmbeddings(),
        allow_dangerous_deserialization=True
    )

    return db.as_retriever(
        search_kwargs={"k": 3}
    )

