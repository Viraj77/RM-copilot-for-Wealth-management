from src.sample_clients import CLIENTS
from src.retriever import get_retriever

retriever = None

def portfolio_lookup(client_id:str):
    return CLIENTS.get(client_id, {})

def rag_search(query: str, client_id: str = None):

    global retriever

    if retriever is None:
        retriever = get_retriever()

    docs = retriever.vectorstore.similarity_search(
        query,
        k=6
    )

    filtered_docs = []

    for d in docs:

        metadata = d.metadata

        doc_client = metadata.get("client_id", "GLOBAL")

        # =====================================
        # CLIENT QUERY
        # =====================================

        if client_id:

            # allow:
            # same client docs
            # global docs

            if doc_client in [client_id, "GLOBAL"]:

                filtered_docs.append({
                    "content": d.page_content,
                    "source": metadata.get("source"),
                    "client_id": doc_client
                })

        # =====================================
        # GENERIC QUERY
        # =====================================

        else:

            # ONLY global docs

            if doc_client == "GLOBAL":

                filtered_docs.append({
                    "content": d.page_content,
                    "source": metadata.get("source"),
                    "client_id": doc_client
                })

    return filtered_docs

def suitability_checker(risk_profile:str, recommendation:str):
    recommendation = recommendation.lower()
    if risk_profile == 'Conservative' and 'equity' in recommendation:
        return 'Needs Review'
    return 'Cleared'
