import json
from pathlib import Path
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.models.sentiment import SentimentSnapshot
from config.settings import settings

def get_client_interactions(client_id: str) -> List[dict]:
    """Load interactions for a specific client from the JSON file."""
    path = Path("data/interactions/interactions.json")
    if not path.exists():
        return []
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    return [i for i in data.get("interactions", []) if i.get("client_id") == client_id]

def analyze_client_sentiment(client_id: str) -> List[SentimentSnapshot]:
    """
    Analyze all historical interactions for a client and return a timeline
    of SentimentSnapshots.
    """
    interactions = get_client_interactions(client_id)
    if not interactions:
        return []
        
    # OPTIMIZATION 4: Model Downgrading
    # Use gpt-4o-mini for simple sentiment extraction to save tokens & cost
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    structured_llm = llm.with_structured_output(SentimentSnapshot)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert behavioral psychologist and wealth management relationship analyst. "
                   "Analyze the provided client interaction and extract the emotional signals. "
                   "Rate trust, anxiety, satisfaction, and engagement on a 1-10 scale."),
        ("human", "Date: {date}\nType: {type}\nContent: {content}")
    ])
    
    chain = prompt | structured_llm
    
    snapshots = []
    for interaction in interactions:
        try:
            # Generate snapshot
            snapshot = chain.invoke({
                "date": interaction["date"],
                "type": interaction["type"],
                "content": interaction["content"]
            })
            snapshots.append(snapshot)
        except Exception as e:
            print(f"Error analyzing interaction {interaction.get('interaction_id')}: {e}")
            
    # Sort chronologically
    snapshots.sort(key=lambda x: x.date)
    return snapshots
