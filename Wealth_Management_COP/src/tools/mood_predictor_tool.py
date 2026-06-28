from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.models.sentiment import SentimentSnapshot, MoodPrediction
from src.tools.sentiment_analyzer_tool import analyze_client_sentiment
from src.tools.portfolio_tool import portfolio_lookup
from config.settings import settings

def predict_meeting_mood(client_id: str, rm_tier: str = "standard") -> MoodPrediction:
    """
    Predict the client's mood for the next meeting based on sentiment history
    and current portfolio performance.
    """
    # 1. Get sentiment history
    sentiment_history: List[SentimentSnapshot] = analyze_client_sentiment(client_id)
    
    # 2. Get current portfolio data
    portfolio_data = portfolio_lookup(client_id, rm_tier)
    
    # OPTIMIZATION 4: Model Downgrading
    # Use gpt-4o-mini for mood prediction to save tokens & cost
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    structured_llm = llm.with_structured_output(MoodPrediction)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a behavioral finance expert advising a wealth manager.
Based on the client's historical sentiment timeline and their CURRENT portfolio context,
predict their likely emotional state (mood) for an upcoming meeting today.

Provide actionable conversation strategies, including topics to avoid (triggers) and topics to emphasize (to build trust).
"""),
        ("human", """
=== CLIENT SENTIMENT HISTORY ===
{sentiment_history}

=== CURRENT PORTFOLIO CONTEXT ===
{portfolio_context}

Analyze this and output the MoodPrediction.
""")
    ])
    
    # Format history
    if not sentiment_history:
        history_str = "No past interactions recorded."
    else:
        history_str = "\n\n".join([
            f"Date: {s.date}\nType: {s.interaction_type}\nTrust: {s.trust_score}/10, Anxiety: {s.anxiety_score}/10, Satisfaction: {s.satisfaction_score}/10\nSummary: {s.summary}"
            for s in sentiment_history
        ])
        
    # Format portfolio
    if not portfolio_data.get("success"):
        port_str = "Portfolio data unavailable."
    else:
        port_str = f"Client: {portfolio_data.get('client_name')}\nRisk Profile: {portfolio_data.get('risk_profile')}\n"
        port_str += f"Total AUM: ${portfolio_data.get('total_aum_usd'):,.2f}\n"
        port_str += "Holdings Summary:\n"
        for h in portfolio_data.get("holdings", []):
            port_str += f"- {h['ticker']}: {h['allocation_pct']}% alloc, {h['gain_loss_pct']}% gain/loss\n"

    chain = prompt | structured_llm
    
    try:
        prediction = chain.invoke({
            "sentiment_history": history_str,
            "portfolio_context": port_str
        })
        return prediction
    except Exception as e:
        print(f"Error predicting mood: {e}")
        # Fallback
        return MoodPrediction(
            predicted_mood="Unknown",
            confidence="Low",
            reasoning=f"Error generating prediction: {str(e)}",
            conversation_strategy="Proceed with standard review protocol.",
            topics_to_avoid=[],
            topics_to_emphasize=["Long term strategy"]
        )

def get_mood_predictor_tool():
    """Return a LangChain-compatible tool for predicting client mood."""
    try:
        from langchain_core.tools import tool
        import json
        
        @tool
        def mood_predictor_tool(client_id: str, rm_tier: str = "standard") -> str:
            """
            Predict a client's mood for an upcoming meeting based on their historical
            sentiment timeline and current portfolio performance.
            
            Use this tool when the RM asks about how to approach a meeting, what the client's 
            current emotional state is, or what topics to avoid/emphasize.
            
            Args:
                client_id: Client identifier in format 'C-NNN' (e.g. 'C-204').
                rm_tier: RM entitlement tier ('standard', 'premium', 'institutional').
                
            Returns:
                JSON string with the mood prediction, reasoning, and conversation strategy.
            """
            prediction = predict_meeting_mood(client_id, rm_tier)
            return prediction.model_dump_json()
            
        return mood_predictor_tool
    except ImportError:
        pass
