import streamlit as st
import pandas as pd
from typing import List

from src.models.sentiment import SentimentSnapshot, MoodPrediction
from src.tools.sentiment_analyzer_tool import analyze_client_sentiment
from src.tools.mood_predictor_tool import predict_meeting_mood
from src.tools.portfolio_tool import portfolio_lookup

@st.cache_data(show_spinner=False)
def get_cached_sentiment(client_id: str):
    return analyze_client_sentiment(client_id)

@st.cache_data(show_spinner=False)
def get_cached_mood(client_id: str, rm_tier: str):
    return predict_meeting_mood(client_id, rm_tier)

def render_sentiment_view(client_id: str, rm_tier: str, embedded: bool = False):
    if not embedded:
        st.title("🧠 Client Sentiment Time-Machine")
        st.markdown(
            "Analyze the emotional trajectory of past interactions and predict the client's mood for your next meeting."
        )
    
    # Check entitlement first
    client_data = portfolio_lookup(client_id, rm_tier)
    if not client_data.get("success"):
        st.error(f"🔒 Access Restricted: {client_data.get('error', 'Unknown Error')}")
        return

    if not embedded:
        st.markdown(f"### Evaluating Sentiment for **{client_data.get('client_name')}** ({client_id})")

    # Spinner while analyzing
    with st.spinner("Analyzing historical interaction transcripts using GPT-4o..."):
        # 1. Fetch sentiment history
        history: List[SentimentSnapshot] = get_cached_sentiment(client_id)
        
        # 2. Predict next meeting mood
        mood_pred: MoodPrediction = get_cached_mood(client_id, rm_tier)
        
    if not history:
        st.warning("No historical interaction data found for this client.")
        return

    # --- 1. MOOD FORECAST ---
    st.markdown("---")
    st.markdown("## 🎯 Next Meeting Forecast")
    
    # Determine color based on mood type
    mood_lower = mood_pred.predicted_mood.lower()
    if any(word in mood_lower for word in ["anxious", "angry", "dissatisfied", "frustrated"]):
        color = "red"
        icon = "⚠️"
    elif any(word in mood_lower for word in ["happy", "confident", "satisfied", "trusting"]):
        color = "green"
        icon = "✅"
    else:
        color = "blue"
        icon = "ℹ️"
        
    st.markdown(f"### {icon} Predicted Mood: **:{color}[{mood_pred.predicted_mood}]** (Confidence: {mood_pred.confidence})")
    
    with st.expander("🧠 Why? (AI Reasoning)", expanded=True):
        st.write(mood_pred.reasoning)
        
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**💡 Strategy:** {mood_pred.conversation_strategy}")
        if mood_pred.topics_to_emphasize:
            st.success("**Lead With:**\n- " + "\n- ".join(mood_pred.topics_to_emphasize))
    with col2:
        if mood_pred.topics_to_avoid:
            st.error("**Avoid/Tread Carefully:**\n- " + "\n- ".join(mood_pred.topics_to_avoid))


    # --- 2. SENTIMENT TIMELINE ---
    st.markdown("---")
    st.markdown("## 📈 Historical Sentiment Trajectory")
    
    # Convert history to DataFrame for charting
    data = []
    for snapshot in history:
        data.append({
            "Date": pd.to_datetime(snapshot.date),
            "Trust": snapshot.trust_score,
            "Anxiety": snapshot.anxiety_score,
            "Satisfaction": snapshot.satisfaction_score,
            "Engagement": snapshot.engagement_score
        })
    df = pd.DataFrame(data)
    df = df.set_index("Date")
    
    # Plot using Streamlit's native line chart
    st.line_chart(df, use_container_width=True)
    
    # Show underlying data summary
    st.markdown("### Interaction Summaries")
    for snapshot in reversed(history): # Show newest first
        with st.container(border=True):
            st.markdown(f"**{snapshot.date} ({snapshot.interaction_type.title()})**")
            st.caption(snapshot.summary)
            # small metric row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Trust", f"{snapshot.trust_score}/10")
            c2.metric("Anxiety", f"{snapshot.anxiety_score}/10")
            c3.metric("Satisfaction", f"{snapshot.satisfaction_score}/10")
            c4.metric("Engagement", f"{snapshot.engagement_score}/10")
