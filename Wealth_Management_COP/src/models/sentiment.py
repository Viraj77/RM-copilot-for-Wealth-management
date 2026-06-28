from pydantic import BaseModel, Field
from typing import List, Optional

class SentimentSnapshot(BaseModel):
    """Extraction of emotional signals from a single interaction."""
    date: str = Field(..., description="Date of the interaction (YYYY-MM-DD)")
    interaction_type: str = Field(..., description="e.g., meeting, email, phone")
    trust_score: int = Field(..., ge=1, le=10, description="1 (Distrustful) to 10 (Highly Trusting)")
    anxiety_score: int = Field(..., ge=1, le=10, description="1 (Calm) to 10 (Highly Anxious/Panicked)")
    satisfaction_score: int = Field(..., ge=1, le=10, description="1 (Very Dissatisfied) to 10 (Very Satisfied)")
    engagement_score: int = Field(..., ge=1, le=10, description="1 (Apathetic) to 10 (Highly Engaged/Proactive)")
    key_topics: List[str] = Field(..., description="Main topics discussed")
    summary: str = Field(..., description="One-sentence summary of the client's emotional state")

class MoodPrediction(BaseModel):
    """Prediction of the client's mood for the next meeting."""
    predicted_mood: str = Field(..., description="The predicted primary emotion (e.g., Anxious, Confident, Demanding)")
    confidence: str = Field(..., description="High, Medium, or Low")
    reasoning: str = Field(..., description="Explanation for the prediction based on history and market context")
    conversation_strategy: str = Field(..., description="Advice for the RM on how to approach the conversation")
    topics_to_avoid: List[str] = Field(default_factory=list, description="Topics that might trigger negative emotions")
    topics_to_emphasize: List[str] = Field(default_factory=list, description="Topics to lead with to build trust")
