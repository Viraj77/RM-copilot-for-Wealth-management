"""
Unit tests for the Sentiment Time-Machine and Mood Predictor tools.
"""
import json
import pytest

from src.tools.sentiment_analyzer_tool import get_sentiment_history
from src.tools.mood_predictor_tool import predict_meeting_mood, get_mood_predictor_tool
from src.models.sentiment import SentimentVerdict

class TestSentimentTool:
    def test_get_sentiment_history_known_client(self):
        history = get_sentiment_history("C-204")
        assert len(history) > 0
        for entry in history:
            assert "date" in entry
            assert "channel" in entry
            assert "summary" in entry
            assert "sentiment" in entry

    def test_get_sentiment_history_unknown_client(self):
        history = get_sentiment_history("C-999")
        assert history == []

    def test_predict_meeting_mood_success(self, monkeypatch):
        # Mock analyze_client_sentiment to return dummy data without calling LLM
        def mock_analyze(client_id):
            from src.models.sentiment import ClientSentimentAnalysis
            return ClientSentimentAnalysis(
                client_id=client_id,
                overall_sentiment=SentimentVerdict.POSITIVE,
                recent_trend="Improving",
                key_concerns=["None"],
                engagement_level="High",
                recommended_approach="Be proactive."
            )
        monkeypatch.setattr("src.tools.mood_predictor_tool.analyze_client_sentiment", mock_analyze)
        
        # Mock portfolio_lookup
        def mock_portfolio(client_id, rm_tier="standard"):
            return {
                "success": True,
                "holdings": [{"ticker": "AAPL", "gain_loss_pct": -5.0}],
                "risk_profile": "balanced"
            }
        monkeypatch.setattr("src.tools.mood_predictor_tool.portfolio_lookup", mock_portfolio)
        
        result = predict_meeting_mood("C-204")
        assert result["success"] is True
        assert result["predicted_mood"] in ["Positive", "Neutral", "Negative", "Volatile"]
        assert "recent_trend" in result
        
    def test_predict_meeting_mood_unknown_client(self, monkeypatch):
        # Mock to return None when client not found
        def mock_analyze(client_id):
            return None
        monkeypatch.setattr("src.tools.mood_predictor_tool.analyze_client_sentiment", mock_analyze)
        
        result = predict_meeting_mood("C-999")
        assert result["success"] is False
        assert "Failed to analyze" in result["error"]
        
    def test_mood_predictor_tool_wrapper(self):
        tool = get_mood_predictor_tool()
        assert tool is not None
        assert tool.name == "predict_meeting_mood"
