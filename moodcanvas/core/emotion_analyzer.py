import numpy as np
from textblob import TextBlob
from typing import Dict, Tuple

class EmotionAnalyzer:
    """Analyzes text entries to determine emotional content and intensity."""

    # Core emotion categories and their associated terms
    EMOTION_CATEGORIES = {
        'joy': ['happy', 'excited', 'delighted', 'pleased', 'content'],
        'sadness': ['sad', 'disappointed', 'depressed', 'gloomy', 'unhappy'],
        'anger': ['angry', 'furious', 'irritated', 'frustrated', 'annoyed'],
        'fear': ['afraid', 'scared', 'anxious', 'worried', 'nervous'],
        'love': ['loving', 'caring', 'affectionate', 'tender', 'passionate'],
        'surprise': ['amazed', 'astonished', 'shocked', 'surprised', 'stunned']
    }

    def analyze_entry(self, text: str) -> Dict[str, float]:
        """Analyze a journal entry and return emotion scores.

        Args:
            text: The journal entry text to analyze

        Returns:
            Dictionary mapping emotion categories to intensity scores (0-1)
        """
        # Get base sentiment
        blob = TextBlob(text.lower())
        sentiment_score = (blob.sentiment.polarity + 1) / 2  # Normalize to 0-1

        # Calculate emotion scores
        scores = {}
        words = set(text.lower().split())
        
        for emotion, terms in self.EMOTION_CATEGORIES.items():
            # Calculate match score based on term frequency
            term_matches = sum(term in words for term in terms)
            raw_score = term_matches / len(terms)
            
            # Weight the score with sentiment
            scores[emotion] = self._calculate_weighted_score(raw_score, sentiment_score)
            
        return self._normalize_scores(scores)

    def get_primary_emotion(self, text: str) -> Tuple[str, float]:
        """Determine the primary emotion expressed in the text.

        Args:
            text: The journal entry text to analyze

        Returns:
            Tuple of (emotion_category, intensity_score)
        """
        scores = self.analyze_entry(text)
        primary_emotion = max(scores.items(), key=lambda x: x[1])
        return primary_emotion

    def _calculate_weighted_score(self, raw_score: float, sentiment: float) -> float:
        """Calculate weighted emotion score using raw term matching and sentiment."""
        return (raw_score * 0.7) + (sentiment * 0.3)

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize emotion scores to ensure they sum to 1.0"""
        total = sum(scores.values())
        if total == 0:
            return {k: 0.0 for k in scores}
        return {k: v/total for k, v in scores.items()}
