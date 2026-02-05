"""DSPy modules for sentiment analysis."""

import dspy
from typing import Literal


class SentimentClassifier(dspy.Signature):
    """Classify the sentiment of a movie review as positive or negative."""
    
    review: str = dspy.InputField(desc="The movie review text to classify")
    sentiment: Literal["positive", "negative"] = dspy.OutputField(
        desc="The sentiment classification: either 'positive' or 'negative'"
    )


class BasicSentimentModule(dspy.Module):
    """Basic sentiment classification module using DSPy Predict."""
    
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(SentimentClassifier)
    
    def forward(self, review: str) -> dspy.Prediction:
        """
        Classify sentiment of a review.
        
        Args:
            review: The review text to classify
            
        Returns:
            DSPy Prediction with sentiment field
        """
        prediction = self.predictor(review=review)
        return prediction


class ChainOfThoughtSentimentModule(dspy.Module):
    """Sentiment classification with chain-of-thought reasoning."""
    
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(SentimentClassifier)
    
    def forward(self, review: str) -> dspy.Prediction:
        """
        Classify sentiment with reasoning.
        
        Args:
            review: The review text to classify
            
        Returns:
            DSPy Prediction with sentiment and rationale fields
        """
        prediction = self.predictor(review=review)
        return prediction


def normalize_sentiment(sentiment: str) -> str:
    """
    Normalize sentiment output to 'positive' or 'negative'.
    
    Args:
        sentiment: Raw sentiment string from model
        
    Returns:
        Normalized sentiment ('positive' or 'negative')
    """
    sentiment_lower = sentiment.lower().strip()
    
    # Handle various formats
    if "positive" in sentiment_lower or sentiment_lower == "pos":
        return "positive"
    elif "negative" in sentiment_lower or sentiment_lower == "neg":
        return "negative"
    else:
        # Default to negative if unclear
        return "negative"
