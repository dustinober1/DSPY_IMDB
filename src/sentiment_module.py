"""DSPy modules for sentiment analysis."""

import dspy
import re


class BasicSentimentModule(dspy.Module):
    """Basic sentiment classification module using text-based prompting."""
    
    def __init__(self):
        super().__init__()
        # Use a simple text-based prompt instead of structured JSON
        self.prompt_template = dspy.Predict("review -> sentiment")
    
    def forward(self, review: str) -> dspy.Prediction:
        """
        Classify sentiment of a review.
        
        Args:
            review: The review text to classify
            
        Returns:
            DSPy Prediction with sentiment field
        """
        # Truncate very long reviews to avoid token limits
        review_text = review[:500] if len(review) > 500 else review
        
        # Create a simple prompt
        prompt = f"Classify this movie review as 'positive' or 'negative'.\n\nReview: {review_text}\n\nSentiment:"
        
        try:
            # Get response from model
            response = dspy.settings.lm(prompt)
            
            # Extract sentiment from response
            sentiment = self._extract_sentiment(response)
            
            # Return as DSPy Prediction
            return dspy.Prediction(sentiment=sentiment)
        except Exception as e:
            # Fallback to negative on error
            return dspy.Prediction(sentiment="negative")
    
    def _extract_sentiment(self, response: str) -> str:
        """Extract sentiment from model response."""
        if isinstance(response, list) and len(response) > 0:
            # Handle list response format
            if isinstance(response[0], dict) and 'text' in response[0]:
                text = response[0]['text']
            else:
                text = str(response[0])
        else:
            text = str(response)
        
        # Normalize and extract
        text_lower = text.lower().strip()
        
        # Look for positive indicators
        if any(word in text_lower for word in ['positive', 'pos', 'good', 'great', 'excellent']):
            return "positive"
        # Look for negative indicators
        elif any(word in text_lower for word in ['negative', 'neg', 'bad', 'poor', 'terrible']):
            return "negative"
        else:
            # Default based on simple heuristics
            # Count positive/negative words in response
            positive_count = sum(1 for word in ['good', 'great', 'love', 'excellent', 'amazing'] if word in text_lower)
            negative_count = sum(1 for word in ['bad', 'poor', 'hate', 'terrible', 'awful'] if word in text_lower)
            
            if positive_count > negative_count:
                return "positive"
            else:
                return "negative"


class ChainOfThoughtSentimentModule(dspy.Module):
    """Sentiment classification with explicit reasoning steps."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, review: str) -> dspy.Prediction:
        """
        Classify sentiment with reasoning.
        
        Args:
            review: The review text to classify
            
        Returns:
            DSPy Prediction with sentiment and rationale fields
        """
        # Truncate very long reviews
        review_text = review[:500] if len(review) > 500 else review
        
        # Create a prompt that encourages reasoning
        prompt = f"""Analyze this movie review and determine if it's positive or negative.

Review: {review_text}

Think step by step:
1. What are the key words and phrases?
2. What is the overall tone?
3. Is the sentiment positive or negative?

Final answer (just say 'positive' or 'negative'):"""
        
        try:
            # Get response from model
            response = dspy.settings.lm(prompt)
            
            # Extract sentiment
            sentiment = self._extract_sentiment(response)
            
            # Extract rationale if available
            rationale = self._extract_rationale(response)
            
            return dspy.Prediction(sentiment=sentiment, rationale=rationale)
        except Exception as e:
            # Fallback
            return dspy.Prediction(sentiment="negative", rationale="Error in processing")
    
    def _extract_sentiment(self, response: str) -> str:
        """Extract sentiment from model response."""
        if isinstance(response, list) and len(response) > 0:
            if isinstance(response[0], dict) and 'text' in response[0]:
                text = response[0]['text']
            else:
                text = str(response[0])
        else:
            text = str(response)
        
        text_lower = text.lower().strip()
        
        # Look for explicit sentiment in the last part of response
        lines = text_lower.split('\n')
        last_line = lines[-1] if lines else text_lower
        
        if 'positive' in last_line:
            return "positive"
        elif 'negative' in last_line:
            return "negative"
        # Check entire response
        elif 'positive' in text_lower:
            return "positive"
        elif 'negative' in text_lower:
            return "negative"
        else:
            # Fallback heuristic
            positive_count = text_lower.count('good') + text_lower.count('great') + text_lower.count('love')
            negative_count = text_lower.count('bad') + text_lower.count('poor') + text_lower.count('hate')
            return "positive" if positive_count > negative_count else "negative"
    
    def _extract_rationale(self, response: str) -> str:
        """Extract reasoning from response."""
        if isinstance(response, list) and len(response) > 0:
            if isinstance(response[0], dict) and 'text' in response[0]:
                text = response[0]['text']
            else:
                text = str(response[0])
        else:
            text = str(response)
        
        # Return first 200 chars as rationale
        return text[:200]


def normalize_sentiment(sentiment: str) -> str:
    """
    Normalize sentiment output to 'positive' or 'negative'.
    
    Args:
        sentiment: Raw sentiment string from model
        
    Returns:
        Normalized sentiment ('positive' or 'negative')
    """
    if not sentiment:
        return "negative"
    
    sentiment_lower = sentiment.lower().strip()
    
    # Handle various formats
    if "positive" in sentiment_lower or sentiment_lower == "pos":
        return "positive"
    elif "negative" in sentiment_lower or sentiment_lower == "neg":
        return "negative"
    else:
        # Default to negative if unclear
        return "negative"
