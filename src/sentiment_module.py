"""DSPy modules for sentiment analysis."""

import dspy
import re


class BasicSentimentModule(dspy.Module):
    """Basic sentiment classification module using text-based prompting."""
    
    def __init__(self):
        super().__init__()
        # Use a simple text-based prompt instead of structured JSON
        self.prompt_template = dspy.Predict("review -> sentiment")
    
    def forward(self, text: str = None, review: str = None) -> dspy.Prediction:
        """
        Classify sentiment of a movie review.
        
        Args:
            text: The review text to classify (DSPy optimizer uses this)
            review: The review text to classify (manual calls use this)
            
        Returns:
            DSPy Prediction with sentiment field
        """
        # Accept either 'text' or 'review' parameter
        review_text = text if text is not None else review
        if review_text is None:
            return dspy.Prediction(sentiment="negative")
        
        # Truncate very long reviews to avoid token limits
        review_text = review_text[:500] if len(review_text) > 500 else review_text
        
        # Create a clear, explicit prompt
        prompt = f"""You are a movie review sentiment classifier. Classify the following review as either POSITIVE or NEGATIVE.

Review: {review_text}

Instructions:
- If the review is mostly favorable, praising the movie, or recommends it, respond with: POSITIVE
- If the review is mostly critical, negative, or discourages watching, respond with: NEGATIVE
- Respond with ONLY the word POSITIVE or NEGATIVE, nothing else.

Classification:"""
        
        try:
            # Get response from model
            response = dspy.settings.lm(prompt)
            
            # Extract sentiment from response
            sentiment = self._extract_sentiment(response, review_text)
            
            # If still unclear, analyze review content directly
            if sentiment is None:
                sentiment = self._analyze_review_content(review_text)
            
            # Return as DSPy Prediction
            return dspy.Prediction(sentiment=sentiment)
        except Exception as e:
            # On error, try to analyze the review content
            try:
                sentiment = self._analyze_review_content(review_text)
                return dspy.Prediction(sentiment=sentiment)
            except:
                # Ultimate fallback
                return dspy.Prediction(sentiment="negative")
    
    def _extract_sentiment(self, response: str, review: str = "") -> str:
        """Extract sentiment from model response with improved logic."""
        # Handle different response formats
        if isinstance(response, dict):
            response = response.get('text', str(response))
        elif isinstance(response, list):
            response = response[0] if response else ""
            if isinstance(response, dict):
                response = response.get('text', str(response))
        
        text_lower = str(response).lower().strip()
        
        # First, look for explicit POSITIVE/NEGATIVE at the start (our prompt format)
        first_line = text_lower.split('\n')[0].strip()
        if first_line == 'positive' or first_line.startswith('positive'):
            return "positive"
        if first_line == 'negative' or first_line.startswith('negative'):
            return "negative"
        
        # Look for clear sentiment indicators in the response
        if 'positive' in text_lower and 'negative' not in text_lower:
            return "positive"
        elif 'negative' in text_lower and 'positive' not in text_lower:
            return "negative"
        elif 'positive' in text_lower and 'negative' in text_lower:
            # Both mentioned - check which comes last
            pos_idx = text_lower.rfind('positive')
            neg_idx = text_lower.rfind('negative')
            return "positive" if pos_idx > neg_idx else "negative"
        
        # If no clear sentiment in response, analyze the review content
        if review:
            return self._analyze_review_content(review)
        
        # Last resort: return None to indicate uncertainty rather than defaulting
        return None
    
    def _analyze_review_content(self, review: str) -> str:
        """Analyze review content directly as fallback."""
        review_lower = review.lower()
        
        # Count sentiment indicators in the review
        positive_words = ['good', 'great', 'love', 'excellent', 'amazing', 'wonderful', 
                         'fantastic', 'best', 'perfect', 'enjoyed', 'brilliant', 'outstanding']
        negative_words = ['bad', 'poor', 'hate', 'terrible', 'awful', 'horrible', 
                         'worst', 'boring', 'waste', 'disappointed', 'disappointing']
        
        positive_count = sum(1 for word in positive_words if word in review_lower)
        negative_count = sum(1 for word in negative_words if review_lower)
        
        # Weight by review length
        if len(review) > 0:
            positive_score = positive_count / (len(review) / 100)
            negative_score = negative_count / (len(review) / 100)
            
            if positive_score > negative_score:
                return "positive"
            elif negative_score > positive_score:
                return "negative"
        
        # If still unclear, default to negative
        return "negative"


class ChainOfThoughtSentimentModule(dspy.Module):
    """Sentiment classification with explicit reasoning steps."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, text: str = None, review: str = None) -> dspy.Prediction:
        """
        Classify sentiment with reasoning.
        
        Args:
            text: The review text to classify (DSPy optimizer uses this)
            review: The review text to classify (manual calls use this)
            
        Returns:
            DSPy Prediction with sentiment and rationale fields
        """
        # Accept either 'text' or 'review' parameter
        review_text = text if text is not None else review
        if review_text is None:
            return dspy.Prediction(sentiment="negative", rationale="No input provided")
        
        # Truncate very long reviews
        review_text = review_text[:500] if len(review_text) > 500 else review_text
        
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
