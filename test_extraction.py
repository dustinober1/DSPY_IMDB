"""Test the extraction logic."""

from src.sentiment_module import BasicSentimentModule

# Create module
module = BasicSentimentModule()

# Test responses
test_cases = [
    (['POSITIVE\n'], "positive"),
    (['NEGATIVE\n'], "negative"),
    (['positive\n'], "positive"),
    (['negative\n'], "negative"),
]

print("="*60)
print("TESTING EXTRACTION LOGIC")
print("="*60)

for response, expected in test_cases:
    result = module._extract_sentiment(response, "")
    status = "✓" if result == expected else "✗"
    print(f"{status} Response: {response} -> {result} (expected: {expected})")

print("="*60)
