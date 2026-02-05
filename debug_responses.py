"""Quick test to see what Gemma3 is actually returning."""

import dspy
from src.model_config import setup_ollama_model

# Setup model
lm = setup_ollama_model(model_name="gemma3:latest")

# Test with a clearly positive review
positive_review = "This movie was absolutely fantastic! I loved every minute of it. The acting was superb and the story was engaging. Highly recommended!"

# Test with a clearly negative review
negative_review = "This was the worst movie I've ever seen. Terrible acting, boring plot, complete waste of time. Do not watch this."

# Create the prompt
def test_prompt(review_text):
    prompt = f"""You are a movie review sentiment classifier. Classify the following review as either POSITIVE or NEGATIVE.

Review: {review_text}

Instructions:
- If the review is mostly favorable, praising the movie, or recommends it, respond with: POSITIVE
- If the review is mostly critical, negative, or discourages watching, respond with: NEGATIVE
- Respond with ONLY the word POSITIVE or NEGATIVE, nothing else.

Classification:"""
    return prompt

print("="*60)
print("TESTING GEMMA3 RESPONSES")
print("="*60)

print("\n1. Positive Review Test:")
print("-" * 60)
response = dspy.settings.lm(test_prompt(positive_review))
print(f"Response type: {type(response)}")
print(f"Response: {response}")
print(f"Response repr: {repr(response)}")

print("\n2. Negative Review Test:")
print("-" * 60)
response = dspy.settings.lm(test_prompt(negative_review))
print(f"Response type: {type(response)}")
print(f"Response: {response}")
print(f"Response repr: {repr(response)}")

print("\n" + "="*60)
