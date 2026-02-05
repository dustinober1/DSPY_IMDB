"""Debug script to see what's happening during classification."""

import dspy
from src.model_config import setup_ollama_model
from src.data_loader import load_imdb_data
from src.sentiment_module import BasicSentimentModule

# Setup
lm = setup_ollama_model(model_name="gemma3:latest")
_, _, test_data = load_imdb_data(subset_size=100)

# Create module
module = BasicSentimentModule()

# Test first 5 examples
print("="*80)
print("DEBUGGING CLASSIFICATION")
print("="*80)

for i, example in enumerate(test_data[:5]):
    print(f"\n{i+1}. True Label: {example.label}")
    print(f"Review (first 100 chars): {example.text[:100]}...")
    
    # Get prediction
    pred = module(review=example.text)
    print(f"Predicted: {pred.sentiment}")
    print(f"Match: {'✓' if pred.sentiment == example.label else '✗'}")
    print("-" * 80)
