"""Check test set distribution."""

from src.data_loader import load_imdb_data
from collections import Counter

# Load data
_, _, test_data = load_imdb_data(subset_size=100)

# Count labels
labels = [ex.label for ex in test_data]
counts = Counter(labels)

print("="*60)
print("TEST SET DISTRIBUTION")
print("="*60)
print(f"Total examples: {len(test_data)}")
print(f"Positive: {counts['positive']} ({counts['positive']/len(test_data)*100:.1f}%)")
print(f"Negative: {counts['negative']} ({counts['negative']/len(test_data)*100:.1f}%)")
print("="*60)

# Try with larger subset
print("\nLarger subset (500 examples):")
_, _, test_data_large = load_imdb_data(subset_size=500)
labels_large = [ex.label for ex in test_data_large]
counts_large = Counter(labels_large)
print(f"Total examples: {len(test_data_large)}")
print(f"Positive: {counts_large['positive']} ({counts_large['positive']/len(test_data_large)*100:.1f}%)")
print(f"Negative: {counts_large['negative']} ({counts_large['negative']/len(test_data_large)*100:.1f}%)")
print("="*60)
