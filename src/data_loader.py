"""Data loading and preparation for IMDB sentiment analysis."""

from typing import List, Tuple
import dspy
from datasets import load_dataset
from tqdm import tqdm


def create_imdb_example(text: str, label: str) -> dspy.Example:
    """Create a properly configured DSPy Example for IMDB data."""
    # Create example with both text and label
    example = dspy.Example(text=text, label=label)
    # Set 'text' as the input field and 'label' as the output
    example = example.with_inputs("text")
    return example


def load_imdb_data(
    subset_size: int = None,
    train_split: float = 0.7,
    dev_split: float = 0.15,
    cache_dir: str = "./data"
) -> Tuple[List[dspy.Example], List[dspy.Example], List[dspy.Example]]:
    """
    Load and prepare the IMDB dataset for DSPy.
    
    Args:
        subset_size: If provided, only load this many examples from train set
        train_split: Proportion of data for training (default 0.7)
        dev_split: Proportion of data for development/validation (default 0.15)
        cache_dir: Directory to cache the dataset
        
    Returns:
        Tuple of (train_data, dev_data, test_data) as lists of IMDBExample
    """
    print("Loading IMDB dataset...")
    
    # Load the dataset
    dataset = load_dataset("stanfordnlp/imdb", cache_dir=cache_dir)
    
    # Convert labels to text
    label_map = {0: "negative", 1: "positive"}
    
    # Process training data
    train_raw = dataset["train"]
    if subset_size:
        train_raw = train_raw.select(range(min(subset_size, len(train_raw))))
    
    print(f"Processing {len(train_raw)} training examples...")
    train_examples = [
        create_imdb_example(
            text=example["text"],
            label=label_map[example["label"]]
        )
        for example in tqdm(train_raw, desc="Processing train")
    ]
    
    # Split train into train/dev
    split_idx = int(len(train_examples) * train_split)
    dev_idx = split_idx + int(len(train_examples) * dev_split)
    
    train_data = train_examples[:split_idx]
    dev_data = train_examples[split_idx:dev_idx]
    
    # Process test data
    test_raw = dataset["test"]
    if subset_size:
        # Use proportional subset for test
        test_subset_size = int(subset_size * 0.3)
        test_raw = test_raw.select(range(min(test_subset_size, len(test_raw))))
    
    print(f"Processing {len(test_raw)} test examples...")
    test_data = [
        create_imdb_example(
            text=example["text"],
            label=label_map[example["label"]]
        )
        for example in tqdm(test_raw, desc="Processing test")
    ]
    
    print(f"\nDataset loaded:")
    print(f"  Train: {len(train_data)} examples")
    print(f"  Dev: {len(dev_data)} examples")
    print(f"  Test: {len(test_data)} examples")
    
    return train_data, dev_data, test_data


def get_sample_examples(n: int = 5) -> List[dspy.Example]:
    """Get a small sample of examples for quick testing."""
    train_data, _, _ = load_imdb_data(subset_size=n)
    return train_data[:n]
