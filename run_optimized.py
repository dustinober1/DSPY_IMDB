"""DSPy-optimized sentiment analysis with few-shot learning."""

import argparse
import json
from pathlib import Path
from datetime import datetime
import dspy

from src.data_loader import load_imdb_data
from src.model_config import setup_ollama_model, test_model_connection
from src.sentiment_module import ChainOfThoughtSentimentModule
from src.evaluation import evaluate_model


def sentiment_metric(example, prediction, trace=None):
    """Metric for DSPy optimization - checks if prediction matches label."""
    return example.label.lower() == prediction.sentiment.lower()


def main():
    parser = argparse.ArgumentParser(
        description="Run DSPy-optimized sentiment analysis on IMDB dataset"
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Use subset of data for faster testing (default: full dataset)"
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=50,
        help="Number of training examples for optimization (default: 50)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="jewelzufo/Qwen3-0.6B-GGUF:IQ4_NL",
        help="Ollama model name (default: jewelzufo/Qwen3-0.6B-GGUF:IQ4_NL)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results (default: results)"
    )
    parser.add_argument(
        "--max-bootstrapped-demos",
        type=int,
        default=4,
        help="Maximum number of few-shot examples (default: 4)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("DSPy OPTIMIZED SENTIMENT ANALYSIS")
    print("="*60)
    
    # Setup model
    print("\n1. Setting up model...")
    lm = setup_ollama_model(model_name=args.model)
    
    # Test connection
    print("\n2. Testing model connection...")
    if not test_model_connection(lm):
        print("Failed to connect to model. Please check Ollama is running.")
        return
    
    # Load data
    print("\n3. Loading data...")
    train_data, dev_data, test_data = load_imdb_data(subset_size=args.subset)
    
    # Use subset of training data for optimization
    train_subset = train_data[:args.train_size]
    print(f"Using {len(train_subset)} examples for optimization")
    
    # Create sentiment module
    print("\n4. Creating sentiment module with Chain of Thought...")
    sentiment_model = ChainOfThoughtSentimentModule()
    
    # Optimize with BootstrapFewShot
    print("\n5. Optimizing with DSPy BootstrapFewShot...")
    print(f"   Max bootstrapped demos: {args.max_bootstrapped_demos}")
    
    optimizer = dspy.BootstrapFewShot(
        metric=sentiment_metric,
        max_bootstrapped_demos=args.max_bootstrapped_demos,
        max_labeled_demos=args.max_bootstrapped_demos
    )
    
    print("   Compiling optimized program...")
    optimized_model = optimizer.compile(
        sentiment_model,
        trainset=train_subset
    )
    
    print("   ✓ Optimization complete")
    
    # Evaluate on test set
    print("\n6. Evaluating optimized model on test set...")
    metrics, predictions = evaluate_model(optimized_model, test_data)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"optimized_results_{timestamp}.json"
    
    results = {
        "timestamp": timestamp,
        "model": args.model,
        "subset_size": args.subset,
        "train_size": args.train_size,
        "test_size": len(test_data),
        "max_bootstrapped_demos": args.max_bootstrapped_demos,
        "metrics": metrics,
        "predictions": predictions
    }
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    # Save latest optimized for comparison
    latest_file = output_dir / "optimized_latest.json"
    with open(latest_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Latest results saved to: {latest_file}")
    
    # Save compiled program
    compiled_dir = Path("compiled_programs")
    compiled_dir.mkdir(exist_ok=True)
    compiled_file = compiled_dir / f"sentiment_optimized_{timestamp}.json"
    optimized_model.save(str(compiled_file))
    print(f"✓ Compiled program saved to: {compiled_file}")
    
    print("\n" + "="*60)
    print("OPTIMIZED EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
