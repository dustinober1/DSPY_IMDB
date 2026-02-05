"""Baseline sentiment analysis without DSPy optimization."""

import argparse
import json
from pathlib import Path
from datetime import datetime

from src.data_loader import load_imdb_data
from src.model_config import setup_ollama_model, test_model_connection
from src.sentiment_module import BasicSentimentModule
from src.evaluation import evaluate_model


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline sentiment analysis on IMDB dataset"
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Use subset of data for faster testing (default: full dataset)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma3:latest",
        help="Ollama model name (default: gemma3:latest)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results (default: results)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("BASELINE SENTIMENT ANALYSIS")
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
    
    # Create baseline module
    print("\n4. Creating baseline sentiment module...")
    sentiment_model = BasicSentimentModule()
    
    # Evaluate on test set
    print("\n5. Evaluating on test set...")
    metrics, predictions = evaluate_model(sentiment_model, test_data)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"baseline_results_{timestamp}.json"
    
    results = {
        "timestamp": timestamp,
        "model": args.model,
        "subset_size": args.subset,
        "test_size": len(test_data),
        "metrics": metrics,
        "predictions": predictions
    }
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    # Save latest baseline for comparison
    latest_file = output_dir / "baseline_latest.json"
    with open(latest_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Latest results saved to: {latest_file}")
    
    print("\n" + "="*60)
    print("BASELINE EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
