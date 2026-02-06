"""Optimized baseline sentiment analysis with parallel processing."""

import argparse
import json
from pathlib import Path
from datetime import datetime

from src.data_loader import load_imdb_data
from src.model_config import setup_ollama_model_optimized, test_model_connection
from src.sentiment_module import BasicSentimentModule
from src.evaluation_batch import evaluate_model_parallel


def main():
    parser = argparse.ArgumentParser(
        description="Run optimized baseline sentiment analysis on IMDB dataset"
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
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for parallel processing (default: 32)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results (default: results)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("OPTIMIZED BASELINE SENTIMENT ANALYSIS")
    print("="*60)
    print(f"Batch size: {args.batch_size}")
    print(f"Workers: {args.workers}")
    
    # Setup model with optimizations
    print("\n1. Setting up optimized model...")
    lm = setup_ollama_model_optimized(model_name=args.model)
    
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
    
    # Evaluate on test set with parallel processing
    print("\n5. Evaluating on test set with parallel processing...")
    start_time = datetime.now()
    metrics, predictions = evaluate_model_parallel(
        sentiment_model, 
        test_data,
        batch_size=args.batch_size,
        max_workers=args.workers
    )
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    print(f"\n⏱️  Evaluation completed in {elapsed:.2f} seconds")
    print(f"   Throughput: {len(test_data)/elapsed:.2f} examples/second")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"baseline_optimized_results_{timestamp}.json"
    
    results = {
        "timestamp": timestamp,
        "model": args.model,
        "subset_size": args.subset,
        "test_size": len(test_data),
        "batch_size": args.batch_size,
        "workers": args.workers,
        "elapsed_seconds": elapsed,
        "throughput": len(test_data)/elapsed,
        "metrics": metrics,
        "predictions": predictions
    }
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    # Save latest baseline for comparison
    latest_file = output_dir / "baseline_optimized_latest.json"
    with open(latest_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Latest results saved to: {latest_file}")
    
    print("\n" + "="*60)
    print("OPTIMIZED BASELINE EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
