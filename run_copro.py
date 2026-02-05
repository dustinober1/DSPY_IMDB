"""Run COPRO optimizer for coordinate prompt optimization."""

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
    try:
        pred_sentiment = prediction.sentiment.lower() if hasattr(prediction, 'sentiment') else str(prediction).lower()
        true_sentiment = example.label.lower()
        return pred_sentiment == true_sentiment
    except:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run COPRO-optimized sentiment analysis on IMDB dataset"
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
        default="gemma3:latest",
        help="Ollama model name (default: gemma3:latest)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results (default: results)"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Optimization depth (default: 3)"
    )
    parser.add_argument(
        "--breadth",
        type=int,
        default=5,
        help="Number of prompt variations per iteration (default: 5)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("COPRO COORDINATE PROMPT OPTIMIZATION")
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
    print("\n4. Creating sentiment module...")
    sentiment_model = ChainOfThoughtSentimentModule()
    
    # Optimize with COPRO
    print("\n5. Optimizing with COPRO...")
    print(f"   Depth: {args.depth}")
    print(f"   Breadth: {args.breadth}")
    print("   This may take several minutes...")
    
    try:
        from dspy.teleprompt import COPRO
        
        optimizer = COPRO(
            metric=sentiment_metric,
            depth=args.depth,
            breadth=args.breadth,
            verbose=True
        )
        
        print("   Compiling optimized program...")
        optimized_model = optimizer.compile(
            sentiment_model,
            trainset=train_subset,
            eval_kwargs={'num_threads': 1}
        )
        
        print("   ✓ COPRO optimization complete")
        
    except ImportError:
        print("   ✗ COPRO not available in this DSPy version")
        print("   Falling back to BootstrapFewShot...")
        
        optimizer = dspy.BootstrapFewShot(
            metric=sentiment_metric,
            max_bootstrapped_demos=4
        )
        optimized_model = optimizer.compile(
            sentiment_model,
            trainset=train_subset
        )
    except Exception as e:
        print(f"   ✗ COPRO optimization failed: {e}")
        print("   Falling back to BootstrapFewShot...")
        
        optimizer = dspy.BootstrapFewShot(
            metric=sentiment_metric,
            max_bootstrapped_demos=4
        )
        optimized_model = optimizer.compile(
            sentiment_model,
            trainset=train_subset
        )
    
    # Evaluate on test set
    print("\n6. Evaluating optimized model on test set...")
    metrics, predictions = evaluate_model(optimized_model, test_data)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"copro_results_{timestamp}.json"
    
    results = {
        "timestamp": timestamp,
        "optimizer": "COPRO",
        "model": args.model,
        "subset_size": args.subset,
        "train_size": args.train_size,
        "test_size": len(test_data),
        "depth": args.depth,
        "breadth": args.breadth,
        "metrics": metrics,
        "predictions": predictions
    }
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    # Save latest for comparison
    latest_file = output_dir / "copro_latest.json"
    with open(latest_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Latest results saved to: {latest_file}")
    
    # Save compiled program
    compiled_dir = Path("compiled_programs")
    compiled_dir.mkdir(exist_ok=True)
    compiled_file = compiled_dir / f"sentiment_copro_{timestamp}.json"
    try:
        optimized_model.save(str(compiled_file))
        print(f"✓ Compiled program saved to: {compiled_file}")
    except:
        print("Note: Could not save compiled program")
    
    print("\n" + "="*60)
    print("COPRO OPTIMIZATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
