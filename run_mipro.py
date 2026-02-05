"""Run MIPRO optimizer for prompt optimization."""

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
        description="Run MIPRO-optimized sentiment analysis on IMDB dataset"
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
        "--num-candidates",
        type=int,
        default=5,
        help="Number of prompt candidates to generate (default: 5)"
    )
    parser.add_argument(
        "--init-temperature",
        type=float,
        default=1.0,
        help="Temperature for prompt generation (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("MIPRO PROMPT OPTIMIZATION")
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
    print(f"Using {len(dev_data)} examples for validation")
    
    # Create sentiment module
    print("\n4. Creating sentiment module...")
    sentiment_model = ChainOfThoughtSentimentModule()
    
    # Optimize with MIPRO
    print("\n5. Optimizing with MIPRO...")
    print(f"   Generating {args.num_candidates} prompt candidates")
    print(f"   Temperature: {args.init_temperature}")
    print("   This may take several minutes...")
    
    try:
        from dspy.teleprompt import MIPROv2
        
        optimizer = MIPROv2(
            metric=sentiment_metric,
            auto=None,  # Disable auto mode to use custom parameters
            num_candidates=args.num_candidates,
            init_temperature=args.init_temperature,
            verbose=True
        )
        
        print("   Compiling optimized program...")
        optimized_model = optimizer.compile(
            sentiment_model,
            trainset=train_subset,
            valset=dev_data[:min(20, len(dev_data))]  # Use small validation set
        )
        
        print("   ✓ MIPRO optimization complete")
        
    except ImportError:
        print("   ✗ MIPRO not available in this DSPy version")
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
        print(f"   ✗ MIPRO optimization failed: {e}")
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
    results_file = output_dir / f"mipro_results_{timestamp}.json"
    
    results = {
        "timestamp": timestamp,
        "optimizer": "MIPROv2",
        "model": args.model,
        "subset_size": args.subset,
        "train_size": args.train_size,
        "test_size": len(test_data),
        "num_candidates": args.num_candidates,
        "init_temperature": args.init_temperature,
        "metrics": metrics,
        "predictions": predictions
    }
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    # Save latest for comparison
    latest_file = output_dir / "mipro_latest.json"
    with open(latest_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Latest results saved to: {latest_file}")
    
    # Save compiled program
    compiled_dir = Path("compiled_programs")
    compiled_dir.mkdir(exist_ok=True)
    compiled_file = compiled_dir / f"sentiment_mipro_{timestamp}.json"
    try:
        optimized_model.save(str(compiled_file))
        print(f"✓ Compiled program saved to: {compiled_file}")
    except:
        print("Note: Could not save compiled program")
    
    print("\n" + "="*60)
    print("MIPRO OPTIMIZATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
