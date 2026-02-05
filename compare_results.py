"""Compare baseline and optimized sentiment analysis results."""

import argparse
import json
from pathlib import Path
import pandas as pd

from src.evaluation import compare_results


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline and optimized sentiment analysis results"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="results/baseline_latest.json",
        help="Path to baseline results JSON (default: results/baseline_latest.json)"
    )
    parser.add_argument(
        "--optimized",
        type=str,
        default="results/optimized_latest.json",
        help="Path to optimized results JSON (default: results/optimized_latest.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/comparison.txt",
        help="Path to save comparison report (default: results/comparison.txt)"
    )
    
    args = parser.parse_args()
    
    # Load results
    print("Loading results...")
    
    baseline_path = Path(args.baseline)
    optimized_path = Path(args.optimized)
    
    if not baseline_path.exists():
        print(f"Error: Baseline results not found at {baseline_path}")
        print("Please run: python run_baseline.py first")
        return
    
    if not optimized_path.exists():
        print(f"Error: Optimized results not found at {optimized_path}")
        print("Please run: python run_optimized.py first")
        return
    
    with open(baseline_path) as f:
        baseline_data = json.load(f)
    
    with open(optimized_path) as f:
        optimized_data = json.load(f)
    
    # Extract metrics
    baseline_metrics = baseline_data["metrics"]
    optimized_metrics = optimized_data["metrics"]
    
    # Create comparison
    print("\n" + "="*70)
    print("BASELINE vs OPTIMIZED COMPARISON")
    print("="*70)
    
    comparison_df = compare_results(baseline_metrics, optimized_metrics)
    
    # Print comparison
    print("\n" + comparison_df.to_string(index=False))
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nBaseline Model: {baseline_data['model']}")
    print(f"  Test Size: {baseline_data['test_size']}")
    print(f"  Accuracy: {baseline_metrics['accuracy']:.4f}")
    
    print(f"\nOptimized Model: {optimized_data['model']}")
    print(f"  Test Size: {optimized_data['test_size']}")
    print(f"  Training Examples: {optimized_data['train_size']}")
    print(f"  Max Few-Shot Demos: {optimized_data['max_bootstrapped_demos']}")
    print(f"  Accuracy: {optimized_metrics['accuracy']:.4f}")
    
    accuracy_improvement = (optimized_metrics['accuracy'] - baseline_metrics['accuracy']) * 100
    print(f"\nAccuracy Improvement: {accuracy_improvement:+.2f}%")
    
    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write("="*70 + "\n")
        f.write("BASELINE vs OPTIMIZED COMPARISON\n")
        f.write("="*70 + "\n\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n" + "="*70 + "\n")
        f.write("SUMMARY\n")
        f.write("="*70 + "\n")
        f.write(f"\nBaseline Model: {baseline_data['model']}\n")
        f.write(f"  Test Size: {baseline_data['test_size']}\n")
        f.write(f"  Accuracy: {baseline_metrics['accuracy']:.4f}\n")
        f.write(f"\nOptimized Model: {optimized_data['model']}\n")
        f.write(f"  Test Size: {optimized_data['test_size']}\n")
        f.write(f"  Training Examples: {optimized_data['train_size']}\n")
        f.write(f"  Max Few-Shot Demos: {optimized_data['max_bootstrapped_demos']}\n")
        f.write(f"  Accuracy: {optimized_metrics['accuracy']:.4f}\n")
        f.write(f"\nAccuracy Improvement: {accuracy_improvement:+.2f}%\n")
    
    print(f"\n✓ Comparison report saved to: {output_path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
