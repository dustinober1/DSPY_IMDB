"""Compare results from different optimization approaches."""

import argparse
import json
from pathlib import Path
import pandas as pd

from src.evaluation import compare_results


def load_results(filepath):
    """Load results from JSON file."""
    try:
        with open(filepath) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Compare all optimization approaches"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing results (default: results)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/full_comparison.txt",
        help="Path to save comparison report (default: results/full_comparison.txt)"
    )
    
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    
    print("="*70)
    print("COMPREHENSIVE OPTIMIZATION COMPARISON")
    print("="*70)
    
    # Load all results
    approaches = {
        "Baseline": results_dir / "baseline_latest.json",
        "BootstrapFewShot": results_dir / "optimized_latest.json",
        "MIPRO": results_dir / "mipro_latest.json",
        "COPRO": results_dir / "copro_latest.json"
    }
    
    results_data = {}
    for name, filepath in approaches.items():
        data = load_results(filepath)
        if data:
            results_data[name] = data
            print(f"✓ Loaded {name} results")
        else:
            print(f"✗ {name} results not found at {filepath}")
    
    if len(results_data) < 2:
        print("\nError: Need at least 2 result files to compare")
        print("Please run the optimization scripts first:")
        print("  python run_baseline.py --subset 100")
        print("  python run_optimized.py --subset 100")
        print("  python run_mipro.py --subset 100")
        print("  python run_copro.py --subset 100")
        return
    
    # Create comparison table
    print("\n" + "="*70)
    print("ACCURACY COMPARISON")
    print("="*70)
    
    comparison_data = []
    for name, data in results_data.items():
        metrics = data["metrics"]
        comparison_data.append({
            "Approach": name,
            "Accuracy": f"{metrics['accuracy']:.4f}",
            "Pos F1": f"{metrics['positive_f1']:.4f}",
            "Neg F1": f"{metrics['negative_f1']:.4f}",
            "Test Size": data.get("test_size", "N/A"),
            "Train Size": data.get("train_size", "N/A")
        })
    
    df = pd.DataFrame(comparison_data)
    print("\n" + df.to_string(index=False))
    
    # Calculate improvements over baseline
    if "Baseline" in results_data:
        print("\n" + "="*70)
        print("IMPROVEMENT OVER BASELINE")
        print("="*70)
        
        baseline_acc = results_data["Baseline"]["metrics"]["accuracy"]
        
        improvements = []
        for name, data in results_data.items():
            if name != "Baseline":
                acc = data["metrics"]["accuracy"]
                improvement = (acc - baseline_acc) * 100
                improvements.append({
                    "Approach": name,
                    "Baseline Acc": f"{baseline_acc:.4f}",
                    "Optimized Acc": f"{acc:.4f}",
                    "Improvement": f"{improvement:+.2f}%"
                })
        
        if improvements:
            imp_df = pd.DataFrame(improvements)
            print("\n" + imp_df.to_string(index=False))
    
    # Detailed metrics comparison
    print("\n" + "="*70)
    print("DETAILED METRICS")
    print("="*70)
    
    for name, data in results_data.items():
        metrics = data["metrics"]
        print(f"\n{name}:")
        print(f"  Accuracy:          {metrics['accuracy']:.4f}")
        print(f"  Positive Precision: {metrics['positive_precision']:.4f}")
        print(f"  Positive Recall:    {metrics['positive_recall']:.4f}")
        print(f"  Positive F1:        {metrics['positive_f1']:.4f}")
        print(f"  Negative Precision: {metrics['negative_precision']:.4f}")
        print(f"  Negative Recall:    {metrics['negative_recall']:.4f}")
        print(f"  Negative F1:        {metrics['negative_f1']:.4f}")
        
        cm = metrics['confusion_matrix']
        print(f"  Confusion Matrix:")
        print(f"    TP: {cm['tp']:4d}  FP: {cm['fp']:4d}")
        print(f"    FN: {cm['fn']:4d}  TN: {cm['tn']:4d}")
    
    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write("="*70 + "\n")
        f.write("COMPREHENSIVE OPTIMIZATION COMPARISON\n")
        f.write("="*70 + "\n\n")
        
        f.write("ACCURACY COMPARISON\n")
        f.write("="*70 + "\n")
        f.write(df.to_string(index=False) + "\n")
        
        if "Baseline" in results_data and improvements:
            f.write("\n" + "="*70 + "\n")
            f.write("IMPROVEMENT OVER BASELINE\n")
            f.write("="*70 + "\n")
            f.write(imp_df.to_string(index=False) + "\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("DETAILED METRICS\n")
        f.write("="*70 + "\n")
        
        for name, data in results_data.items():
            metrics = data["metrics"]
            f.write(f"\n{name}:\n")
            f.write(f"  Accuracy:          {metrics['accuracy']:.4f}\n")
            f.write(f"  Positive Precision: {metrics['positive_precision']:.4f}\n")
            f.write(f"  Positive Recall:    {metrics['positive_recall']:.4f}\n")
            f.write(f"  Positive F1:        {metrics['positive_f1']:.4f}\n")
            f.write(f"  Negative Precision: {metrics['negative_precision']:.4f}\n")
            f.write(f"  Negative Recall:    {metrics['negative_recall']:.4f}\n")
            f.write(f"  Negative F1:        {metrics['negative_f1']:.4f}\n")
            
            cm = metrics['confusion_matrix']
            f.write(f"  Confusion Matrix:\n")
            f.write(f"    TP: {cm['tp']:4d}  FP: {cm['fp']:4d}\n")
            f.write(f"    FN: {cm['fn']:4d}  TN: {cm['tn']:4d}\n")
    
    print(f"\n✓ Full comparison report saved to: {output_path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
