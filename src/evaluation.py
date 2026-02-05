"""Evaluation metrics and utilities for sentiment analysis."""

from typing import List, Dict, Tuple
import pandas as pd
from collections import defaultdict
from src.data_loader import IMDBExample
from src.sentiment_module import normalize_sentiment


def calculate_metrics(
    predictions: List[str],
    ground_truth: List[str]
) -> Dict[str, float]:
    """
    Calculate accuracy and per-class metrics.
    
    Args:
        predictions: List of predicted sentiments
        ground_truth: List of true sentiments
        
    Returns:
        Dictionary with accuracy, precision, recall, and F1 scores
    """
    # Normalize predictions
    predictions = [normalize_sentiment(p) for p in predictions]
    ground_truth = [normalize_sentiment(g) for g in ground_truth]
    
    # Calculate confusion matrix
    tp_pos = sum(1 for p, g in zip(predictions, ground_truth) 
                 if p == "positive" and g == "positive")
    tn_neg = sum(1 for p, g in zip(predictions, ground_truth) 
                 if p == "negative" and g == "negative")
    fp_pos = sum(1 for p, g in zip(predictions, ground_truth) 
                 if p == "positive" and g == "negative")
    fn_pos = sum(1 for p, g in zip(predictions, ground_truth) 
                 if p == "negative" and g == "positive")
    
    # Overall accuracy
    accuracy = (tp_pos + tn_neg) / len(predictions) if predictions else 0.0
    
    # Positive class metrics
    precision_pos = tp_pos / (tp_pos + fp_pos) if (tp_pos + fp_pos) > 0 else 0.0
    recall_pos = tp_pos / (tp_pos + fn_pos) if (tp_pos + fn_pos) > 0 else 0.0
    f1_pos = (2 * precision_pos * recall_pos / (precision_pos + recall_pos) 
              if (precision_pos + recall_pos) > 0 else 0.0)
    
    # Negative class metrics
    precision_neg = tn_neg / (tn_neg + fn_pos) if (tn_neg + fn_pos) > 0 else 0.0
    recall_neg = tn_neg / (tn_neg + fp_pos) if (tn_neg + fp_pos) > 0 else 0.0
    f1_neg = (2 * precision_neg * recall_neg / (precision_neg + recall_neg) 
              if (precision_neg + recall_neg) > 0 else 0.0)
    
    return {
        "accuracy": accuracy,
        "positive_precision": precision_pos,
        "positive_recall": recall_pos,
        "positive_f1": f1_pos,
        "negative_precision": precision_neg,
        "negative_recall": recall_neg,
        "negative_f1": f1_neg,
        "confusion_matrix": {
            "tp": tp_pos,
            "tn": tn_neg,
            "fp": fp_pos,
            "fn": fn_pos
        }
    }


def evaluate_model(
    model,
    test_data: List[IMDBExample],
    verbose: bool = True
) -> Tuple[Dict[str, float], List[str]]:
    """
    Evaluate a sentiment model on test data.
    
    Args:
        model: DSPy sentiment module
        test_data: List of test examples
        verbose: Whether to print progress
        
    Returns:
        Tuple of (metrics dict, list of predictions)
    """
    predictions = []
    ground_truth = []
    
    if verbose:
        from tqdm import tqdm
        iterator = tqdm(test_data, desc="Evaluating")
    else:
        iterator = test_data
    
    for example in iterator:
        try:
            pred = model(review=example.text)
            predictions.append(pred.sentiment)
            ground_truth.append(example.label)
        except Exception as e:
            if verbose:
                print(f"\nError on example: {e}")
            # Default to negative on error
            predictions.append("negative")
            ground_truth.append(example.label)
    
    metrics = calculate_metrics(predictions, ground_truth)
    
    if verbose:
        print_metrics(metrics)
    
    return metrics, predictions


def print_metrics(metrics: Dict[str, float]):
    """Pretty print evaluation metrics."""
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("\nPositive Class:")
    print(f"  Precision: {metrics['positive_precision']:.4f}")
    print(f"  Recall:    {metrics['positive_recall']:.4f}")
    print(f"  F1 Score:  {metrics['positive_f1']:.4f}")
    print("\nNegative Class:")
    print(f"  Precision: {metrics['negative_precision']:.4f}")
    print(f"  Recall:    {metrics['negative_recall']:.4f}")
    print(f"  F1 Score:  {metrics['negative_f1']:.4f}")
    print("\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"  TP: {cm['tp']:4d}  FP: {cm['fp']:4d}")
    print(f"  FN: {cm['fn']:4d}  TN: {cm['tn']:4d}")
    print("="*50 + "\n")


def compare_results(
    baseline_metrics: Dict[str, float],
    optimized_metrics: Dict[str, float]
) -> pd.DataFrame:
    """
    Compare baseline and optimized model results.
    
    Args:
        baseline_metrics: Metrics from baseline model
        optimized_metrics: Metrics from optimized model
        
    Returns:
        DataFrame with comparison
    """
    comparison = {
        "Metric": [
            "Accuracy",
            "Positive Precision",
            "Positive Recall",
            "Positive F1",
            "Negative Precision",
            "Negative Recall",
            "Negative F1"
        ],
        "Baseline": [
            baseline_metrics["accuracy"],
            baseline_metrics["positive_precision"],
            baseline_metrics["positive_recall"],
            baseline_metrics["positive_f1"],
            baseline_metrics["negative_precision"],
            baseline_metrics["negative_recall"],
            baseline_metrics["negative_f1"]
        ],
        "Optimized": [
            optimized_metrics["accuracy"],
            optimized_metrics["positive_precision"],
            optimized_metrics["positive_recall"],
            optimized_metrics["positive_f1"],
            optimized_metrics["negative_precision"],
            optimized_metrics["negative_recall"],
            optimized_metrics["negative_f1"]
        ]
    }
    
    df = pd.DataFrame(comparison)
    df["Improvement"] = df["Optimized"] - df["Baseline"]
    df["Improvement %"] = (df["Improvement"] / df["Baseline"] * 100).round(2)
    
    return df
