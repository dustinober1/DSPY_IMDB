"""Batch evaluation for faster processing."""

from typing import List, Dict, Tuple
import dspy
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.sentiment_module import normalize_sentiment


def calculate_metrics(
    predictions: List[str],
    ground_truth: List[str]
) -> Dict[str, float]:
    """Calculate accuracy and per-class metrics."""
    predictions = [normalize_sentiment(p) for p in predictions]
    ground_truth = [normalize_sentiment(g) for g in ground_truth]
    
    tp_pos = sum(1 for p, g in zip(predictions, ground_truth) 
                 if p == "positive" and g == "positive")
    tn_neg = sum(1 for p, g in zip(predictions, ground_truth) 
                 if p == "negative" and g == "negative")
    fp_pos = sum(1 for p, g in zip(predictions, ground_truth) 
                 if p == "positive" and g == "negative")
    fn_pos = sum(1 for p, g in zip(predictions, ground_truth) 
                 if p == "negative" and g == "positive")
    
    accuracy = (tp_pos + tn_neg) / len(predictions) if predictions else 0.0
    
    precision_pos = tp_pos / (tp_pos + fp_pos) if (tp_pos + fp_pos) > 0 else 0.0
    recall_pos = tp_pos / (tp_pos + fn_pos) if (tp_pos + fn_pos) > 0 else 0.0
    f1_pos = (2 * precision_pos * recall_pos / (precision_pos + recall_pos) 
              if (precision_pos + recall_pos) > 0 else 0.0)
    
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


def evaluate_batch(
    model,
    batch: List,
    batch_idx: int
) -> Tuple[List[str], List[str]]:
    """Evaluate a batch of examples."""
    predictions = []
    ground_truth = []
    
    for example in batch:
        try:
            review_text = example.text if hasattr(example, 'text') else str(example)
            pred = model(review=review_text)
            predictions.append(pred.sentiment)
            true_label = example.label if hasattr(example, 'label') else 'negative'
            ground_truth.append(true_label)
        except Exception as e:
            predictions.append("negative")
            true_label = example.label if hasattr(example, 'label') else 'negative'
            ground_truth.append(true_label)
    
    return predictions, ground_truth


def evaluate_model_parallel(
    model,
    test_data: List,
    batch_size: int = 32,
    max_workers: int = 4,
    verbose: bool = True
) -> Tuple[Dict[str, float], List[str]]:
    """
    Evaluate a sentiment model on test data using parallel batch processing.
    
    Args:
        model: DSPy sentiment module
        test_data: List of test examples
        batch_size: Number of examples per batch
        max_workers: Number of parallel workers
        verbose: Whether to print progress
        
    Returns:
        Tuple of (metrics dict, list of predictions)
    """
    all_predictions = []
    all_ground_truth = []
    
    # Split data into batches
    batches = [test_data[i:i + batch_size] for i in range(0, len(test_data), batch_size)]
    
    if verbose:
        print(f"Processing {len(test_data)} examples in {len(batches)} batches with {max_workers} workers")
    
    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(evaluate_batch, model, batch, idx): idx 
            for idx, batch in enumerate(batches)
        }
        
        if verbose:
            pbar = tqdm(total=len(batches), desc="Evaluating batches")
        
        for future in as_completed(futures):
            predictions, ground_truth = future.result()
            all_predictions.extend(predictions)
            all_ground_truth.extend(ground_truth)
            if verbose:
                pbar.update(1)
        
        if verbose:
            pbar.close()
    
    metrics = calculate_metrics(all_predictions, all_ground_truth)
    
    if verbose:
        print_metrics(metrics)
    
    return metrics, all_predictions


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
