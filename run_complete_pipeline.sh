#!/bin/bash
# Complete DSPy IMDB Sentiment Analysis Pipeline with Gemma3

echo "============================================================"
echo "DSPy IMDB SENTIMENT ANALYSIS - COMPLETE PIPELINE"
echo "============================================================"
echo ""
echo "Using model: gemma3:latest"
echo "Dataset: IMDB (500 training examples, 150 test examples)"
echo ""

# Step 1: Baseline
echo "Step 1/3: Running baseline (zero-shot)..."
echo "Expected: ~85-90% accuracy"
python run_baseline.py --subset 500
echo ""

# Step 2: Optimized with BootstrapFewShot
echo "Step 2/3: Running DSPy optimization (BootstrapFewShot)..."
echo "Expected: ~90-95% accuracy (5-10% improvement)"
python run_optimized.py --subset 500 --train-size 200 --max-bootstrapped-demos 6
echo ""

# Step 3: Compare results
echo "Step 3/3: Comparing results..."
python compare_results.py
echo ""

echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"
echo ""
echo "Results saved in results/ directory"
echo "Check results/comparison_report.txt for detailed analysis"
echo ""
