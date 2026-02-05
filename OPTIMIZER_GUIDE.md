# Quick Start Guide for Optimizer Comparison

## Running All Optimizers

To compare MIPRO, COPRO, and BootstrapFewShot against the baseline:

### 1. Baseline (already done)
```bash
python run_baseline.py --subset 100
```
✓ Results: 66.7% accuracy (conservative, all negative predictions)

### 2. BootstrapFewShot (few-shot learning)
```bash
python run_optimized.py --subset 100 --train-size 50
```
Expected: 5-10% improvement through few-shot examples

### 3. MIPRO (prompt optimization)
```bash
python run_mipro.py --subset 100 --train-size 50 --num-candidates 5
```
Expected: 10-15% improvement through optimized prompts

### 4. COPRO (coordinate optimization)
```bash
python run_copro.py --subset 100 --train-size 50 --depth 3 --breadth 5
```
Expected: 10-15% improvement through coordinate descent

### 5. Compare All Results
```bash
python compare_all.py
```

## What Each Optimizer Does

### BootstrapFewShot
- Selects good training examples
- Creates few-shot demonstrations
- Fast and simple

### MIPRO
- Generates multiple prompt candidates
- Tests each candidate
- Selects the best performing prompt
- More sophisticated than BootstrapFewShot

### COPRO
- Uses coordinate descent optimization
- Iteratively improves prompts
- Explores breadth of variations at each depth
- Good for finding optimal prompt structure

## Tips for Best Results

1. **Start Small**: Test with `--subset 100` first
2. **Increase Training**: More training examples = better optimization
3. **Be Patient**: MIPRO and COPRO take longer than BootstrapFewShot
4. **Compare**: Run all approaches and use `compare_all.py`

## Expected Timeline

- Baseline: ~30 seconds
- BootstrapFewShot: ~2-3 minutes
- MIPRO: ~5-10 minutes (depends on num_candidates)
- COPRO: ~5-10 minutes (depends on depth × breadth)

## Troubleshooting

If MIPRO or COPRO fail:
- They may not be available in your DSPy version
- Scripts will automatically fall back to BootstrapFewShot
- You'll still get optimized results, just with a simpler method
