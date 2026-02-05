# Running DSPy IMDB POC with Gemma3

All scripts have been updated to use `gemma3:latest` as the default model.

## Quick Start (Recommended Order)

### 1. Test with Small Subset First
```bash
# Baseline (zero-shot) - should get ~75-80% accuracy
python run_baseline.py --subset 100

# Optimized with BootstrapFewShot - should get ~80-85% accuracy
python run_optimized.py --subset 100 --train-size 50

# Compare the results
python compare_results.py
```

### 2. Medium-Sized Test (More Reliable)
```bash
# Baseline with 500 examples
python run_baseline.py --subset 500

# Optimized with more training data
python run_optimized.py --subset 500 --train-size 200

# Compare
python compare_results.py
```

### 3. Full Dataset (Takes Hours)
```bash
# Baseline on full 25K test set
python run_baseline.py

# Optimized with substantial training
python run_optimized.py --train-size 500

# Compare
python compare_results.py
```

## Expected Results with Gemma3

**Baseline (Zero-shot):**
- Accuracy: 75-80%
- Should classify both positive and negative reviews

**Optimized (BootstrapFewShot):**
- Accuracy: 80-85%
- 5-10% improvement over baseline
- Better balanced predictions

**Advanced Optimizers (MIPRO/COPRO):**
- May or may not work better than BootstrapFewShot
- Worth trying on medium subsets (500-1000 examples)
- Will take significantly longer

## Troubleshooting

**If you get connection errors:**
```bash
# Check Ollama is running
ollama list

# Verify gemma3 is available
ollama run gemma3:latest "Hello"
```

**If accuracy is still low:**
- Make sure you're using enough training examples (--train-size 50+)
- Try increasing subset size for more reliable metrics
- Check that the model is actually gemma3 in the output

## Advanced Options

**Try different optimizers:**
```bash
# MIPRO (prompt optimization)
python run_mipro.py --subset 200 --train-size 100 --num-candidates 3

# COPRO (coordinate optimization)
python run_copro.py --subset 200 --train-size 100 --depth 2 --breadth 3
```

**Compare all approaches:**
```bash
python compare_all.py
```

## What to Expect

With Gemma3, you should see:
- ✅ Much better baseline accuracy (75-80% vs 67% with Qwen)
- ✅ Consistent predictions (not all negative)
- ✅ DSPy optimization actually working
- ✅ 5-10% improvement from optimization
- ✅ Faster inference than larger models
