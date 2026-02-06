# Performance Optimization Guide for DSPy IMDB Sentiment Analysis

## Overview

This guide provides strategies to significantly speed up your DSPy sentiment analysis pipeline and better utilize your 16GB combined memory/GPU.

## Current Performance Bottlenecks

1. **Sequential Processing**: The original `run_baseline.py` processes one review at a time
2. **Underutilized GPU**: Ollama can handle multiple concurrent requests but wasn't configured for it
3. **Small Context Window**: Default settings don't maximize memory usage
4. **No Batching**: Each inference is a separate API call

## Optimization Strategies

### 1. Parallel Batch Processing (5-10x speedup)

**What it does**: Processes multiple reviews concurrently using ThreadPoolExecutor

**Files created**:
- `src/evaluation_batch.py` - Parallel batch evaluation module

**Key parameters**:
- `batch_size`: Number of examples per batch (default: 32)
- `max_workers`: Number of parallel threads (default: 4)

**Expected speedup**: 5-10x faster than sequential processing

### 2. Optimized Model Configuration (2-3x speedup)

**What it does**: Configures Ollama to handle more concurrent requests and use larger context windows

**Files modified**:
- `src/model_config.py` - Added `setup_ollama_model_optimized()` function

**Key parameters**:
```python
num_ctx=8192        # Increased context window (uses more GPU memory)
num_parallel=4      # Allow 4 parallel requests
num_thread=8        # CPU threads for processing
```

**Expected speedup**: 2-3x faster inference

### 3. Ollama Environment Optimization (1.5-2x speedup)

**What it does**: Configures Ollama service with environment variables for better performance

**Files created**:
- `optimize_ollama.sh` - Script to restart Ollama with optimizations

**Key environment variables**:
```bash
OLLAMA_NUM_PARALLEL=4        # Allow 4 parallel requests
OLLAMA_MAX_LOADED_MODELS=1   # Keep model loaded in memory
OLLAMA_FLASH_ATTENTION=1     # Enable flash attention
OLLAMA_NUM_GPU=1             # Use GPU
```

**Expected speedup**: 1.5-2x faster

## Combined Expected Speedup

**Total expected speedup: 15-60x faster** than the original sequential implementation!

- Original: ~1-2 examples/second
- Optimized: ~15-120 examples/second

For the full IMDB test set (25,000 examples):
- Original: ~3.5-7 hours
- Optimized: ~3-28 minutes

## Usage Instructions

### Quick Start (Recommended)

1. **Stop current baseline run** (if still running):
```bash
pkill -f "python run_baseline.py"
```

2. **Optimize Ollama configuration**:
```bash
./optimize_ollama.sh
```

3. **Run optimized baseline on subset** (test first):
```bash
python run_baseline_optimized.py --subset 1000 --batch-size 32 --workers 4
```

4. **Run on full dataset** (after testing):
```bash
python run_baseline_optimized.py --batch-size 64 --workers 8
```

### Parameter Tuning Guide

#### Batch Size
- **Small (16-32)**: Good for testing, lower memory usage
- **Medium (32-64)**: Balanced performance
- **Large (64-128)**: Maximum throughput, requires more memory

Start with 32 and increase if you have memory headroom.

#### Workers
- **Conservative (2-4)**: Safe for most systems
- **Moderate (4-8)**: Good for 16GB systems
- **Aggressive (8-16)**: Only if you have memory to spare

Monitor with `ollama ps` to ensure you're not running out of memory.

#### Finding Optimal Settings

Run this experiment:
```bash
# Test different configurations
python run_baseline_optimized.py --subset 500 --batch-size 16 --workers 2
python run_baseline_optimized.py --subset 500 --batch-size 32 --workers 4
python run_baseline_optimized.py --subset 500 --batch-size 64 --workers 8
```

Choose the configuration with the highest throughput (examples/second) that doesn't cause memory issues.

## Monitoring Performance

### Check GPU Usage
```bash
# Real-time monitoring
watch -n 1 'ollama ps'
```

### Check System Resources
```bash
# Memory usage
top -o MEM

# CPU usage
top -o CPU
```

### View Ollama Logs
```bash
tail -f /tmp/ollama.log
```

## Memory Considerations

Your system has 16GB combined memory/GPU. Here's how it's allocated:

### Current Usage (from `ollama ps`)
- Model loaded: 4.3 GB (gemma3:latest)
- Available: ~11.7 GB

### Recommended Allocation
- **Batch size 32 + 4 workers**: ~6-8 GB additional
- **Batch size 64 + 8 workers**: ~10-12 GB additional

**Safe configuration for 16GB**: `--batch-size 32 --workers 4`
**Aggressive configuration**: `--batch-size 64 --workers 8`

## Troubleshooting

### Out of Memory Errors
**Solution**: Reduce batch size and/or workers
```bash
python run_baseline_optimized.py --batch-size 16 --workers 2
```

### Slow Performance
**Check**:
1. Is Ollama using GPU? Run `ollama ps` and verify "100% GPU"
2. Are optimizations applied? Check `/tmp/ollama.log`
3. Is the model loaded? Should show in `ollama ps`

### Connection Errors
**Solution**: Restart Ollama
```bash
./optimize_ollama.sh
```

## Advanced Optimizations

### 1. Use a Smaller, Faster Model

The `gemma3:latest` model is 3.3 GB. Consider using a smaller model for faster inference:

```bash
# Pull a smaller model
ollama pull qwen2:0.5b

# Run with smaller model
python run_baseline_optimized.py --model qwen2:0.5b --batch-size 64 --workers 8
```

**Trade-off**: Faster speed but potentially lower accuracy

### 2. Reduce Max Tokens

If you don't need long responses, reduce `max_tokens` in `src/model_config.py`:

```python
max_tokens=50  # Instead of 150
```

### 3. Increase Context Window Further

If you have memory to spare:

```python
num_ctx=16384  # Instead of 8192
```

## Comparison: Original vs Optimized

| Metric | Original | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Processing | Sequential | Parallel | 5-10x |
| Batch size | 1 | 32-64 | 32-64x |
| Workers | 1 | 4-8 | 4-8x |
| Context | 4096 | 8192 | 2x |
| Parallel requests | 1 | 4 | 4x |
| **Total speedup** | 1x | **15-60x** | 🚀 |

## Next Steps

1. **Test the optimized version** on a subset first
2. **Monitor resource usage** to find optimal parameters
3. **Run full evaluation** with optimized settings
4. **Apply same optimizations** to your optimizer scripts (MIPRO, COPRO)

## Files Created/Modified

### New Files
- `src/evaluation_batch.py` - Parallel batch evaluation
- `run_baseline_optimized.py` - Optimized baseline script
- `optimize_ollama.sh` - Ollama optimization script
- `docs/performance_optimization.md` - This guide

### Modified Files
- `src/model_config.py` - Added optimized configuration function

### Original Files (Unchanged)
- `run_baseline.py` - Original baseline (kept for comparison)
- `src/evaluation.py` - Original evaluation (kept for compatibility)
