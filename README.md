# DSPy IMDB Sentiment Analysis POC

A proof of concept demonstrating DSPy's ability to optimize small language models for sentiment analysis using the Stanford IMDB movie review dataset.

## Overview

This project compares:
- **Baseline**: Zero-shot sentiment classification with Ollama Qwen 0.6B
- **Optimized**: DSPy-optimized few-shot learning with automatic prompt engineering

The goal is to determine if a small model (0.6B parameters) can achieve competitive performance when optimized with DSPy's techniques.

## Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai) installed and running
- At least 4GB of free disk space (for dataset caching)

## Quick Start

### 1. Setup

```bash
# Clone or navigate to the repository
cd /Users/dustinober/Projects/DSPY_IMDB

# Run the setup script
bash scripts/setup.sh

# Activate the virtual environment
source .venv/bin/activate

# Pull the Qwen model (if not already available)
ollama pull qwen2:0.5b
```

### 2. Run Baseline Evaluation

```bash
# Run on a subset for quick testing
python run_baseline.py --subset 100

# Or run on full test set (slower)
python run_baseline.py
```

### 3. Run Optimized Evaluation

```bash
# Run with DSPy optimization
python run_optimized.py --subset 100 --train-size 50

# Or with more training examples for better optimization
python run_optimized.py --subset 500 --train-size 200
```

### 4. Compare Results

```bash
python compare_results.py
```

## Project Structure

```
DSPY_IMDB/
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # IMDB dataset loading and preprocessing
│   ├── model_config.py     # Ollama model configuration
│   ├── sentiment_module.py # DSPy sentiment analysis modules
│   └── evaluation.py       # Metrics and evaluation utilities
├── scripts/
│   └── setup.sh           # Automated setup script
├── run_baseline.py        # Baseline evaluation script
├── run_optimized.py       # DSPy-optimized evaluation script
├── compare_results.py     # Results comparison script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Usage Examples

### Custom Model

```bash
# Use a different Ollama model
python run_baseline.py --model llama2:7b --subset 100
python run_optimized.py --model llama2:7b --subset 100
```

### Adjust Optimization Parameters

```bash
# Use more few-shot examples
python run_optimized.py --subset 200 --train-size 100 --max-bootstrapped-demos 8
```

### Try Advanced Optimizers

**MIPRO (Multi-prompt Instruction Proposal Optimizer)**
```bash
# MIPRO generates and tests multiple prompt variations
python run_mipro.py --subset 100 --train-size 50 --num-candidates 5
```

**COPRO (Coordinate Prompt Optimization)**
```bash
# COPRO uses coordinate descent for prompt optimization
python run_copro.py --subset 100 --train-size 50 --depth 3 --breadth 5
```

**Compare All Approaches**
```bash
# After running baseline, optimized, MIPRO, and COPRO
python compare_all.py
```

### Full Dataset Evaluation

```bash
# Warning: This will take significant time
python run_baseline.py
python run_optimized.py --train-size 500
```

## How It Works

### Baseline Approach
1. Loads IMDB dataset
2. Uses zero-shot prompting with the small LM
3. Evaluates accuracy on test set

### Optimized Approach
1. Loads IMDB dataset
2. Uses DSPy's `BootstrapFewShot` optimizer to:
   - Select effective few-shot examples
   - Optimize prompt structure
   - Generate chain-of-thought reasoning
3. Compiles optimized program
4. Evaluates on test set

### DSPy Optimization

DSPy automatically:
- Selects the most informative training examples
- Optimizes prompt templates
- Generates effective few-shot demonstrations
- Implements chain-of-thought reasoning

## Expected Results

With a small model like Qwen 0.6B:
- **Baseline**: 50-70% accuracy (depending on model capability)
- **Optimized**: 5-15% improvement over baseline

The improvement demonstrates DSPy's ability to enhance small model performance through:
- Better prompt engineering
- Strategic few-shot example selection
- Structured reasoning patterns

## Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if not running
ollama serve
```

### Model Not Found

```bash
# List available models
ollama list

# Pull the required model
ollama pull qwen2:0.5b
```

### Memory Issues

If you encounter memory issues with the full dataset:
- Use `--subset` parameter to limit dataset size
- Reduce `--train-size` for optimization
- Use a smaller model

### Slow Performance

- Start with small subsets (--subset 50)
- Reduce training examples (--train-size 20)
- Use a faster model if available

## Further Exploration

### Try Different Optimizers

Modify `run_optimized.py` to use other DSPy optimizers:
- `MIPRO`: More sophisticated optimization
- `BayesianSignatureOptimizer`: Bayesian approach
- `SignatureOptimizer`: Prompt optimization

### Experiment with Modules

Try different DSPy modules in `src/sentiment_module.py`:
- `ReAct`: Reasoning and acting
- `ProgramOfThought`: Structured reasoning
- Custom multi-step pipelines

## Documentation

See the `docs/` directory for additional documentation:
- `SETUP.md`: Detailed setup instructions
- `RESULTS.md`: Template for documenting your results

## License

This is a proof of concept project for educational purposes.

## References

- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [Ollama](https://ollama.ai)
- [IMDB Dataset](https://huggingface.co/datasets/stanfordnlp/imdb)
