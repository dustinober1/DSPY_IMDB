# Troubleshooting Guide

## JSON Parsing Errors with Small Models

### Problem
When using small models like Qwen 0.6B, you may encounter errors like:
```
Error on example: Adapter JSONAdapter failed to parse the LM response
```

### Root Cause
Small models often don't reliably produce JSON-formatted responses that DSPy's structured adapters expect. They may:
- Return plain text instead of JSON
- Include extra text before/after JSON
- Format JSON incorrectly

### Solution
We've implemented a text-based prompting approach instead of relying on DSPy's structured JSON parsing:

1. **Direct Text Prompting**: Use simple text prompts instead of `dspy.Signature` with typed fields
2. **Robust Response Parsing**: Extract sentiment from various response formats
3. **Fallback Analysis**: If model response is unclear, analyze the review content directly
4. **Error Handling**: Gracefully handle parsing errors with sensible defaults

### Implementation Details

The fixed implementation in `src/sentiment_module.py`:
- Uses direct LM calls: `dspy.settings.lm(prompt)`
- Handles multiple response formats (dict, list, string)
- Implements multi-level sentiment extraction:
  1. Look for explicit "positive"/"negative" in response
  2. Count sentiment words in response
  3. Analyze review content as fallback
  4. Default to "negative" if still unclear

## Low Accuracy / Conservative Predictions

### Problem
The baseline model may show conservative behavior, classifying most reviews as negative.

### Why This Happens
Small models (0.6B parameters) have limited capacity for:
- Understanding complex sentiment
- Following structured output formats
- Nuanced classification

### Expected Baseline Performance
- **Baseline (zero-shot)**: 50-70% accuracy
- **Optimized (DSPy)**: 60-80% accuracy (5-15% improvement)

### How DSPy Helps
DSPy optimization improves performance through:
- **Few-shot examples**: Showing the model good examples
- **Prompt optimization**: Finding better ways to phrase the task
- **Chain-of-thought**: Encouraging step-by-step reasoning

## Running the Optimized Version

To see if DSPy can improve the small model's performance:

```bash
# Run DSPy-optimized version
python run_optimized.py --subset 100 --train-size 50

# Compare results
python compare_results.py
```

The optimized version uses `BootstrapFewShot` to:
1. Select effective training examples
2. Generate few-shot demonstrations
3. Compile an optimized program

## Tips for Better Results

### 1. Increase Training Examples
```bash
python run_optimized.py --subset 200 --train-size 100
```

### 2. Adjust Few-Shot Demos
```bash
python run_optimized.py --subset 100 --train-size 50 --max-bootstrapped-demos 8
```

### 3. Try a Larger Model
If available, use a larger model:
```bash
# Pull a larger model
ollama pull qwen2:1.5b

# Run with larger model
python run_baseline.py --model qwen2:1.5b --subset 100
```

### 4. Experiment with Prompts
Modify the prompts in `src/sentiment_module.py` to:
- Be more explicit about the task
- Include examples in the prompt
- Use different phrasing

## Understanding the Results

### Confusion Matrix
```
TP: True Positives (correctly identified positive reviews)
TN: True Negatives (correctly identified negative reviews)
FP: False Positives (negative reviews classified as positive)
FN: False Negatives (positive reviews classified as negative)
```

### Conservative Classifier
If you see:
- High precision, low recall for one class
- All predictions going to one class

This means the model is being conservative. DSPy optimization should help balance this.

## Next Steps

1. **Run Optimized Version**: See if DSPy improves accuracy
2. **Analyze Errors**: Look at specific examples where the model fails
3. **Iterate on Prompts**: Experiment with different prompt formulations
4. **Try Different Optimizers**: Explore MIPRO or other DSPy optimizers
5. **Scale Up**: Test with larger subsets and more training examples
