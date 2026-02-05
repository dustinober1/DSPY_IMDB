# Experimental Results

Document your experimental results here.

## Experiment 1: Initial Baseline

**Date:** _[Add date]_

**Configuration:**
- Model: qwen2:0.5b
- Test Size: _[Add size]_
- Subset: _[Add if used]_

**Baseline Results:**
```
Accuracy: __%
Positive F1: __
Negative F1: __
```

**Observations:**
- _[Add your observations]_

---

## Experiment 2: DSPy Optimization

**Date:** _[Add date]_

**Configuration:**
- Model: qwen2:0.5b
- Test Size: _[Add size]_
- Training Examples: _[Add number]_
- Max Few-Shot Demos: _[Add number]_

**Optimized Results:**
```
Accuracy: __%
Positive F1: __
Negative F1: __
```

**Improvement:**
```
Accuracy Improvement: +__%
```

**Observations:**
- _[Add your observations]_
- _[Note any interesting patterns]_
- _[Document failure cases]_

---

## Experiment 3: [Custom Experiment]

**Date:** _[Add date]_

**Configuration:**
- _[Add your configuration]_

**Results:**
- _[Add your results]_

**Observations:**
- _[Add your observations]_

---

## Key Findings

### What Worked Well
- _[Document successful approaches]_

### Challenges
- _[Document difficulties encountered]_

### Lessons Learned
- _[Document insights gained]_

### Future Directions
- _[Document potential improvements]_

---

## Sample Predictions

### Example 1: Correct Positive Classification

**Review:**
```
[Add review text]
```

**Baseline:** [prediction]
**Optimized:** [prediction]
**Ground Truth:** positive

---

### Example 2: Correct Negative Classification

**Review:**
```
[Add review text]
```

**Baseline:** [prediction]
**Optimized:** [prediction]
**Ground Truth:** negative

---

### Example 3: Baseline Error, Optimized Correct

**Review:**
```
[Add review text]
```

**Baseline:** [wrong prediction]
**Optimized:** [correct prediction]
**Ground Truth:** [actual label]

**Analysis:** _[Why did optimization help?]_

---

### Example 4: Both Models Incorrect

**Review:**
```
[Add review text]
```

**Baseline:** [wrong prediction]
**Optimized:** [wrong prediction]
**Ground Truth:** [actual label]

**Analysis:** _[Why did both fail?]_

---

## Performance Analysis

### Accuracy by Review Length

| Length Range | Baseline Acc | Optimized Acc | Improvement |
|--------------|--------------|---------------|-------------|
| Short (<100) | __% | __% | +__% |
| Medium (100-500) | __% | __% | +__% |
| Long (>500) | __% | __% | +__% |

### Common Error Patterns

1. **Sarcasm/Irony**
   - _[Document how models handle sarcastic reviews]_

2. **Mixed Sentiment**
   - _[Document how models handle reviews with both positive and negative aspects]_

3. **Technical Language**
   - _[Document how models handle film terminology]_

---

## Conclusion

_[Summarize your overall findings about using DSPy with small models for sentiment analysis]_
