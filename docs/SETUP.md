# Setup Guide

Detailed instructions for setting up the DSPy IMDB Sentiment Analysis POC.

## Prerequisites

### 1. Python Installation

Ensure you have Python 3.8 or higher installed:

```bash
python3 --version
```

If not installed, download from [python.org](https://www.python.org/downloads/).

### 2. Ollama Installation

#### macOS

```bash
# Download and install from the website
open https://ollama.ai/download

# Or use Homebrew
brew install ollama
```

#### Linux

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

#### Verify Installation

```bash
ollama --version
```

### 3. Start Ollama Server

```bash
# Start the Ollama server
ollama serve
```

Keep this terminal open, or run Ollama as a background service.

### 4. Pull Qwen Model

In a new terminal:

```bash
# Pull the Qwen 0.5B model
ollama pull qwen2:0.5b

# Verify the model is available
ollama list
```

## Project Setup

### 1. Navigate to Project Directory

```bash
cd /Users/dustinober/Projects/DSPY_IMDB
```

### 2. Run Automated Setup

```bash
bash scripts/setup.sh
```

This script will:
- Create a Python virtual environment
- Install all required dependencies
- Verify Ollama installation
- Check for available models
- Create necessary directories

### 3. Manual Setup (Alternative)

If you prefer manual setup:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data results logs docs compiled_programs
```

## Verify Installation

### Test Data Loading

```bash
source .venv/bin/activate
python -c "from src.data_loader import load_imdb_data; train, dev, test = load_imdb_data(subset_size=10); print(f'✓ Loaded {len(train)} train, {len(dev)} dev, {len(test)} test examples')"
```

### Test Model Connection

```bash
python -c "from src.model_config import setup_ollama_model, test_model_connection; lm = setup_ollama_model(); test_model_connection(lm)"
```

### Quick Test Run

```bash
# Run baseline on 10 examples
python run_baseline.py --subset 10
```

If all tests pass, you're ready to go!

## Troubleshooting

### Virtual Environment Issues

If activation fails:

```bash
# Recreate the virtual environment
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Ollama Connection Failed

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it
ollama serve
```

### Model Not Available

```bash
# List available models
ollama list

# Pull the model if missing
ollama pull qwen2:0.5b

# Or try an alternative model
ollama pull qwen:0.6b
```

### Dataset Download Issues

The IMDB dataset will be automatically downloaded on first use. If download fails:

```bash
# Clear cache and retry
rm -rf data/
python run_baseline.py --subset 10
```

### Permission Issues

```bash
# Make setup script executable
chmod +x scripts/setup.sh

# Run with bash explicitly
bash scripts/setup.sh
```

## Next Steps

Once setup is complete:

1. Read the main [README.md](../README.md) for usage instructions
2. Run baseline evaluation: `python run_baseline.py --subset 100`
3. Run optimized evaluation: `python run_optimized.py --subset 100`
4. Compare results: `python compare_results.py`

## Environment Variables (Optional)

Create a `.env` file for custom configuration:

```bash
# .env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2:0.5b
DATA_CACHE_DIR=./data
RESULTS_DIR=./results
```

Load in your scripts:

```python
from dotenv import load_dotenv
load_dotenv()
```
