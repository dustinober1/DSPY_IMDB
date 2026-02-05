#!/bin/bash

# DSPy IMDB Sentiment Analysis POC - Setup Script

set -e  # Exit on error

echo "================================================"
echo "DSPy IMDB Sentiment Analysis POC - Setup"
echo "================================================"

# Check Python version
echo ""
echo "Checking Python version..."
python3 --version

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Check Ollama installation
echo ""
echo "Checking Ollama installation..."
if command -v ollama &> /dev/null; then
    echo "✓ Ollama is installed"
    ollama --version
else
    echo "✗ Ollama is not installed"
    echo ""
    echo "Please install Ollama from: https://ollama.ai"
    echo "Then run: ollama pull qwen2:0.5b"
    exit 1
fi

# Check if Ollama is running
echo ""
echo "Checking if Ollama is running..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✓ Ollama server is running"
else
    echo "✗ Ollama server is not running"
    echo "Please start Ollama with: ollama serve"
    exit 1
fi

# List available models
echo ""
echo "Available Ollama models:"
curl -s http://localhost:11434/api/tags | python3 -c "import sys, json; models = json.load(sys.stdin).get('models', []); print('\n'.join([f\"  - {m['name']}\" for m in models]) if models else '  (none)')"

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p data
mkdir -p results
mkdir -p logs
mkdir -p docs

echo ""
echo "================================================"
echo "Setup complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Activate the virtual environment: source .venv/bin/activate"
echo "  2. Ensure you have a Qwen model: ollama pull qwen2:0.5b"
echo "  3. Run baseline: python run_baseline.py --subset 100"
echo "  4. Run optimized: python run_optimized.py --subset 100"
echo ""
