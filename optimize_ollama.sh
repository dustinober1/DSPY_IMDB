#!/bin/bash

# Ollama Performance Optimization Script
# This script configures Ollama to use more GPU memory and handle more concurrent requests

echo "================================================"
echo "Ollama Performance Optimization"
echo "================================================"

# Stop Ollama if running
echo "Stopping Ollama service..."
pkill ollama 2>/dev/null || true
sleep 2

# Set environment variables for better performance
export OLLAMA_NUM_PARALLEL=4        # Allow 4 parallel requests
export OLLAMA_MAX_LOADED_MODELS=1   # Keep 1 model loaded (saves memory)
export OLLAMA_FLASH_ATTENTION=1     # Enable flash attention for speed
export OLLAMA_NUM_GPU=1             # Use 1 GPU

echo "Starting Ollama with optimized settings..."
echo "  OLLAMA_NUM_PARALLEL=4"
echo "  OLLAMA_MAX_LOADED_MODELS=1"
echo "  OLLAMA_FLASH_ATTENTION=1"
echo "  OLLAMA_NUM_GPU=1"

# Start Ollama in the background
nohup ollama serve > /tmp/ollama.log 2>&1 &

sleep 3

# Verify Ollama is running
if pgrep -x "ollama" > /dev/null; then
    echo "✓ Ollama started successfully with optimizations"
    echo ""
    echo "To view logs: tail -f /tmp/ollama.log"
else
    echo "✗ Failed to start Ollama"
    exit 1
fi

# Load the model with optimized parameters
echo ""
echo "Loading gemma3:latest with optimized parameters..."
ollama run gemma3:latest --verbose <<EOF
exit
EOF

echo ""
echo "================================================"
echo "Optimization complete!"
echo "================================================"
echo ""
echo "Recommended next steps:"
echo "1. Run: python run_baseline_optimized.py --subset 1000 --batch-size 32 --workers 4"
echo "2. Monitor GPU usage: watch -n 1 'ollama ps'"
echo "3. For full dataset: python run_baseline_optimized.py --batch-size 64 --workers 8"
