"""Optimized model configuration for faster inference."""

import dspy
from typing import Optional


def setup_ollama_model_optimized(
    model_name: str = "gemma3:latest",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0,
    max_tokens: int = 150,
    num_ctx: int = 8192,  # Increased context window
    num_parallel: int = 4,  # Allow parallel requests
) -> dspy.LM:
    """
    Configure DSPy to use Ollama with optimized settings for speed.
    
    Args:
        model_name: Name of the Ollama model
        base_url: URL of the Ollama server
        temperature: Sampling temperature (0.0 for deterministic)
        max_tokens: Maximum tokens to generate
        num_ctx: Context window size (higher = more memory but better batching)
        num_parallel: Number of parallel requests to allow
        
    Returns:
        Configured DSPy LM instance
    """
    print(f"Setting up Ollama model with optimized settings: {model_name}")
    print(f"  Base URL: {base_url}")
    print(f"  Temperature: {temperature}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Context window: {num_ctx}")
    print(f"  Parallel requests: {num_parallel}")
    
    # Configure DSPy with Ollama and optimization parameters
    lm = dspy.LM(
        model=f"ollama/{model_name}",
        api_base=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        # Additional Ollama-specific parameters for performance
        model_kwargs={
            "num_ctx": num_ctx,  # Context window
            "num_parallel": num_parallel,  # Parallel processing
            "num_thread": 8,  # CPU threads for processing
        }
    )
    
    # Set as default LM for DSPy
    dspy.configure(lm=lm)
    
    print("✓ Model configured with optimizations")
    return lm


def setup_ollama_model(
    model_name: str = "gemma3:latest",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0,
    max_tokens: int = 150,
) -> dspy.LM:
    """
    Configure DSPy to use Ollama as the language model backend.
    
    Args:
        model_name: Name of the Ollama model (e.g., 'qwen2:0.5b', 'qwen:0.6b')
        base_url: URL of the Ollama server
        temperature: Sampling temperature (0.0 for deterministic)
        max_tokens: Maximum tokens to generate
        
    Returns:
        Configured DSPy LM instance
    """
    print(f"Setting up Ollama model: {model_name}")
    print(f"  Base URL: {base_url}")
    print(f"  Temperature: {temperature}")
    print(f"  Max tokens: {max_tokens}")
    
    # Configure DSPy with Ollama
    lm = dspy.LM(
        model=f"ollama/{model_name}",
        api_base=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    # Set as default LM for DSPy
    dspy.configure(lm=lm)
    
    print("✓ Model configured successfully")
    return lm


def test_model_connection(lm: Optional[dspy.LM] = None) -> bool:
    """
    Test the connection to the Ollama model.
    
    Args:
        lm: Optional LM instance to test. If None, uses configured default.
        
    Returns:
        True if connection successful, False otherwise
    """
    try:
        if lm is None:
            lm = dspy.settings.lm
            
        # Try a simple generation
        response = lm("Say 'OK' if you can read this.")
        print(f"✓ Model connection test successful")
        print(f"  Response: {response[:100]}...")
        return True
    except Exception as e:
        print(f"✗ Model connection test failed: {e}")
        return False
