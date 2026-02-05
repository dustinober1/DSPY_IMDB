"""Model configuration for Ollama integration with DSPy."""

import dspy
from typing import Optional


def setup_ollama_model(
    model_name: str = "jewelzufo/Qwen3-0.6B-GGUF:IQ4_NL",
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
