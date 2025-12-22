# Installation Guide

## Requirements

- Python 3.8 or higher
- pip or conda package manager
- API key for OpenAI or Anthropic (for LLM representations)

## Basic Installation

### Using pip

```bash
pip install refrag
```

### With specific LLM provider

```bash
# OpenAI only
pip install "refrag[openai]"

# Anthropic only
pip install "refrag[anthropic]"

# Both providers
pip install "refrag[all]"
```

### From source (for development)

```bash
git clone https://github.com/yourusername/refrag.git
cd refrag
pip install -e ".[dev]"
```

## API Key Setup

### OpenAI

Get your API key from [platform.openai.com](https://platform.openai.com)

```bash
# Set environment variable (Linux/Mac)
export OPENAI_API_KEY="sk-..."

# Or in Python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
```

### Anthropic

Get your API key from [console.anthropic.com](https://console.anthropic.com)

```bash
# Set environment variable
export ANTHROPIC_API_KEY="sk-ant-..."

# Or in Python
import os
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
```

## Verify Installation

```python
import refrag
print(refrag.__version__)

# Test basic functionality
from refrag import REFRAGRetriever

retriever = REFRAGRetriever(
    llm_provider="openai",
    llm_model="gpt-4o-mini"
)
print("✓ REFRAG installed successfully")
```

## Optional Dependencies

### GPU Support (for faster embeddings)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Development Tools

```bash
pip install -e ".[dev]"
# Includes: pytest, black, flake8, jupyter
```

## Troubleshooting

### ImportError: No module named 'sentence_transformers'

```bash
pip install sentence-transformers
```

### OpenAI/Anthropic API errors

Verify your API key is set:

```python
import os
print(os.getenv("OPENAI_API_KEY"))  # Should print your key
```

### CUDA/GPU issues

For CPU-only mode:

```python
retriever = REFRAGRetriever(
    device="cpu"  # Force CPU usage
)
```

## Next Steps

- **[Quick Start →](quickstart.md)** - Build your first REFRAG system
- **[How It Works →](how-it-works.md)** - Understand the architecture
- **[API Reference →](api-reference.md)** - Detailed API documentation
