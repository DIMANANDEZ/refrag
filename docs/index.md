# REFRAG Documentation

Welcome to REFRAG - Representation-Focused Retrieval Augmented Generation.

## What is REFRAG?

REFRAG improves retrieval quality in RAG systems by using LLM-generated representations instead of raw text chunks. This results in better semantic matching and more efficient context usage.

## Quick Links

- **[Installation](installation.md)** - Get REFRAG set up
- **[Quick Start](quickstart.md)** - Working example in 5 minutes
- **[How It Works](how-it-works.md)** - Technical deep dive
- **[API Reference](api-reference.md)** - Complete API documentation
- **[Examples](examples/basic-usage.md)** - Common use cases
- **[Benchmarking](benchmarking.md)** - Performance evaluation
- **[FAQ](faq.md)** - Common questions
- **[Contributing](contributing.md)** - Help improve REFRAG

## Why REFRAG?

Traditional RAG systems struggle with:

- ❌ Retrieving relevant but verbose chunks
- ❌ Missing semantic connections
- ❌ Large context windows wasting tokens
- ❌ Poor ranking of similar documents

REFRAG solves this by:

- ✅ LLM-optimized representations
- ✅ Better semantic understanding
- ✅ Condensed, focused context
- ✅ Improved retrieval precision

## Key Features

- **Multiple LLM Providers**: OpenAI, Anthropic, extensible
- **Flexible Embeddings**: Sentence-transformers, custom models
- **Smart Caching**: Reuse representations across sessions
- **Optional Reranking**: LLM-powered result refinement
- **Simple API**: 3 lines to get started

## Getting Started

```python
from refrag import REFRAGRetriever

retriever = REFRAGRetriever(llm_provider="openai")
retriever.index(documents)
results = retriever.retrieve(query, top_k=5)
```

## Community

- **GitHub**: [github.com/yourusername/refrag](https://github.com/yourusername/refrag)
- **Issues**: [Report bugs or request features](https://github.com/yourusername/refrag/issues)
- **Discussions**: [Ask questions](https://github.com/yourusername/refrag/discussions)

## Status

**Early Alpha (v0.1.0)** - API subject to change. Production use at your own risk.

---

**Next**: [Install REFRAG →](installation.md)
