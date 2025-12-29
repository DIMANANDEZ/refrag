# REFRAG: Representation-Focused Retrieval Augmented Generation

Practical implementation of REFRAG for improved RAG systems.

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/Shaivpidadi/refrag)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2509.01092-b31b1b.svg)](https://arxiv.org/abs/2509.01092)

## 🚀 What is REFRAG?

Traditional RAG systems use large chunks (512-1024 tokens) and send everything to the LLM. **REFRAG** optimizes this with:

1. **Micro-chunking**: 16-32 token chunks for fine-grained retrieval
2. **Fast indexing**: Direct encoding (NO LLM calls during indexing)
3. **Query-time compression**: Dynamic policy decides RAW vs COMPRESSED chunks
4. **Mixed context**: High-priority chunks get full detail, others compressed to keywords

### Key Benefits

- **Blazing fast indexing**: No LLM overhead during indexing (seconds vs minutes)
- **Fine-grained retrieval**: Micro-chunks enable precise information extraction
- **Smaller context windows**: Query-time compression reduces token usage
- **Better quality**: Keep full detail for relevant chunks, compress the rest

Based on concepts from [REFRAG research (arXiv:2509.01092)](https://arxiv.org/abs/2509.01092). This implementation focuses on the core REFRAG approach: micro-chunking with query-time compression.

### Visualizing the "Mixed Context" Strategy

```
[Query]: "How does the transformer attention mechanism work?"

Standard RAG Context (Expensive):
┌─────────────────────────────────────────────────────────────┐
│ [Chunk 1: 512 tokens]                                       │
│ ...full text about RNNs and sequential processing...        │
│ (Irrelevant but you still pay for 512 tokens)               │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│ [Chunk 2: 512 tokens]                                       │
│ ...The attention mechanism computes a weighted sum...       │
│ (Relevant - you need this!)                                 │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│ [Chunk 3: 512 tokens]                                       │
│ ...full text about CNNs and image processing...             │
│ (Irrelevant but you still pay for 512 tokens)               │
└─────────────────────────────────────────────────────────────┘
Total: ~1,536 tokens

REFRAG Context (Efficient):
┌─────────────────────────────────────────────────────────────┐
│ [COMPRESSED] RNNs sequential vanishing gradient LSTM        │
│ (30 tokens - just keywords)                                 │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│ [RAW] The attention mechanism computes a weighted sum of    │
│ values based on query-key similarity, enabling the model... │
│ (512 tokens - full detail preserved)                        │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│ [COMPRESSED] CNNs convolution pooling image classification  │
│ (25 tokens - just keywords)                                 │
└─────────────────────────────────────────────────────────────┘
Total: ~567 tokens

Result: 63% fewer tokens, same answer quality ✨
```

### Use Cases

**When to use REFRAG:**
- Large document collections (1000+ docs)
- Token cost is a concern
- Need precise retrieval (not just chunks)
- Fast indexing required

**When traditional RAG is fine:**
- Small collections (< 100 docs)
- Context window not a bottleneck
- Simplicity over optimization

## Quick Example

```python
from refrag import REFRAGRetriever, MicroChunker

# 1. Micro-chunk your documents
chunker = MicroChunker(chunk_size=32)  # 32 tokens per chunk
chunks = chunker.chunk_documents(documents)

# 2. Fast indexing (NO LLM!)
retriever = REFRAGRetriever()
retriever.index(chunks)  # Fast! Just encoder embeddings

# 3. Retrieve with query-time compression
result = retriever.retrieve_with_compression(
    query="Tell me about machine learning",
    top_k=10
)

# Result contains:
# - Mixed RAW + COMPRESSED context
# - Top 30% chunks: Full detail
# - Bottom 70% chunks: Keywords only
```

## Core Features

### 1. Micro-Chunking (16-32 tokens)
```python
from refrag import MicroChunker

chunker = MicroChunker(chunk_size=32)
chunks = chunker.chunk_text("Your document here...")
# Creates small, precise chunks for better retrieval
```

### 2. Fast Indexing (No LLM)
```python
retriever = REFRAGRetriever()
retriever.index(chunks)  # Direct encoding only!
# 100x+ faster than LLM-based approaches
```

### 3. Query-Time Compression
```python
result = retriever.retrieve_with_compression(query, top_k=10)
# Automatically decides: RAW vs COMPRESSED per chunk
# Based on relevance scores
```

### 4. Mixed Context Output
```
[RAW]Python is a programming language created by Guido van Rossum.[/RAW]
[COMPRESSED]machine learning AI neural networks[/COMPRESSED]
[RAW]JavaScript runs in web browsers for interactive sites.[/RAW]
```

## 🔌 Compatibility

REFRAG is **model-agnostic**. It prepares the context *before* it reaches the LLM.

| Component | Support |
|-----------|---------|
| **LLMs** | ✅ OpenAI (GPT-4, GPT-4o) <br> ✅ Anthropic (Claude 3, Claude 3.5) <br> ✅ Open-source (Llama 3, Mistral, Gemini) <br> ✅ Any LLM API that accepts text input |
| **Embeddings** | ✅ Any HuggingFace `sentence-transformers` model <br> ✅ OpenAI Embeddings <br> ✅ Custom embedding models |
| **Vector DBs** | ⚠️ In-memory (current) <br> 🔜 FAISS (planned) <br> 🔜 Qdrant (planned) <br> 🔜 Weaviate (planned) |
| **Frameworks** | ✅ Standalone <br> ✅ LangChain (easy integration) <br> 🔜 LlamaIndex (planned) |

### How It Works with Your LLM

REFRAG sits **between** retrieval and LLM generation:

```python
# 1. REFRAG prepares optimized context
result = retriever.retrieve_with_compression(query)
context = result['context']  # Mixed RAW + COMPRESSED

# 2. Send to ANY LLM
# OpenAI
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}]
)

# Anthropic
response = anthropic.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}]
)

# Llama (via Ollama/HuggingFace)
# Works the same way!
```

## 📦 Installation

### Via pip (coming soon)
```bash
pip install refrag
```

### From source
```bash
git clone https://github.com/Shaivpidadi/refrag.git
cd refrag
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- sentence-transformers >= 2.2.0
- transformers >= 4.30.0
- torch >= 2.0.0
- numpy >= 1.21.0

**Note:** No OpenAI/Anthropic API keys needed! This implementation doesn't use LLMs during indexing.

## 🏗️ Architecture

REFRAG consists of 5 core components:

```
┌─────────────────────────────────────────────────────────┐
│ 1. MicroChunker: 16-32 token chunks                    │
│    - Token-based (not character-based)                  │
│    - Fine-grained retrieval                             │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ 2. FastEncoder: Direct embedding (NO LLM!)             │
│    - sentence-transformers model                        │
│    - Seconds to index, not minutes                      │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ 3. CompressionPolicy: Decide RAW vs COMPRESSED         │
│    - Query-time decisions (not pre-compression)         │
│    - Based on similarity scores                         │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ 4. ChunkCompressor: Extract keywords                    │
│    - For low-priority chunks                            │
│    - Fast heuristic-based                               │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ 5. MixedContextDecoder: Build final context            │
│    - Combines RAW + COMPRESSED                          │
│    - Ready for LLM input                                │
└─────────────────────────────────────────────────────────┘
```

## ⚡ Benchmarks

We benchmarked REFRAG against **standard RAG** on the HotpotQA dataset (Wikipedia question-answering) with **49,691 real documents**.

### The REFRAG Advantage: Token Efficiency

REFRAG and standard RAG both use **fast direct encoding** (no LLM during indexing). The difference is in **how they use context**.

**Example benchmark results** (HotpotQA dataset, 49,691 documents):

| Metric | Standard RAG | REFRAG | Improvement |
|--------|--------------|--------|-------------|
| **Chunk Strategy** | Large chunks (512 tokens) | Micro-chunks (32 tokens) | Fine-grained retrieval |
| **Compression** | None | Query-time adaptive | Smart context |
| **Indexing Speed** | Fast (direct encoding) | Fast (direct encoding) | **Same** |
| **Avg Tokens to LLM** | ~177 tokens/query | ~83 tokens/query | **53% reduction** |
| **Retrieval Speed** | 62.4ms/query | 22.5ms/query | **2.8x faster** |
| **LLM API Cost** | Baseline | 53% lower | **$$ Savings** |
| **Retrieval Quality** | Good | Good | **Same** |

### Example Results on 49,691 Documents (208,081 chunks)

**Indexing (Both Fast)**:
- Standard RAG: ~60s to index
- REFRAG: ~58s to index
- **Both use direct encoding** (no LLM calls)

**Token Efficiency (REFRAG's Strength)**:
- Standard RAG: ~177 tokens/query sent to LLM
- REFRAG: ~83 tokens/query sent to LLM
- **53% fewer tokens** = 53% cost savings on every query

**⚠️ Benchmark Disclaimer**: These are results from a specific run on our test setup (M4 MacBook, `all-MiniLM-L6-v2` encoder, HotpotQA dataset with 5,000 samples). Your results may vary based on hardware, dataset, and configuration.

**Run your own benchmark** to see actual performance on your data:
```bash
pip install datasets
PYTHONPATH=. python examples/compare_with_vanilla_rag.py
```

The benchmark script calculates and reports **actual measured values** (not hardcoded claims). Results will vary based on your dataset and hardware.

### Comparison Table

| Feature | Standard RAG | REFRAG |
|---------|--------------|--------|
| **Chunk size** | 512-1024 tokens | 16-32 tokens (micro) |
| **Indexing** | Direct encoding | Direct encoding |
| **Indexing speed** | Fast | **Same (fast)** |
| **Compression** | None | Query-time adaptive |
| **Context format** | All chunks same | Mixed RAW/COMPRESSED |
| **Tokens to LLM** | ~177/query | **~83/query** |
| **Token efficiency** | Baseline | **53% better** |
| **LLM API cost** | Baseline | **53% lower** |
| **Retrieval precision** | Chunk-level | **Token-level** |
| **Tested on** | - | **49,691 real docs** |

## ⚙️ Advanced Configuration

You can customize the underlying components to fit your specific needs:

```python
from refrag import REFRAGRetriever, MicroChunker, CompressionPolicy, ChunkCompressor

# 1. Change Chunk Size (Standard is 16-32)
chunker = MicroChunker(chunk_size=64)  # Larger chunks for longer contexts
chunks = chunker.chunk_documents(documents)

# 2. Change Encoder Model (Supports any HuggingFace sentence-transformers model)
# Use a larger, more accurate model if needed
retriever = REFRAGRetriever(
    embedding_model="BAAI/bge-small-en-v1.5"  # Or "all-mpnet-base-v2", etc.
)

# Or with GPU support:
retriever = REFRAGRetriever(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    device="cuda"  # or "cpu", "mps" for Apple Silicon
)

# 3. Adjust Compression Aggressiveness
# raw_percentage: 0.0-1.0 (Higher = more chunks kept as RAW)
from refrag import CompressionPolicy

policy = CompressionPolicy(
    raw_percentage=0.4,        # Keep top 40% as RAW (default: 0.3)
    min_raw_chunks=3,          # Always keep at least 3 RAW (default: 2)
    similarity_threshold=0.6   # Minimum score for RAW consideration
)
retriever = REFRAGRetriever(compression_policy=policy)

# 4. Custom Compression Method
from refrag import ChunkCompressor

compressor = ChunkCompressor(
    compression_method="keywords",  # "keywords", "entities", or "first_n"
    max_keywords=10                 # More keywords = better context (default: 5)
)
retriever = REFRAGRetriever(compressor=compressor)

# 5. Custom Context Format
from refrag import MixedContextDecoder

decoder = MixedContextDecoder(
    format_style="separated"  # "tagged", "separated", or "inline"
)
retriever = REFRAGRetriever(decoder=decoder)

# 6. Combine Everything
retriever = REFRAGRetriever(
    embedding_model="BAAI/bge-large-en-v1.5",
    compression_policy=CompressionPolicy(raw_percentage=0.5),
    compressor=ChunkCompressor(compression_method="entities", max_keywords=8),
    decoder=MixedContextDecoder(format_style="inline")
)
```

### Performance Tuning

```python
# For speed (smaller model, more compression)
retriever = REFRAGRetriever(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # Fast
    compression_policy=CompressionPolicy(raw_percentage=0.2),  # Compress 80%
    device="cuda"  # GPU acceleration
)

# For quality (larger model, less compression)
retriever = REFRAGRetriever(
    embedding_model="BAAI/bge-large-en-v1.5",  # High accuracy
    compression_policy=CompressionPolicy(raw_percentage=0.5),  # Keep 50% RAW
)

# For token efficiency (maximum compression)
retriever = REFRAGRetriever(
    compression_policy=CompressionPolicy(raw_percentage=0.1),  # Keep only 10% RAW
    compressor=ChunkCompressor(max_keywords=3)  # Minimal keywords
)
```

## 🚦 Usage Examples

### Basic Usage
See [examples/basic_usage.py](examples/basic_usage.py) for a complete working example.

### Benchmark on HotpotQA Dataset

Run the comprehensive benchmark using real-world data:
```bash
pip install datasets  # Install HuggingFace datasets
PYTHONPATH=. python examples/compare_with_vanilla_rag.py
```

**Results on 49,691 Wikipedia documents (HotpotQA)**:
- **Dataset**: 5,000 samples → 49,691 documents → 208,081 micro-chunks
- **Indexing**: Same speed as standard RAG (~58s, both use direct encoding)
- **Token reduction**: 53% fewer tokens sent to LLM per query
- **Cost savings**: 53% reduction in LLM API costs
- **Retrieval speed**: 2.8x faster with compression
- **Quality**: Same accuracy as standard RAG, better context efficiency

See [HOTPOTQA_BENCHMARK_RESULTS.md](HOTPOTQA_BENCHMARK_RESULTS.md) for detailed analysis.

## 🎓 How It Works

### Indexing Phase (FAST)
1. Split documents into 16-32 token micro-chunks
2. Encode directly with sentence-transformers
3. Store embeddings + original chunks
4. **No LLM calls** = blazing fast!

### Retrieval Phase (SMART)
1. Embed query
2. Find top-k similar chunks via vector search
3. Apply compression policy (decide RAW vs COMPRESSED)
4. Compress low-priority chunks to keywords
5. Build mixed context for LLM

### Why This Works
- **Micro-chunks**: Precise retrieval at sub-document level
- **No LLM during indexing**: 100x+ speed improvement
- **Query-time compression**: Adaptive based on relevance
- **Mixed context**: Best of both worlds (detail + coverage)

### ⚠️ Note on "Vector" vs "Text" Compression

The original [REFRAG paper](https://arxiv.org/abs/2509.01092) performs compression in **Vector Space** (injecting raw embeddings into the LLM).

Since commercial APIs (GPT-4, Claude) do not allow vector injection, this library adapts the architecture to **Text Space**:
- **Vector Space (Paper):** `[Vector_A] [Vector_B]` → LLM
- **Text Space (This Repo):** `[Keywords_A] [Keywords_B]` → LLM

This allows you to get the *token-saving benefits* of REFRAG on standard APIs without needing your own GPU cluster.

## ⚠️ Limitations

### Current Implementation
- Uses **heuristic-based** compression policy (not RL-based like paper)
- English-only keyword extraction (stopwords hardcoded)
- No vector database integration yet (in-memory only)
- Text-space compression (not vector-space like original paper)

### Roadmap
- [ ] RL-based compression policy training
- [ ] FAISS/Qdrant integration for large-scale deployments
- [ ] Multi-language support (non-English stopwords)
- [ ] Streaming/incremental indexing
- [ ] Vector-space compression (requires custom LLM)
- [ ] Built-in reranking support
- [ ] LlamaIndex/LangChain integration

### Known Issues
- Very small chunks (< 16 tokens) may lose context
- Compression quality varies by domain (technical docs work best)
- Capitalized-word heuristic may miss important lowercase keywords

## 📚 Citation

This implementation is based on the following paper:

```bibtex
@misc{lin2025refragrethinkingragbased,
      title={REFRAG: Rethinking RAG based Decoding}, 
      author={Xiaoqiang Lin and Aritra Ghosh and Bryan Kian Hsiang Low and Anshumali Shrivastava and Vijai Mohan},
      year={2025},
      eprint={2509.01092},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.01092}, 
}
```

If you use this implementation in your research, please cite both the original paper and this repository.

## 🙏 Acknowledgments

Based on [REFRAG research by Meta AI](https://arxiv.org/abs/2509.01092). This is an independent implementation for the open-source community.

**Disclaimer:** This is not an official Meta product. For the official implementation, please refer to Meta's repositories.
