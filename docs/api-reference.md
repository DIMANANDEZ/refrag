# API Reference

Complete reference for all REFRAG classes and methods.

## Core Classes

### REFRAGEmbedder

Generate LLM-powered representations and embeddings.

```python
from refrag import REFRAGEmbedder
```

#### Constructor

```python
REFRAGEmbedder(
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    llm_provider: str = "openai",
    llm_model: Optional[str] = None,
    api_key: Optional[str] = None,
    representation_prompt: Optional[str] = None,
    device: Optional[str] = None,
    normalize: bool = True,
    cache_representations: bool = True
)
```

**Parameters:**

- `embedding_model` (str): Sentence-transformers model name or path
- `llm_provider` (str): "openai" or "anthropic"
- `llm_model` (str, optional): Specific LLM model. Defaults:
  - OpenAI: "gpt-4o-mini"
  - Anthropic: "claude-3-5-haiku-20241022"
- `api_key` (str, optional): API key. Falls back to environment variables
- `representation_prompt` (str, optional): Custom prompt template with `{chunk}` placeholder
- `device` (str, optional): "cuda", "cpu", or None for auto
- `normalize` (bool): Normalize embeddings to unit length
- `cache_representations` (bool): Cache LLM-generated representations

#### Methods

##### embed_documents()

```python
embed_documents(
    documents: List[str],
    show_progress: bool = False
) -> Dict[str, Any]
```

Generate representations and embeddings for documents.

**Returns:**

```python
{
    'embeddings': np.ndarray,        # Shape: (n_docs, embedding_dim)
    'representations': List[str],    # LLM-generated reps
    'original': List[str]            # Original documents
}
```

##### embed_query()

```python
embed_query(query: str) -> np.ndarray
```

Embed query directly (no representation generation).

**Returns:** Query embedding vector

##### batch_embed_with_representations()

```python
batch_embed_with_representations(
    documents: List[str],
    batch_size: int = 10,
    show_progress: bool = True
) -> Dict[str, Any]
```

Process large document sets in batches.

**Parameters:**

- `batch_size`: Number of documents per LLM batch

**Returns:** Same as `embed_documents()`

##### get_cache_stats()

```python
get_cache_stats() -> Dict[str, int]
```

Get representation cache statistics.

**Returns:**

```python
{
    'cache_enabled': bool,
    'cached_items': int
}
```

---

### REFRAGRetriever

Main interface for indexing and retrieving documents.

```python
from refrag import REFRAGRetriever
```

#### Constructor

```python
REFRAGRetriever(
    embedder: Optional[REFRAGEmbedder] = None,
    **embedder_kwargs
)
```

**Parameters:**

- `embedder` (REFRAGEmbedder, optional): Pre-configured embedder
- `**embedder_kwargs`: Arguments passed to REFRAGEmbedder if embedder not provided

**Example:**

```python
# Option 1: Let retriever create embedder
retriever = REFRAGRetriever(
    llm_provider="openai",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

# Option 2: Provide custom embedder
embedder = REFRAGEmbedder(representation_prompt=custom_prompt)
retriever = REFRAGRetriever(embedder=embedder)
```

#### Methods

##### index()

```python
index(
    documents: List[str],
    show_progress: bool = False,
    batch_size: int = 10
)
```

Index documents with REFRAG representations.

**Parameters:**

- `documents`: List of document chunks
- `show_progress`: Display progress bar
- `batch_size`: Documents per batch

##### retrieve()

```python
retrieve(
    query: str,
    top_k: int = 5,
    return_scores: bool = False
) -> List[Dict[str, Any]]
```

Retrieve most relevant documents.

**Returns:**

```python
[
    {
        'text': str,              # Original document
        'representation': str,    # LLM-generated rep
        'score': float           # Similarity score (if return_scores=True)
    },
    ...
]
```

##### get_stats()

```python
get_stats() -> Dict[str, Any]
```

Get retriever statistics.

**Returns:**

```python
{
    'indexed': bool,
    'num_documents': int,
    'embedding_dim': int,
    'cache_enabled': bool,
    'cached_items': int
}
```

---

### REFRAGReranker

Optional LLM-based reranking for precision.

```python
from refrag import REFRAGReranker
```

#### Constructor

```python
REFRAGReranker(
    llm_provider: str = "openai",
    llm_model: Optional[str] = None,
    api_key: Optional[str] = None,
    rerank_prompt: Optional[str] = None
)
```

**Parameters:**

- `llm_provider`: "openai" or "anthropic"
- `llm_model`: Specific model (defaults to fast models)
- `api_key`: API key
- `rerank_prompt`: Custom reranking prompt

#### Methods

##### rerank()

```python
rerank(
    query: str,
    results: List[Dict[str, Any]],
    top_k: Optional[int] = None
) -> List[Dict[str, Any]]
```

Rerank results using LLM scoring.

**Parameters:**

- `query`: Original query
- `results`: Results from retrieve()
- `top_k`: Return only top-k after reranking

**Returns:** Results with added `'rerank_score'` field (0-10 scale)

---

## Utility Functions

### chunking

```python
from refrag.utils import chunk_text, chunk_documents
```

##### chunk_text()

```python
chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    separator: str = "\n\n"
) -> List[str]
```

Split text into overlapping chunks.

##### chunk_documents()

```python
chunk_documents(
    documents: List[str],
    chunk_size: int = 512,
    overlap: int = 50
) -> List[str]
```

Chunk multiple documents into flat list.

### metrics

```python
from refrag.utils import calculate_metrics
```

##### calculate_metrics()

```python
calculate_metrics(
    retrieved_docs: List[str],
    relevant_docs: List[str],
    k: int = 5
) -> Dict[str, float]
```

Calculate retrieval metrics.

**Returns:**

```python
{
    'precision@k': float,
    'recall@k': float,
    'f1@k': float,
    'k': int
}
```

---

## Configuration

### Model Presets

```python
from refrag.models.config import get_embedding_model, get_llm_model
```

**Embedding models:**

- `"mini"`: all-MiniLM-L6-v2 (fast, 384 dims)
- `"mpnet"`: all-mpnet-base-v2 (balanced, 768 dims)
- `"e5-large"`: e5-large-v2 (quality, 1024 dims)

**LLM models:**

- OpenAI: `"fast"` (gpt-4o-mini), `"quality"` (gpt-4o)
- Anthropic: `"fast"` (haiku), `"quality"` (sonnet)

**Usage:**

```python
embedding_model = get_embedding_model("mpnet")
llm_model = get_llm_model("openai", "fast")

retriever = REFRAGRetriever(
    embedding_model=embedding_model,
    llm_model=llm_model
)
```

---

## Complete Example

```python
from refrag import REFRAGRetriever, REFRAGReranker
from refrag.utils import chunk_documents

# Prepare documents
raw_docs = ["long document 1...", "long document 2..."]
chunks = chunk_documents(raw_docs, chunk_size=512, overlap=50)

# Index with REFRAG
retriever = REFRAGRetriever(
    llm_provider="openai",
    llm_model="gpt-4o-mini",
    embedding_model="sentence-transformers/all-mpnet-base-v2"
)
retriever.index(chunks, show_progress=True, batch_size=20)

# Retrieve
results = retriever.retrieve(
    query="What is the main finding?",
    top_k=10,
    return_scores=True
)

# Optional: Rerank top results
reranker = REFRAGReranker(llm_provider="openai")
final_results = reranker.rerank(
    query="What is the main finding?",
    results=results,
    top_k=3
)

# Use results
for r in final_results:
    print(f"Score: {r['rerank_score']}")
    print(f"Text: {r['text']}\n")
```

---

## Environment Variables

| Variable            | Description       |
| ------------------- | ----------------- |
| `OPENAI_API_KEY`    | OpenAI API key    |
| `ANTHROPIC_API_KEY` | Anthropic API key |

---

## Next Steps

- **[Examples →](examples/basic-usage.md)** - Common usage patterns
- **[How It Works →](how-it-works.md)** - Technical deep dive
- **[Benchmarking →](benchmarking.md)** - Performance evaluation
