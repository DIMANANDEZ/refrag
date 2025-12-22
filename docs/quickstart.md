# Quick Start Guide

Get REFRAG running in 5 minutes.

## Installation

```bash
pip install "refrag[openai]"
export OPENAI_API_KEY="your-key-here"
```

## Basic Example

```python
from refrag import REFRAGRetriever

# 1. Initialize retriever
retriever = REFRAGRetriever(
    llm_provider="openai",
    llm_model="gpt-4o-mini",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Prepare your documents
documents = [
    "Python is a high-level programming language created by Guido van Rossum in 1991.",
    "Machine learning is a subset of AI that learns from data without explicit programming.",
    "The Eiffel Tower in Paris was built in 1889 and stands 330 meters tall.",
]

# 3. Index documents
retriever.index(documents, show_progress=True)

# 4. Query
results = retriever.retrieve(
    query="Tell me about programming languages",
    top_k=2,
    return_scores=True
)

# 5. View results
for i, result in enumerate(results, 1):
    print(f"\n--- Result {i} (Score: {result['score']:.3f}) ---")
    print(f"Original: {result['text']}")
    print(f"Representation: {result['representation']}")
```

## Expected Output

```
Indexing 3 documents with REFRAG...
Generating representation 1/3...
Generating representation 2/3...
Generating representation 3/3...
✓ Indexed 3 documents

--- Result 1 (Score: 0.876) ---
Original: Python is a high-level programming language created by Guido van Rossum in 1991.
Representation: Python: High-level programming language by Guido van Rossum (1991), known for simplicity and readability.

--- Result 2 (Score: 0.654) ---
Original: Machine learning is a subset of AI that learns from data without explicit programming.
Representation: Machine learning: AI subset enabling data-driven learning without explicit code.
```

## What Just Happened?

1. **LLM Generated Representations**: Each document was condensed by GPT-4o-mini
2. **Embeddings Created**: Representations were embedded (not raw text)
3. **Query Matched**: Your query was matched against representation embeddings
4. **Results Ranked**: Top-k most relevant documents returned

## Key Difference from Vanilla RAG

```python
# Vanilla RAG:
"Python is a high-level programming language created by Guido van Rossum in 1991."
→ Embed raw text → Retrieve

# REFRAG:
"Python is a high-level programming language created by Guido van Rossum in 1991."
→ LLM generates: "Python: High-level language by Guido van Rossum (1991)..."
→ Embed representation → Retrieve (better matching!)
```

## Next Steps

### Compare with Vanilla RAG

```bash
python examples/compare_with_vanilla_rag.py
```

### Customize Representations

```python
custom_prompt = """
Extract only technical facts from this text.
Format: Key entities and their relationships.

Text: {chunk}

Facts:
"""

retriever = REFRAGRetriever(
    representation_prompt=custom_prompt
)
```

### Add Reranking

```python
from refrag import REFRAGReranker

# Get initial results
results = retriever.retrieve(query, top_k=10)

# Rerank for precision
reranker = REFRAGReranker(llm_provider="openai")
final_results = reranker.rerank(query, results, top_k=3)
```

### Batch Processing

```python
# For large document sets
retriever.index(
    documents,
    batch_size=20,  # Process 20 docs at a time
    show_progress=True
)
```

## Common Patterns

### Load from File

```python
with open("documents.txt", "r") as f:
    documents = [line.strip() for line in f if line.strip()]

retriever.index(documents)
```

### Save/Load Index

```python
# After indexing
import pickle

with open("refrag_index.pkl", "wb") as f:
    pickle.dump(retriever.document_data, f)

# Later, reload
with open("refrag_index.pkl", "rb") as f:
    retriever.document_data = pickle.load(f)
```

### Multiple Queries

```python
queries = [
    "What is Python?",
    "Tell me about AI",
    "Facts about Paris"
]

for query in queries:
    results = retriever.retrieve(query, top_k=1)
    print(f"{query} → {results[0]['text']}")
```

## Learn More

- **[How It Works →](how-it-works.md)** - Technical deep dive
- **[API Reference →](api-reference.md)** - Complete API docs
- **[Examples →](examples/basic-usage.md)** - More use cases
- **[Benchmarking →](benchmarking.md)** - Performance analysis
