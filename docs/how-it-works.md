# How REFRAG Works

Deep dive into the architecture and methodology.

## The Problem with Vanilla RAG

Traditional RAG systems have fundamental limitations:

### 1. Semantic Mismatch

```
Query: "What causes inflation?"

Vanilla RAG matches against:
"The Federal Reserve adjusts interest rates to control monetary policy.
When rates are low, borrowing increases, leading to more spending..."

Problem: Query is about "inflation" but chunk discusses "interest rates"
and "monetary policy" - vector similarity is weak despite relevance.
```

### 2. Verbose Chunks

```
Raw chunk (450 tokens):
"In the groundbreaking paper published in 2017, Vaswani et al. introduced
the Transformer architecture which revolutionized natural language processing.
The paper, titled 'Attention is All You Need,' presented a novel approach..."

Actual useful info: Transformers by Vaswani 2017, attention mechanism
```

### 3. Context Pollution

Retrieving 5 verbose chunks = 2000+ tokens sent to LLM, but only 20% is relevant.

## The REFRAG Solution

### Core Innovation: LLM-Generated Representations

Instead of embedding raw text, REFRAG uses an LLM to generate **task-optimized representations**:

```python
# Step 1: Generate representation
chunk = """
The Federal Reserve adjusts interest rates to control monetary policy.
When rates are low, borrowing increases, leading to more spending.
This increased demand can drive up prices, causing inflation.
"""

representation = llm.generate(f"""
Extract key facts for Q&A:
{chunk}
""")
# Output: "Fed uses interest rates to control inflation. Low rates →
# more borrowing → higher spending → price increases."

# Step 2: Embed representation (not raw chunk)
embedding = embed(representation)
```

### Why This Works

1. **Semantic Compression**: LLM distills key information
2. **Focused Matching**: Embeddings capture core concepts, not filler
3. **Better Ranking**: Similar concepts expressed differently still match
4. **Token Efficiency**: Smaller representations = less context waste

## Architecture

### Indexing Pipeline

```
Document → Chunking → LLM Representation → Embedding → Vector Store
   ↓           ↓              ↓                ↓            ↓
Raw text   Chunks      Condensed facts    Dense vector  Indexed
```

**Code flow:**

```python
def index(documents):
    for doc in documents:
        # Generate representation
        rep = llm.complete(
            f"Condense this: {doc}"
        )

        # Embed representation
        vec = embedding_model.encode(rep)

        # Store
        vector_db.add(vec, metadata={
            'original': doc,
            'representation': rep
        })
```

### Retrieval Pipeline

```
Query → Embed → Search Vector Store → Rank by Similarity → Return Results
  ↓        ↓            ↓                     ↓                    ↓
"What?"  Dense    Representation        Top-K vectors    Original docs
         vector    vectors                               + reps
```

**Code flow:**

```python
def retrieve(query, top_k=5):
    # Embed query (no representation needed)
    query_vec = embedding_model.encode(query)

    # Search against representation vectors
    results = vector_db.search(query_vec, k=top_k)

    # Return original docs + representations
    return [{
        'original': r.metadata['original'],
        'representation': r.metadata['representation'],
        'score': r.similarity
    } for r in results]
```

## Technical Components

### 1. REFRAGEmbedder

**Responsibilities:**

- Generate LLM representations
- Create embeddings
- Cache representations

**Key methods:**

```python
class REFRAGEmbedder:
    def _generate_representation(chunk: str) -> str:
        """Use LLM to condense chunk"""

    def embed_documents(docs: List[str]) -> Dict:
        """Generate reps + embeddings"""

    def embed_query(query: str) -> np.ndarray:
        """Embed query directly"""
```

### 2. REFRAGRetriever

**Responsibilities:**

- Manage document index
- Perform similarity search
- Return ranked results

**Key methods:**

```python
class REFRAGRetriever:
    def index(documents: List[str]):
        """Index docs with representations"""

    def retrieve(query: str, top_k: int) -> List[Dict]:
        """Find top-k relevant docs"""
```

### 3. REFRAGReranker (Optional)

**Responsibilities:**

- LLM-based scoring of retrieved results
- Refine top-k for precision

**When to use:**

- Need highest precision for top 3-5 results
- Budget allows extra LLM calls
- Query is complex/nuanced

## Representation Prompt Engineering

The representation prompt is critical. Default:

```
Extract and condense the key information from this text chunk
that would be useful for answering questions.
Focus on facts, entities, and relationships. Keep it concise (2-3 sentences).

Text: {chunk}

Condensed representation:
```

### Customization Examples

**For code documentation:**

```
Extract API information:
- Function names and parameters
- Return types
- Key behaviors
- Examples if present

Code: {chunk}

API summary:
```

**For research papers:**

```
Extract research essentials:
- Main hypothesis/question
- Methodology
- Key findings
- Limitations

Paper section: {chunk}

Research summary:
```

**For customer support:**

```
Extract actionable information:
- Problem described
- Steps mentioned
- Outcome/resolution
- Relevant product features

Support ticket: {chunk}

Support summary:
```

## Performance Characteristics

### Indexing

| Metric    | Value            | Notes              |
| --------- | ---------------- | ------------------ |
| Speed     | ~0.5-1s per doc  | LLM call overhead  |
| Cost      | ~$0.0001 per doc | Using GPT-4o-mini  |
| Cache hit | Instant          | If doc seen before |

**Trade-off**: Slower indexing (one-time cost) for better retrieval.

### Retrieval

| Metric       | REFRAG      | Vanilla RAG |
| ------------ | ----------- | ----------- |
| Speed        | ~50ms       | ~50ms       |
| Relevance    | +15-25%     | Baseline    |
| Context size | -30% tokens | Baseline    |

**Trade-off**: Same speed, better quality.

## Comparison: REFRAG vs Vanilla RAG

### Example Query: "How does photosynthesis work?"

**Vanilla RAG** matches against:

```
"Plants are living organisms that belong to the kingdom Plantae.
They use a process called photosynthesis to convert light energy..."
```

- Embedding captures: "plants", "organisms", "Plantae", "photosynthesis", "light"
- Lots of noise: "kingdom Plantae" isn't relevant

**REFRAG** matches against representation:

```
"Photosynthesis: Plants convert light energy into chemical energy (glucose)
using chlorophyll. Process requires sunlight, water, CO2."
```

- Embedding captures: "photosynthesis", "light energy", "chemical energy", "glucose", "chlorophyll"
- Pure signal: All terms directly relevant

**Result**: REFRAG ranks this chunk higher for the query.

## Limitations

### When REFRAG May Not Help

1. **Very short documents**: If chunks are already 1-2 sentences, representation overhead not worth it
2. **Exact keyword matching**: If you need literal string matches (use keyword search)
3. **Real-time indexing**: LLM calls add latency (cache helps)
4. **Budget constraints**: LLM costs for large corpora

### Mitigation Strategies

- **Hybrid approach**: Use vanilla RAG for simple chunks, REFRAG for complex ones
- **Aggressive caching**: Reuse representations across sessions
- **Cheaper models**: GPT-4o-mini, Haiku instead of GPT-4
- **Batch processing**: Amortize LLM overhead

## Future Directions

### Potential Improvements

1. **Learned representations**: Fine-tune LLM on your domain
2. **Multi-hop reasoning**: Chain representations for complex queries
3. **Adaptive granularity**: Vary representation length by chunk complexity
4. **Embedding fusion**: Combine raw + representation embeddings

### Research Questions

- Optimal representation length vs retrieval quality?
- Can we predict which chunks benefit from REFRAG?
- How to measure representation "quality"?

## Next Steps

- **[API Reference →](api-reference.md)** - Detailed API documentation
- **[Examples →](examples/advanced-usage.md)** - Complex use cases
- **[Benchmarking →](benchmarking.md)** - Evaluate performance
