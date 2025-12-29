# examples/basic_usage.py
"""
REFRAG usage example - Correct implementation

Shows:
1. Micro-chunking (16-32 tokens)
2. Fast indexing (no LLM)
3. Query-time compression
4. Mixed RAW/COMPRESSED context
"""

from refrag import REFRAGRetriever, MicroChunker
import time

# Sample documents
documents = [
    "Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum in 1991.",
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data without explicit programming.",
    "The Eiffel Tower is located in Paris, France. It was built in 1889 and stands 330 meters tall.",
    "Photosynthesis is the process by which plants convert sunlight into chemical energy stored in glucose.",
    "The Pacific Ocean is the largest ocean on Earth, covering about 46% of the water surface.",
    "JavaScript is a programming language primarily used for web development. It runs in browsers and enables interactive websites.",
    "Deep learning uses neural networks with multiple layers to learn complex patterns from data.",
    "The Great Wall of China is one of the world's most famous landmarks, stretching over 13,000 miles.",
    "Cellular respiration is the process by which cells convert glucose into ATP, the energy currency of cells.",
    "The Amazon River is the second longest river in the world and has the largest drainage basin."
]


def main():
    print("=" * 60)
    print("REFRAG: Fast Retrieval with Query-Time Compression")
    print("=" * 60)

    # Step 1: Micro-chunking
    print("\n[1/4] Micro-chunking documents (16-32 tokens per chunk)...")
    chunker = MicroChunker(chunk_size=32)
    chunks = chunker.chunk_documents(documents)
    print(f"✓ Created {len(chunks)} micro-chunks from {len(documents)} documents")
    print(f"  Example chunk: '{chunks[0]}'")

    # Step 2: Fast indexing (NO LLM)
    print("\n[2/4] Indexing with fast encoder (no LLM calls)...")
    retriever = REFRAGRetriever(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )

    start_time = time.time()
    retriever.index(chunks, show_progress=True, batch_size=32)
    index_time = time.time() - start_time

    print(f"✓ Indexing completed in {index_time:.2f} seconds")
    print(f"  Stats: {retriever.get_stats()}")

    # Step 3: Basic retrieval
    print("\n[3/4] Basic retrieval (no compression)...")
    query = "Tell me about programming languages"
    print(f"  Query: '{query}'")

    results = retriever.retrieve(query, top_k=5, return_scores=True)

    print(f"\n  Top 3 results:")
    for i, result in enumerate(results[:3], 1):
        print(f"  {i}. [{result['score']:.3f}] {result['text']}")

    # Step 4: Retrieval with compression (CORE REFRAG FEATURE)
    print("\n[4/4] Retrieval with query-time compression...")
    print(f"  Query: '{query}'")

    compressed_results = retriever.retrieve_with_compression(
        query,
        top_k=10,
        return_context=True,
        format_style="tagged"
    )

    print(f"\n  Retrieved {len(compressed_results['chunks'])} chunks:")
    raw_count = sum(compressed_results['is_raw'])
    compressed_count = len(compressed_results['is_raw']) - raw_count
    print(f"  - RAW chunks: {raw_count}")
    print(f"  - COMPRESSED chunks: {compressed_count}")

    print("\n  Mixed context preview:")
    print("  " + "-" * 56)
    context_preview = compressed_results['context'].split('\n')[:6]
    for line in context_preview:
        print(f"  {line}")
    if len(compressed_results['context'].split('\n')) > 6:
        print("  ...")
    print("  " + "-" * 56)

    print("\n  Compression details:")
    for i in range(min(5, len(compressed_results['chunks']))):
        chunk = compressed_results['chunks'][i]
        compressed = compressed_results['compressed'][i]
        is_raw = compressed_results['is_raw'][i]
        score = compressed_results['scores'][i]

        print(f"\n  Chunk {i+1} (score: {score:.3f}):")
        print(f"    Type: {'RAW' if is_raw else 'COMPRESSED'}")
        print(f"    Original: {chunk[:60]}...")
        if not is_raw:
            print(f"    Compressed: {compressed}")

    print("\n" + "=" * 60)
    print("REFRAG Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()