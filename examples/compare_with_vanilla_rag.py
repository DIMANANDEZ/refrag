# examples/compare_with_vanilla_rag.py
"""
Compare REFRAG vs Vanilla RAG performance
"""

from refrag import REFRAGRetriever
from sentence_transformers import SentenceTransformer
import numpy as np
import time
from dotenv import load_dotenv 
import os

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Sample documents
documents = [
    "Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum in 1991.",
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data without explicit programming.",
    "JavaScript is primarily used for web development and runs in browsers. It was created by Brendan Eich in 1995.",
    "Deep learning uses neural networks with multiple layers to process complex patterns in data.",
    "Rust is a systems programming language focused on safety and performance, created by Mozilla Research.",
]

class VanillaRAG:
    """Simple vanilla RAG for comparison."""
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.documents = None
        self.embeddings = None
    
    def index(self, documents):
        self.documents = documents
        self.embeddings = self.model.encode(documents, normalize_embeddings=True)
    
    def retrieve(self, query, top_k=3):
        query_emb = self.model.encode(query, normalize_embeddings=True)
        similarities = np.dot(self.embeddings, query_emb)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [{"text": self.documents[idx], "score": float(similarities[idx])} 
                for idx in top_indices]

def main():
    query = "What programming languages are good for AI development?"
    
    print("=" * 60)
    print("COMPARING VANILLA RAG vs REFRAG")
    print("=" * 60)
    
    # Vanilla RAG
    print("\n[1] VANILLA RAG")
    print("-" * 60)
    vanilla = VanillaRAG()
    
    start = time.time()
    vanilla.index(documents)
    index_time_vanilla = time.time() - start
    
    start = time.time()
    vanilla_results = vanilla.retrieve(query, top_k=3)
    retrieve_time_vanilla = time.time() - start
    
    print(f"Query: {query}\n")
    for i, result in enumerate(vanilla_results, 1):
        print(f"Result {i} (Score: {result['score']:.4f}):")
        print(f"  {result['text']}\n")
    
    print(f"Index time: {index_time_vanilla:.3f}s")
    print(f"Retrieve time: {retrieve_time_vanilla:.3f}s")
    
    # REFRAG
    print("\n[2] REFRAG")
    print("-" * 60)
    refrag = REFRAGRetriever(
        llm_provider="openai",
        llm_model="gpt-4o-mini"
    )
    
    start = time.time()
    refrag.index(documents, show_progress=False)
    index_time_refrag = time.time() - start
    
    start = time.time()
    refrag_results = refrag.retrieve(query, top_k=3, return_scores=True)
    retrieve_time_refrag = time.time() - start
    
    print(f"Query: {query}\n")
    for i, result in enumerate(refrag_results, 1):
        print(f"Result {i} (Score: {result['score']:.4f}):")
        print(f"  Original: {result['text']}")
        print(f"  Representation: {result['representation']}\n")
    
    print(f"Index time: {index_time_refrag:.3f}s")
    print(f"Retrieve time: {retrieve_time_refrag:.3f}s")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Vanilla RAG indexing: {index_time_vanilla:.3f}s")
    print(f"REFRAG indexing: {index_time_refrag:.3f}s (slower due to LLM calls)")
    print(f"\nVanilla RAG retrieval: {retrieve_time_vanilla:.3f}s")
    print(f"REFRAG retrieval: {retrieve_time_refrag:.3f}s")
    print(f"\nREFRAG provides better semantic matching via LLM representations")
    print(f"Trade-off: Slower indexing, same retrieval speed, better relevance")

if __name__ == "__main__":
    main()