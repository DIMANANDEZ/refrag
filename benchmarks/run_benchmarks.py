"""
Benchmark script to compare Refrag with vanilla RAG.

This script runs performance benchmarks on multiple datasets and metrics.
"""

import json
import time
from typing import List, Dict, Tuple
from pathlib import Path

from refrag.core.embedder import Embedder
from refrag.core.retriever import Retriever
from refrag.core.reranker import Reranker
from refrag.utils.metrics import evaluate_retrieval, compare_systems


class BenchmarkRunner:
    """Run benchmarks comparing different RAG configurations."""
    
    def __init__(self, dataset_path: str):
        """
        Initialize benchmark runner.
        
        Args:
            dataset_path: Path to the benchmark dataset JSON file
        """
        self.dataset_path = dataset_path
        self.load_dataset()
        self.results = {
            "vanilla_rag": [],
            "refrag": [],
            "refrag_with_chunking": []
        }
    
    def load_dataset(self):
        """Load benchmark dataset."""
        with open(self.dataset_path, 'r') as f:
            self.dataset = json.load(f)
        print(f"Loaded {len(self.dataset)} test queries")
    
    def run_vanilla_rag(
        self,
        documents: List[str],
        queries: List[str],
        ground_truth: List[set]
    ) -> Dict:
        """Run vanilla RAG without reranking."""
        print("\n=== Running Vanilla RAG ===")
        
        # Initialize
        embedder = Embedder()
        doc_embeddings = embedder.embed_documents(documents)
        
        retriever = Retriever(
            embedding_dim=doc_embeddings.shape[1],
            index_type="flat"
        )
        retriever.add_documents(doc_embeddings, documents)
        
        # Run queries
        results = []
        total_time = 0
        
        for i, query in enumerate(queries):
            start_time = time.time()
            query_embedding = embedder.embed_query(query)
            docs, scores, _ = retriever.retrieve(query_embedding, k=5)
            query_time = time.time() - start_time
            total_time += query_time
            
            # Evaluate
            retrieved_indices = {documents.index(doc) for doc in docs}
            metrics = evaluate_retrieval(
                [str(idx) for idx in retrieved_indices],
                {str(idx) for idx in ground_truth[i]},
                k_values=[1, 3, 5]
            )
            results.append(metrics)
        
        avg_time = total_time / len(queries)
        print(f"Average query time: {avg_time:.4f}s")
        
        return {
            "results": results,
            "avg_time": avg_time
        }
    
    def run_refrag(
        self,
        documents: List[str],
        queries: List[str],
        ground_truth: List[set]
    ) -> Dict:
        """Run Refrag with reranking."""
        print("\n=== Running Refrag ===")
        
        # Initialize
        embedder = Embedder()
        doc_embeddings = embedder.embed_documents(documents)
        
        retriever = Retriever(
            embedding_dim=doc_embeddings.shape[1],
            index_type="flat"
        )
        retriever.add_documents(doc_embeddings, documents)
        reranker = Reranker(method="cross_encoder")
        
        # Run queries
        results = []
        total_time = 0
        
        for i, query in enumerate(queries):
            start_time = time.time()
            
            # Retrieve
            query_embedding = embedder.embed_query(query)
            docs, scores, _ = retriever.retrieve(query_embedding, k=10)
            
            # Rerank
            reranked_docs, _, _ = reranker.rerank(
                query, docs, scores, top_k=5
            )
            
            query_time = time.time() - start_time
            total_time += query_time
            
            # Evaluate
            retrieved_indices = {documents.index(doc) for doc in reranked_docs}
            metrics = evaluate_retrieval(
                [str(idx) for idx in retrieved_indices],
                {str(idx) for idx in ground_truth[i]},
                k_values=[1, 3, 5]
            )
            results.append(metrics)
        
        avg_time = total_time / len(queries)
        print(f"Average query time: {avg_time:.4f}s")
        
        return {
            "results": results,
            "avg_time": avg_time
        }
    
    def print_summary(self, vanilla_results: Dict, refrag_results: Dict):
        """Print benchmark summary."""
        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 70)
        
        # Compare systems
        comparison = compare_systems(
            vanilla_results["results"],
            refrag_results["results"]
        )
        
        print("\nMetric Comparison:")
        print(f"{'Metric':<20} {'Vanilla RAG':<15} {'Refrag':<15} {'Improvement':<15}")
        print("-" * 70)
        
        for metric, values in comparison.items():
            improvement = values["improvement"]
            improvement_str = f"+{improvement:.2f}%" if improvement > 0 else f"{improvement:.2f}%"
            
            print(
                f"{metric:<20} "
                f"{values['system_a_mean']:<15.4f} "
                f"{values['system_b_mean']:<15.4f} "
                f"{improvement_str:<15}"
            )
        
        print(f"\nTiming:")
        print(f"  Vanilla RAG: {vanilla_results['avg_time']:.4f}s")
        print(f"  Refrag:      {refrag_results['avg_time']:.4f}s")
        
        time_overhead = (
            (refrag_results['avg_time'] - vanilla_results['avg_time']) 
            / vanilla_results['avg_time'] * 100
        )
        print(f"  Time overhead: {time_overhead:.2f}%")


def main():
    """Main benchmark execution."""
    print("Refrag Benchmark Suite")
    print("=" * 70)
    
    # Sample benchmark data
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are inspired by biological brain structures.",
        "Deep learning uses multiple layers to process information.",
        "NLP helps computers understand human language.",
        "Computer vision enables machines to interpret visual data.",
        "Reinforcement learning trains agents through rewards.",
        "Supervised learning uses labeled training data.",
        "Unsupervised learning finds patterns in unlabeled data.",
        "Transfer learning reuses pre-trained models.",
        "Generative AI creates new content based on training data.",
    ]
    
    queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "Explain natural language processing",
    ]
    
    # Ground truth (relevant document indices for each query)
    ground_truth = [
        {0, 1},  # Query 0: ML-related docs
        {1, 2},  # Query 1: Neural network docs
        {3, 4},  # Query 2: NLP and related docs
    ]
    
    # Create benchmark runner
    runner = BenchmarkRunner("benchmarks/datasets/sample_qa.json")
    
    # Run benchmarks
    vanilla_results = runner.run_vanilla_rag(documents, queries, ground_truth)
    refrag_results = runner.run_refrag(documents, queries, ground_truth)
    
    # Print summary
    runner.print_summary(vanilla_results, refrag_results)
    
    # Save results
    output_dir = Path("benchmarks/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "latest_results.json", "w") as f:
        json.dump({
            "vanilla_rag": vanilla_results,
            "refrag": refrag_results
        }, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'latest_results.json'}")


if __name__ == "__main__":
    main()

