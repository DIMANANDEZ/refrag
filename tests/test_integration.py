"""
Integration tests for the full Refrag pipeline.
"""

import pytest
import numpy as np
from refrag.core.embedder import Embedder
from refrag.core.retriever import Retriever
from refrag.core.reranker import Reranker
from refrag.utils.chunking import chunk_text, ChunkingStrategy
from refrag.utils.metrics import evaluate_retrieval


class TestIntegration:
    """Integration tests for the complete RAG pipeline."""
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for integration testing."""
        return [
            "Python is a high-level programming language.",
            "Machine learning is a subset of artificial intelligence.",
            "Neural networks are inspired by the human brain.",
            "Deep learning uses multiple layers of neural networks.",
            "Natural language processing helps computers understand text.",
        ]
    
    @pytest.fixture
    def embedder(self):
        """Create embedder instance."""
        return Embedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            normalize=True
        )
    
    def test_full_pipeline_without_reranking(self, embedder, sample_documents):
        """Test the full pipeline without reranking."""
        # Embed documents
        doc_embeddings = embedder.embed_documents(sample_documents)
        
        # Create retriever and add documents
        retriever = Retriever(
            embedding_dim=doc_embeddings.shape[1],
            index_type="flat"
        )
        retriever.add_documents(doc_embeddings, sample_documents)
        
        # Query
        query = "What is machine learning?"
        query_embedding = embedder.embed_query(query)
        docs, scores, metadata = retriever.retrieve(query_embedding, k=3)
        
        # Verify results
        assert len(docs) == 3
        assert len(scores) == 3
        assert all(isinstance(doc, str) for doc in docs)
        assert all(isinstance(score, (float, np.floating)) for score in scores)
    
    def test_full_pipeline_with_reranking(self, embedder, sample_documents):
        """Test the full pipeline with reranking."""
        # Embed documents
        doc_embeddings = embedder.embed_documents(sample_documents)
        
        # Create retriever and add documents
        retriever = Retriever(
            embedding_dim=doc_embeddings.shape[1],
            index_type="flat"
        )
        retriever.add_documents(doc_embeddings, sample_documents)
        
        # Query and retrieve
        query = "Explain neural networks"
        query_embedding = embedder.embed_query(query)
        docs, scores, metadata = retriever.retrieve(query_embedding, k=5)
        
        # Rerank
        reranker = Reranker(method="cross_encoder")
        reranked_docs, reranked_scores, _ = reranker.rerank(
            query, docs, scores, top_k=3
        )
        
        # Verify results
        assert len(reranked_docs) == 3
        assert len(reranked_scores) == 3
        assert all(isinstance(doc, str) for doc in reranked_docs)
    
    def test_pipeline_with_chunking(self, embedder):
        """Test pipeline with document chunking."""
        # Long document
        long_text = """
        Artificial intelligence has revolutionized many fields. Machine learning,
        a subset of AI, enables computers to learn from data. Deep learning uses
        neural networks with multiple layers. Natural language processing helps
        computers understand human language. Computer vision enables image recognition.
        """
        
        # Chunk the document
        chunks = chunk_text(
            long_text,
            strategy=ChunkingStrategy.SENTENCE,
            chunk_size=100
        )
        
        assert len(chunks) > 0
        
        # Embed chunks
        chunk_embeddings = embedder.embed_documents(chunks)
        
        # Create retriever
        retriever = Retriever(
            embedding_dim=chunk_embeddings.shape[1],
            index_type="flat"
        )
        retriever.add_documents(chunk_embeddings, chunks)
        
        # Query
        query = "What is deep learning?"
        query_embedding = embedder.embed_query(query)
        docs, scores, _ = retriever.retrieve(query_embedding, k=2)
        
        assert len(docs) == 2
        assert any("deep learning" in doc.lower() for doc in docs)
    
    def test_evaluation_metrics(self, embedder, sample_documents):
        """Test evaluation metrics integration."""
        # Setup retrieval
        doc_embeddings = embedder.embed_documents(sample_documents)
        retriever = Retriever(
            embedding_dim=doc_embeddings.shape[1],
            index_type="flat"
        )
        retriever.add_documents(doc_embeddings, sample_documents)
        
        # Query
        query = "Tell me about machine learning"
        query_embedding = embedder.embed_query(query)
        docs, scores, _ = retriever.retrieve(query_embedding, k=3)
        
        # Simulate ground truth
        retrieved_indices = [str(sample_documents.index(doc)) for doc in docs]
        relevant_indices = {"1", "2"}  # Indices of relevant documents
        
        # Evaluate
        metrics = evaluate_retrieval(
            retrieved_indices,
            relevant_indices,
            k_values=[1, 3]
        )
        
        assert "precision@1" in metrics
        assert "recall@1" in metrics
        assert "mrr" in metrics
        assert all(0 <= v <= 1 for v in metrics.values())
    
    def test_batch_processing(self, embedder, sample_documents):
        """Test batch processing of queries."""
        # Setup
        doc_embeddings = embedder.embed_documents(sample_documents)
        retriever = Retriever(
            embedding_dim=doc_embeddings.shape[1],
            index_type="flat"
        )
        retriever.add_documents(doc_embeddings, sample_documents)
        
        # Multiple queries
        queries = [
            "What is Python?",
            "Explain machine learning",
            "How do neural networks work?"
        ]
        
        # Batch embed and retrieve
        query_embeddings = embedder.batch_embed(queries)
        results = retriever.batch_retrieve(query_embeddings, k=2)
        
        assert len(results) == len(queries)
        for docs, scores, metadata in results:
            assert len(docs) == 2
            assert len(scores) == 2
    
    def test_end_to_end_accuracy(self, embedder):
        """Test end-to-end accuracy with known queries and documents."""
        documents = [
            "The capital of France is Paris.",
            "The capital of Germany is Berlin.",
            "The capital of Italy is Rome.",
            "Paris is known for the Eiffel Tower.",
            "Berlin is known for the Brandenburg Gate.",
        ]
        
        # Setup
        doc_embeddings = embedder.embed_documents(documents)
        retriever = Retriever(
            embedding_dim=doc_embeddings.shape[1],
            index_type="flat"
        )
        retriever.add_documents(doc_embeddings, documents)
        
        # Query that should clearly match first document
        query = "What is the capital city of France?"
        query_embedding = embedder.embed_query(query)
        docs, scores, _ = retriever.retrieve(query_embedding, k=1)
        
        # The most relevant document should be retrieved
        assert "France" in docs[0] or "Paris" in docs[0]

