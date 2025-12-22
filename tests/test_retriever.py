"""
Unit tests for the Retriever module.
"""

import pytest
import numpy as np
from refrag.core.retriever import Retriever


class TestRetriever:
    """Test cases for Retriever class."""
    
    @pytest.fixture
    def sample_embeddings(self):
        """Generate sample embeddings for testing."""
        np.random.seed(42)
        return np.random.randn(10, 128).astype(np.float32)
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [f"Document {i}" for i in range(10)]
    
    @pytest.fixture
    def retriever(self):
        """Create a retriever instance."""
        return Retriever(
            embedding_dim=128,
            index_type="flat",
            metric="cosine"
        )
    
    def test_retriever_initialization(self, retriever):
        """Test that retriever initializes correctly."""
        assert retriever is not None
        assert retriever.embedding_dim == 128
        assert retriever.index is not None
    
    def test_add_documents(self, retriever, sample_embeddings, sample_documents):
        """Test adding documents to the index."""
        retriever.add_documents(sample_embeddings, sample_documents)
        
        assert len(retriever.documents) == len(sample_documents)
        assert len(retriever.metadata) == len(sample_documents)
    
    def test_retrieve(self, retriever, sample_embeddings, sample_documents):
        """Test document retrieval."""
        retriever.add_documents(sample_embeddings, sample_documents)
        
        query_embedding = sample_embeddings[0]
        docs, scores, metadata = retriever.retrieve(query_embedding, k=3)
        
        assert len(docs) == 3
        assert len(scores) == 3
        assert len(metadata) == 3
        assert all(isinstance(doc, str) for doc in docs)
    
    def test_retrieve_returns_closest(self, retriever, sample_embeddings, sample_documents):
        """Test that retrieval returns the closest documents."""
        retriever.add_documents(sample_embeddings, sample_documents)
        
        # Query with the first embedding should return first document
        query_embedding = sample_embeddings[0]
        docs, scores, _ = retriever.retrieve(query_embedding, k=1)
        
        assert docs[0] == sample_documents[0]
    
    def test_batch_retrieve(self, retriever, sample_embeddings, sample_documents):
        """Test batch retrieval."""
        retriever.add_documents(sample_embeddings, sample_documents)
        
        query_embeddings = sample_embeddings[:3]
        results = retriever.batch_retrieve(query_embeddings, k=2)
        
        assert len(results) == 3
        for docs, scores, metadata in results:
            assert len(docs) == 2
            assert len(scores) == 2
            assert len(metadata) == 2
    
    def test_retrieve_with_metadata(self, retriever, sample_embeddings, sample_documents):
        """Test retrieval with metadata."""
        metadata = [{"id": i, "type": "test"} for i in range(10)]
        retriever.add_documents(sample_embeddings, sample_documents, metadata)
        
        query_embedding = sample_embeddings[0]
        docs, scores, retrieved_metadata = retriever.retrieve(query_embedding, k=3)
        
        assert all(isinstance(m, dict) for m in retrieved_metadata)
        assert all("id" in m for m in retrieved_metadata)
    
    def test_empty_index_retrieval(self, retriever):
        """Test retrieval from empty index."""
        query_embedding = np.random.randn(128).astype(np.float32)
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises(Exception):
            retriever.retrieve(query_embedding, k=5)
    
    def test_different_index_types(self, sample_embeddings, sample_documents):
        """Test different index types."""
        for index_type in ["flat", "hnsw"]:
            retriever = Retriever(
                embedding_dim=128,
                index_type=index_type,
                metric="cosine"
            )
            retriever.add_documents(sample_embeddings, sample_documents)
            
            query_embedding = sample_embeddings[0]
            docs, scores, _ = retriever.retrieve(query_embedding, k=3)
            
            assert len(docs) == 3

