"""
Unit tests for the Embedder module.
"""

import pytest
import numpy as np
from refrag.core.embedder import Embedder


class TestEmbedder:
    """Test cases for Embedder class."""
    
    @pytest.fixture
    def embedder(self):
        """Create an embedder instance for testing."""
        return Embedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            normalize=True
        )
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            "This is the first document.",
            "This is the second document.",
            "And this is the third one."
        ]
    
    def test_embedder_initialization(self, embedder):
        """Test that embedder initializes correctly."""
        assert embedder is not None
        assert embedder.model is not None
        assert embedder.normalize is True
    
    def test_embed_documents(self, embedder, sample_documents):
        """Test document embedding generation."""
        embeddings = embedder.embed_documents(sample_documents)
        
        assert embeddings is not None
        assert embeddings.shape[0] == len(sample_documents)
        assert embeddings.shape[1] > 0  # Has embedding dimension
        assert isinstance(embeddings, np.ndarray)
    
    def test_embed_query(self, embedder):
        """Test query embedding generation."""
        query = "This is a test query."
        embedding = embedder.embed_query(query)
        
        assert embedding is not None
        assert len(embedding.shape) == 1  # 1D array
        assert embedding.shape[0] > 0
        assert isinstance(embedding, np.ndarray)
    
    def test_embedding_normalization(self, embedder):
        """Test that embeddings are normalized when specified."""
        documents = ["Test document for normalization."]
        embeddings = embedder.embed_documents(documents)
        
        # Check if normalized (L2 norm should be close to 1)
        norm = np.linalg.norm(embeddings[0])
        assert np.isclose(norm, 1.0, atol=1e-5)
    
    def test_batch_embed(self, embedder, sample_documents):
        """Test batch embedding."""
        embeddings = embedder.batch_embed(sample_documents, batch_size=2)
        
        assert embeddings.shape[0] == len(sample_documents)
        assert embeddings.shape[1] > 0
    
    def test_embedding_consistency(self, embedder):
        """Test that same input produces same embedding."""
        text = "Consistent embedding test."
        
        embedding1 = embedder.embed_query(text)
        embedding2 = embedder.embed_query(text)
        
        assert np.allclose(embedding1, embedding2)
    
    def test_empty_documents(self, embedder):
        """Test handling of empty documents."""
        # Should handle gracefully, though behavior depends on implementation
        empty_docs = [""]
        embeddings = embedder.embed_documents(empty_docs)
        
        assert embeddings is not None
        assert embeddings.shape[0] == 1

