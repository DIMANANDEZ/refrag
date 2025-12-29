# refrag/core/embedder.py
"""
REFRAG Embedder: Fast direct encoding

Key difference from traditional RAG:
- Uses micro-chunks (16-32 tokens)
- Direct embedding (NO LLM calls during indexing)
- Fast indexing speed
"""

from typing import List, Optional, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer


class REFRAGEmbedder:
    """
    REFRAG Fast Encoder: Direct embedding.

    This is the correct REFRAG approach:
    1. Embed micro-chunks directly (no LLM during indexing)
    2. Use fast encoder model
    3. Compression happens at query time, not indexing time
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize: bool = True
    ):
        """
        Initialize REFRAG Fast Encoder.

        Args:
            embedding_model: Sentence transformer model for encoding
            device: Device for embedding model (cuda/cpu)
            normalize: Normalize embeddings for cosine similarity
        """
        self.normalize = normalize

        # Load embedding model
        if embedding_model.startswith("sentence-transformers/"):
            model_name = embedding_model.replace("sentence-transformers/", "")
            self.embedding_model = SentenceTransformer(model_name, device=device)
        else:
            self.embedding_model = SentenceTransformer(embedding_model, device=device)

    def embed_documents(
        self,
        documents: List[str],
        show_progress: bool = False,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Fast embedding of document chunks (NO LLM calls).

        Args:
            documents: List of document chunks (micro-chunks)
            show_progress: Show progress bar
            batch_size: Batch size for encoding

        Returns:
            Document embeddings (np.ndarray)
        """
        # Direct encoding - NO LLM involved!
        embeddings = self.embedding_model.encode(
            documents,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress,
            batch_size=batch_size
        )

        return np.array(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed query for retrieval.

        Args:
            query: Search query

        Returns:
            Query embedding
        """
        embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=self.normalize
        )
        return np.array(embedding)

    def batch_embed(
        self,
        documents: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Batch encode documents efficiently.

        Args:
            documents: List of document chunks
            batch_size: Batch size for encoding
            show_progress: Show progress during encoding

        Returns:
            Document embeddings
        """
        return self.embed_documents(
            documents,
            show_progress=show_progress,
            batch_size=batch_size
        )