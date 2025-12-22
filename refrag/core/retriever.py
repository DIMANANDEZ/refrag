# refrag/core/retriever.py
"""
REFRAG Retriever: Optimized retrieval using representations
"""

from typing import List, Dict, Any, Optional
import numpy as np
from refrag.core.embedder import REFRAGEmbedder


class REFRAGRetriever:
    """
    REFRAG Retriever: Uses representation-based embeddings for retrieval.
    """
    
    def __init__(
        self,
        embedder: Optional[REFRAGEmbedder] = None,
        **embedder_kwargs
    ):
        """
        Initialize REFRAG Retriever.
        
        Args:
            embedder: Pre-configured REFRAGEmbedder (optional)
            **embedder_kwargs: Arguments to pass to REFRAGEmbedder if embedder not provided
        """
        self.embedder = embedder or REFRAGEmbedder(**embedder_kwargs)
        self.document_data = None
    
    def index(
        self,
        documents: List[str],
        show_progress: bool = False,
        batch_size: int = 10
    ):
        """
        Index documents by generating representations and embeddings.
        
        Args:
            documents: List of document chunks to index
            show_progress: Show progress during indexing
            batch_size: Batch size for processing
        """
        print(f"Indexing {len(documents)} documents with REFRAG...")
        
        self.document_data = self.embedder.batch_embed_with_representations(
            documents,
            batch_size=batch_size,
            show_progress=show_progress
        )
        
        print(f"✓ Indexed {len(documents)} documents")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        return_scores: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            return_scores: Include similarity scores in results
            
        Returns:
            List of result dictionaries containing:
                - 'text': Original document text
                - 'representation': LLM-generated representation
                - 'score': Similarity score (if return_scores=True)
        """
        if self.document_data is None:
            raise ValueError("No documents indexed. Call .index() first.")
        
        # Embed query
        query_embedding = self.embedder.embed_query(query)
        
        # Compute similarities
        similarities = self._cosine_similarity(
            query_embedding,
            self.document_data['embeddings']
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Build results
        results = []
        for idx in top_indices:
            result = {
                'text': self.document_data['original'][idx],
                'representation': self.document_data['representations'][idx]
            }
            if return_scores:
                result['score'] = float(similarities[idx])
            results.append(result)
        
        return results
    
    @staticmethod
    def _cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and document vectors."""
        # Assuming vectors are already normalized
        return np.dot(doc_vecs, query_vec)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed documents."""
        if self.document_data is None:
            return {"indexed": False}
        
        return {
            "indexed": True,
            "num_documents": len(self.document_data['original']),
            "embedding_dim": self.document_data['embeddings'].shape[1],
            **self.embedder.get_cache_stats()
        }