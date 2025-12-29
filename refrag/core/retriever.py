# refrag/core/retriever.py
"""
REFRAG Retriever: Fast retrieval with query-time compression
"""

from typing import List, Dict, Any, Optional
import numpy as np
from refrag.core.embedder import REFRAGEmbedder
from refrag.core.policy import CompressionPolicy
from refrag.core.compressor import ChunkCompressor
from refrag.core.decoder import MixedContextDecoder


class REFRAGRetriever:
    """
    REFRAG Retriever: Implements the complete REFRAG pipeline.

    Key features:
    1. Fast indexing (direct encoding, no LLM)
    2. Query-time compression decisions
    3. Mixed RAW/COMPRESSED context

    Thread Safety:
        ⚠️ WARNING: This retriever is NOT thread-safe for concurrent indexing
        and retrieval operations. Calling .index() while queries are being
        processed can lead to race conditions where self.documents and
        self.embeddings are swapped mid-flight.

        For multi-threaded applications:
        - Index documents once during initialization
        - Use a single retriever instance per thread, OR
        - Use a lock (threading.Lock) to serialize index() calls
        - Concurrent retrieve() calls are safe if index() is not running
    """

    def __init__(
        self,
        embedder: Optional[REFRAGEmbedder] = None,
        compression_policy: Optional[CompressionPolicy] = None,
        compressor: Optional[ChunkCompressor] = None,
        decoder: Optional[MixedContextDecoder] = None,
        **embedder_kwargs
    ):
        """
        Initialize REFRAG Retriever.

        Args:
            embedder: Pre-configured embedder (optional)
            compression_policy: Compression decision policy (optional)
            compressor: Chunk compressor (optional)
            decoder: Context decoder (optional)
            **embedder_kwargs: Arguments for embedder if not provided
        """
        self.embedder = embedder or REFRAGEmbedder(**embedder_kwargs)
        self.policy = compression_policy or CompressionPolicy()
        self.compressor = compressor or ChunkCompressor()
        self.decoder = decoder or MixedContextDecoder()

        self.documents = None
        self.embeddings = None

    def index(
        self,
        documents: List[str],
        show_progress: bool = False,
        batch_size: int = 32
    ):
        """
        Index documents with FAST direct encoding (NO LLM).

        Args:
            documents: List of document chunks (micro-chunks)
            show_progress: Show progress during indexing
            batch_size: Batch size for encoding
        """
        print(f"Indexing {len(documents)} documents with REFRAG (fast mode)...")

        # Store original documents
        self.documents = documents

        # Fast embedding (NO LLM!)
        self.embeddings = self.embedder.embed_documents(
            documents,
            show_progress=show_progress,
            batch_size=batch_size
        )

        print(f"✓ Indexed {len(documents)} documents (fast!)")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        return_scores: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents (basic retrieval, no compression).

        Args:
            query: Search query
            top_k: Number of results
            return_scores: Include similarity scores

        Returns:
            List of result dictionaries with 'text' and optionally 'score'
        """
        if self.documents is None or self.embeddings is None:
            raise ValueError("No documents indexed. Call .index() first.")

        # Embed query
        query_embedding = self.embedder.embed_query(query)

        # Compute similarities
        similarities = self._cosine_similarity(query_embedding, self.embeddings)

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Build results
        results = []
        for idx in top_indices:
            result = {'text': self.documents[idx]}
            if return_scores:
                result['score'] = float(similarities[idx])
            results.append(result)

        return results

    def retrieve_with_compression(
        self,
        query: str,
        top_k: int = 10,
        return_context: bool = True,
        format_style: str = "tagged"
    ) -> Dict[str, Any]:
        """
        Retrieve with REFRAG compression (query-time).

        This is the CORE REFRAG feature:
        1. Retrieve top-k chunks
        2. Apply compression policy (decide RAW vs COMPRESSED)
        3. Compress low-priority chunks
        4. Build mixed context

        Args:
            query: Search query
            top_k: Number of chunks to retrieve
            return_context: Return formatted context string
            format_style: Context format ("tagged", "separated", "inline")

        Returns:
            Dict with:
                - 'chunks': Original chunks
                - 'compressed': Compressed versions
                - 'is_raw': Boolean list (RAW decisions)
                - 'scores': Similarity scores
                - 'context': Formatted context (if return_context=True)
        """
        if self.documents is None or self.embeddings is None:
            raise ValueError("No documents indexed. Call .index() first.")

        # Embed query
        query_embedding = self.embedder.embed_query(query)

        # Compute similarities
        similarities = self._cosine_similarity(query_embedding, self.embeddings)

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_similarities = similarities[top_indices]

        # Get top chunks
        top_chunks = [self.documents[idx] for idx in top_indices]

        # QUERY-TIME COMPRESSION DECISION
        is_raw = self.policy.decide(top_similarities, top_k)

        # Compress ONLY low-priority chunks (optimization: don't compress RAW chunks)
        compressed_chunks = []
        for i, chunk in enumerate(top_chunks):
            if is_raw[i]:
                # RAW chunks don't need compression (decoder won't use it anyway)
                compressed_chunks.append("")  # Placeholder
            else:
                # COMPRESSED chunks: actually compress them
                compressed_chunks.append(self.compressor.compress(chunk))

        # Build result
        result = {
            'chunks': top_chunks,
            'compressed': compressed_chunks,
            'is_raw': is_raw,
            'scores': top_similarities.tolist()
        }

        # Build mixed context if requested
        if return_context:
            self.decoder.format_style = format_style
            context = self.decoder.build_context(
                top_chunks,
                compressed_chunks,
                is_raw,
                include_metadata=False,
                scores=top_similarities
            )
            result['context'] = context

        return result

    @staticmethod
    def _cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and document vectors."""
        # Assuming vectors are already normalized
        return np.dot(doc_vecs, query_vec)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed documents."""
        if self.documents is None:
            return {"indexed": False}

        return {
            "indexed": True,
            "num_documents": len(self.documents),
            "embedding_dim": self.embeddings.shape[1] if self.embeddings is not None else 0
        }