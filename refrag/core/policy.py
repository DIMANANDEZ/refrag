# refrag/core/policy.py
"""
REFRAG Compression Policy: Decides which chunks to keep RAW vs COMPRESSED

This happens at QUERY TIME, not during indexing.
"""

from typing import List
import numpy as np


class CompressionPolicy:
    """
    Heuristic-based compression policy for REFRAG.

    Decides which retrieved chunks should be:
    - RAW: High relevance, keep full text
    - COMPRESSED: Lower relevance, compress to keywords

    This is a simplified version. The paper uses RL-trained policy,
    but heuristic works well for MVP.
    """

    def __init__(
        self,
        raw_percentage: float = 0.3,
        min_raw_chunks: int = 2,
        similarity_threshold: float = 0.5
    ):
        """
        Initialize compression policy.

        Args:
            raw_percentage: Percentage of top chunks to keep RAW (default: 30%)
            min_raw_chunks: Minimum number of chunks to keep RAW
            similarity_threshold: Minimum similarity for RAW consideration
        """
        self.raw_percentage = raw_percentage
        self.min_raw_chunks = min_raw_chunks
        self.similarity_threshold = similarity_threshold

    def decide(
        self,
        similarities: np.ndarray,
        top_k: int
    ) -> List[bool]:
        """
        Decide compression strategy for retrieved chunks.

        Args:
            similarities: Similarity scores for all retrieved chunks
            top_k: Number of chunks to consider

        Returns:
            List of bools: True = RAW, False = COMPRESSED
        """
        # Calculate how many chunks should be RAW
        num_raw = max(
            self.min_raw_chunks,
            int(top_k * self.raw_percentage)
        )

        # Calculate threshold for RAW chunks
        if len(similarities) > num_raw:
            threshold = np.partition(similarities, -num_raw)[-num_raw]
        else:
            threshold = self.similarity_threshold

        # Decide: RAW if similarity is high enough
        decisions = [
            (sim >= threshold and sim >= self.similarity_threshold)
            for sim in similarities
        ]

        return decisions

    def decide_with_budget(
        self,
        similarities: np.ndarray,
        token_budget: int,
        chunk_sizes: List[int],
        compression_ratio: float = 0.3
    ) -> List[bool]:
        """
        Decide compression with token budget constraint.

        More sophisticated policy that respects context window limits.

        Args:
            similarities: Similarity scores
            token_budget: Maximum tokens allowed in context
            chunk_sizes: Token count for each chunk
            compression_ratio: How much compression reduces size

        Returns:
            List of bools: True = RAW, False = COMPRESSED
        """
        # Sort chunks by similarity (descending)
        sorted_indices = np.argsort(similarities)[::-1]

        decisions = [False] * len(similarities)
        current_tokens = 0

        for idx in sorted_indices:
            chunk_tokens = chunk_sizes[idx]

            # Can we fit this chunk as RAW?
            if current_tokens + chunk_tokens <= token_budget:
                decisions[idx] = True  # RAW
                current_tokens += chunk_tokens
            else:
                # Try compressed version
                compressed_tokens = int(chunk_tokens * compression_ratio)
                if current_tokens + compressed_tokens <= token_budget:
                    decisions[idx] = False  # COMPRESSED
                    current_tokens += compressed_tokens
                else:
                    # Skip this chunk entirely
                    pass

        return decisions
