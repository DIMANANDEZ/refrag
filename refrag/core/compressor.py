# refrag/core/compressor.py
"""
REFRAG Chunk Compressor: Extracts keywords from low-priority chunks

Uses lightweight NLP for fast compression at query time.
"""

from typing import List, Set
import re


class ChunkCompressor:
    """
    Fast keyword-based compression for REFRAG.

    Compresses low-priority chunks to save context window space.
    Uses simple but effective heuristics.
    """

    def __init__(
        self,
        compression_method: str = "keywords",
        max_keywords: int = 5
    ):
        """
        Initialize chunk compressor.

        Args:
            compression_method: "keywords", "first_n", or "entities"
            max_keywords: Maximum keywords to extract per chunk
        """
        self.compression_method = compression_method
        self.max_keywords = max_keywords

        # Critical semantic words that should NEVER be filtered
        # These words change meaning and must be preserved
        self.preserve_words = {
            # Negation (CRITICAL: "do not press" vs "do press")
            'not', 'no', 'never', 'neither', 'nor', 'none', "n't", "don't", "can't",
            "won't", "wouldn't", "shouldn't", "couldn't", "isn't", "aren't", "wasn't", "weren't",
            # Exception/Exclusion
            'except', 'without', 'unless', 'only', 'but',
            # Temporal (order matters)
            'before', 'after', 'until', 'since', 'while', 'when',
            # Comparison
            'than', 'more', 'less', 'most', 'least',
            # Modality (requirements)
            'must', 'required', 'forbidden', 'prohibited'
        }

        # Common stopwords to filter out (non-semantic function words)
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }

    def compress(self, text: str) -> str:
        """
        Compress text to keywords.

        Args:
            text: Text to compress

        Returns:
            Compressed representation (keywords)
        """
        if self.compression_method == "keywords":
            return self._extract_keywords(text)
        elif self.compression_method == "first_n":
            return self._first_n_words(text)
        elif self.compression_method == "entities":
            return self._extract_entities(text)
        else:
            return self._extract_keywords(text)

    def compress_batch(self, texts: List[str]) -> List[str]:
        """
        Compress multiple chunks.

        Args:
            texts: List of text chunks

        Returns:
            List of compressed representations
        """
        return [self.compress(text) for text in texts]

    def _extract_keywords(self, text: str) -> str:
        """
        Extract important keywords using simple heuristics.

        Focus on:
        - Critical semantic words (negations, modality, temporal)
        - Capitalized words (likely names/entities)
        - Longer words (likely content words)
        - Non-stopwords
        """
        # Tokenize
        words = re.findall(r'\b\w+\b', text)

        # First, collect all preserve_words that appear in text (ALWAYS include these)
        preserved = []
        for word in words:
            word_lower = word.lower()
            # Check for contractions (e.g., "don't")
            if word_lower in self.preserve_words or any(contr in word_lower for contr in ["n't"]):
                if word not in preserved:  # Deduplicate
                    preserved.append(word)

        # Score remaining words
        scored_words = []
        for word in words:
            word_lower = word.lower()

            # Skip words already in preserved list
            if word in preserved:
                continue

            # Skip very short words unless they're critical
            if len(word) < 3:
                continue

            score = 0

            # CRITICAL: Bonus for preserve_words (negation, modality, etc.)
            if word_lower in self.preserve_words:
                score += 100  # Ensure these are always selected

            # Bonus for capitalization (entities)
            if word[0].isupper():
                score += 2

            # Bonus for length (content words)
            score += len(word) * 0.1

            # Penalty for stopwords (but NOT for preserve_words)
            if word_lower in self.stopwords and word_lower not in self.preserve_words:
                score -= 10

            scored_words.append((word, score))

        # Sort by score and take top keywords
        scored_words.sort(key=lambda x: x[1], reverse=True)

        # Calculate remaining slots after preserved words
        remaining_slots = max(0, self.max_keywords - len(preserved))
        top_keywords = [word for word, score in scored_words[:remaining_slots]]

        # Combine: preserved words FIRST (maintain order), then top keywords
        all_keywords = preserved + top_keywords

        return ' '.join(all_keywords)

    def _extract_entities(self, text: str) -> str:
        """
        Extract named entities (simple version: capitalized words).
        """
        # Find capitalized words (likely proper nouns)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)

        # Deduplicate while preserving order
        seen: Set[str] = set()
        unique_entities = []
        for entity in entities:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)

        return ' '.join(unique_entities[:self.max_keywords])

    def _first_n_words(self, text: str, n: int = 10) -> str:
        """
        Simple compression: take first N words.
        """
        words = text.split()
        return ' '.join(words[:n])
