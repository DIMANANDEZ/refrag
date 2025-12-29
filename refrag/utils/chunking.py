# refrag/utils/chunking.py
"""
Document chunking utilities for REFRAG

REFRAG uses micro-chunking: 16-32 tokens per chunk for fine-grained retrieval
"""

from typing import List
from transformers import AutoTokenizer


class MicroChunker:
    """
    REFRAG Micro-Chunker: Splits text into small token-based chunks (16-32 tokens).

    This is a core component of REFRAG that differs from traditional RAG.
    Smaller chunks enable:
    - Fine-grained retrieval
    - Better compression decisions
    - More precise context building
    """

    def __init__(
        self,
        chunk_size: int = 32,
        tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize MicroChunker.

        Args:
            chunk_size: Number of tokens per chunk (default: 32)
            tokenizer_name: Tokenizer to use for chunking
        """
        self.chunk_size = chunk_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into micro-chunks of fixed token size.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks (each ~chunk_size tokens)
        """
        if not text:
            return []

        # Tokenize the entire text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        # Split into chunks of chunk_size tokens
        chunks = []
        for i in range(0, len(tokens), self.chunk_size):
            chunk_tokens = tokens[i:i + self.chunk_size]
            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            if chunk_text.strip():  # Only add non-empty chunks
                chunks.append(chunk_text.strip())

        return chunks

    def chunk_documents(self, documents: List[str]) -> List[str]:
        """
        Chunk multiple documents into micro-chunks.

        Args:
            documents: List of document texts

        Returns:
            Flattened list of all micro-chunks
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_text(doc)
            all_chunks.extend(chunks)
        return all_chunks


# Backwards compatibility functions
def chunk_text(
    text: str,
    chunk_size: int = 32,
    overlap: int = 0,
    separator: str = "\n\n"
) -> List[str]:
    """
    Legacy function for compatibility. Now uses micro-chunking.

    Args:
        text: Text to chunk
        chunk_size: Number of tokens per chunk (not characters)
        overlap: Ignored (micro-chunking doesn't use overlap)
        separator: Ignored (micro-chunking is token-based)

    Returns:
        List of text chunks
    """
    chunker = MicroChunker(chunk_size=chunk_size)
    return chunker.chunk_text(text)


def chunk_documents(
    documents: List[str],
    chunk_size: int = 32,
    overlap: int = 0
) -> List[str]:
    """
    Legacy function for compatibility. Now uses micro-chunking.

    Args:
        documents: List of document texts
        chunk_size: Number of tokens per chunk (not characters)
        overlap: Ignored (micro-chunking doesn't use overlap)

    Returns:
        Flattened list of all chunks
    """
    chunker = MicroChunker(chunk_size=chunk_size)
    return chunker.chunk_documents(documents)