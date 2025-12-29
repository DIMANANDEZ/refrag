"""
REFRAG: Representation-Focused Retrieval Augmented Generation

Correct implementation matching the REFRAG paper:
- Micro-chunking (16-32 tokens)
- Fast direct encoding (no LLM during indexing)
- Query-time compression (RAW vs COMPRESSED)
- Mixed context decoding
"""

__version__ = "1.0.0"

from refrag.core.embedder import REFRAGEmbedder
from refrag.core.retriever import REFRAGRetriever
from refrag.core.reranker import REFRAGReranker
from refrag.core.policy import CompressionPolicy
from refrag.core.compressor import ChunkCompressor
from refrag.core.decoder import MixedContextDecoder
from refrag.utils.chunking import MicroChunker, chunk_text, chunk_documents

__all__ = [
    "REFRAGEmbedder",
    "REFRAGRetriever",
    "REFRAGReranker",
    "CompressionPolicy",
    "ChunkCompressor",
    "MixedContextDecoder",
    "MicroChunker",
    "chunk_text",
    "chunk_documents",
]