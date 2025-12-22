"""
REFRAG: Representation-Focused Retrieval Augmented Generation

Early open-source implementation of representation-optimized RAG.
"""

__version__ = "0.1.0"

from refrag.core.embedder import REFRAGEmbedder
from refrag.core.retriever import REFRAGRetriever
from refrag.core.reranker import REFRAGReranker

__all__ = [
    "REFRAGEmbedder",
    "REFRAGRetriever", 
    "REFRAGReranker",
]