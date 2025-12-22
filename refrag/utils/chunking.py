# refrag/utils/chunking.py
"""
Document chunking utilities
"""

from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    separator: str = "\n\n"
) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Target size of each chunk (in characters)
        overlap: Overlap between chunks (in characters)
        separator: Preferred split separator
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    # Try to split on separator first
    if separator in text:
        parts = text.split(separator)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for part in parts:
            part_size = len(part)
            
            if current_size + part_size > chunk_size and current_chunk:
                chunks.append(separator.join(current_chunk))
                # Keep last item for overlap
                if overlap > 0 and current_chunk:
                    current_chunk = [current_chunk[-1]]
                    current_size = len(current_chunk[0])
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(part)
            current_size += part_size
        
        if current_chunk:
            chunks.append(separator.join(current_chunk))
        
        return chunks
    
    # Fallback: simple character-based chunking
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks


def chunk_documents(
    documents: List[str],
    chunk_size: int = 512,
    overlap: int = 50
) -> List[str]:
    """
    Chunk multiple documents.
    
    Args:
        documents: List of document texts
        chunk_size: Target chunk size
        overlap: Overlap between chunks
        
    Returns:
        Flattened list of all chunks
    """
    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(chunks)
    return all_chunks