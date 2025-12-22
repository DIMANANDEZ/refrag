# refrag/core/embedder.py
"""
REFRAG Embedder: Representation-Focused RAG
"""

from typing import List, Optional, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import os

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


class REFRAGEmbedder:
    """
    REFRAG: Generates LLM-powered representations before embedding.
    
    Key difference from vanilla RAG:
    1. Use LLM to create task-specific representation of each chunk
    2. Embed the representation instead of raw text
    3. Results in better retrieval relevance + smaller context
    """
    
    DEFAULT_REPRESENTATION_PROMPT = """Extract and condense the key information from this text chunk that would be useful for answering questions. 
Focus on facts, entities, and relationships. Keep it concise (2-3 sentences).

Text: {chunk}

Condensed representation:"""
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_provider: str = "openai",  # "openai" or "anthropic"
        llm_model: Optional[str] = None,
        api_key: Optional[str] = None,
        representation_prompt: Optional[str] = None,
        device: Optional[str] = None,
        normalize: bool = True,
        cache_representations: bool = True
    ):
        """
        Initialize REFRAG Embedder.
        
        Args:
            embedding_model: Model for embedding representations
            llm_provider: "openai" or "anthropic"
            llm_model: LLM model name (defaults based on provider)
            api_key: API key for LLM (or set via env var)
            representation_prompt: Custom prompt for representation generation
            device: Device for embedding model
            normalize: Normalize embeddings
            cache_representations: Cache LLM-generated representations
        """
        self.llm_provider = llm_provider.lower()
        self.normalize = normalize
        self.cache = {} if cache_representations else None
        
        # Set default LLM model based on provider
        if llm_model is None:
            if self.llm_provider == "openai":
                self.llm_model = "gpt-4o-mini"
            elif self.llm_provider == "anthropic":
                self.llm_model = "claude-3-5-haiku-20241022"
            else:
                raise ValueError(f"Unsupported LLM provider: {llm_provider}")
        else:
            self.llm_model = llm_model
        
        # Initialize LLM client
        self._init_llm_client(api_key)
        
        # Load embedding model
        if embedding_model.startswith("sentence-transformers/"):
            model_name = embedding_model.replace("sentence-transformers/", "")
            self.embedding_model = SentenceTransformer(model_name, device=device)
        else:
            self.embedding_model = SentenceTransformer(embedding_model, device=device)
        
        # Set representation prompt
        self.representation_prompt = representation_prompt or self.DEFAULT_REPRESENTATION_PROMPT
    
    def _init_llm_client(self, api_key: Optional[str]):
        """Initialize the LLM client based on provider."""
        if self.llm_provider == "openai":
            if not HAS_OPENAI:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
            self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        
        elif self.llm_provider == "anthropic":
            if not HAS_ANTHROPIC:
                raise ImportError("Anthropic package not installed. Run: pip install anthropic")
            self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    def _generate_representation(self, chunk: str) -> str:
        """
        Generate LLM-powered representation of a chunk.
        
        This is the core REFRAG innovation.
        """
        # Check cache first
        if self.cache is not None and chunk in self.cache:
            return self.cache[chunk]
        
        # Generate representation using LLM
        prompt = self.representation_prompt.format(chunk=chunk)
        
        if self.llm_provider == "openai":
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150
            )
            representation = response.choices[0].message.content.strip()
        
        elif self.llm_provider == "anthropic":
            response = self.client.messages.create(
                model=self.llm_model,
                max_tokens=150,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            representation = response.content[0].text.strip()
        
        # Cache it
        if self.cache is not None:
            self.cache[chunk] = representation
        
        return representation
    
    def embed_documents(
        self,
        documents: List[str],
        show_progress: bool = False
    ) -> Dict[str, Any]:
        """
        Generate REFRAG embeddings for documents.
        
        Args:
            documents: List of document chunks
            show_progress: Show progress during generation
            
        Returns:
            Dict with:
                - 'embeddings': Embeddings of representations (np.ndarray)
                - 'representations': The LLM-generated representations (List[str])
                - 'original': Original document texts (List[str])
        """
        # Step 1: Generate representations using LLM
        representations = []
        for i, doc in enumerate(documents):
            if show_progress:
                print(f"Generating representation {i+1}/{len(documents)}...")
            rep = self._generate_representation(doc)
            representations.append(rep)
        
        # Step 2: Embed the representations (not original docs)
        embeddings = self.embedding_model.encode(
            representations,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress
        )
        
        return {
            'embeddings': np.array(embeddings),
            'representations': representations,
            'original': documents
        }
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed query directly (no representation needed for queries).
        
        Args:
            query: Search query
            
        Returns:
            Query embedding
        """
        embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=self.normalize
        )
        return embedding
    
    def batch_embed_with_representations(
        self,
        documents: List[str],
        batch_size: int = 10,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Batch process documents with LLM representations.
        
        More efficient for large document sets.
        
        Args:
            documents: List of document chunks
            batch_size: Number of documents to process per batch
            show_progress: Show progress during generation
            
        Returns:
            Dict with embeddings, representations, and original texts
        """
        all_representations = []
        
        # Process in batches to avoid rate limits
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            if show_progress:
                print(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}...")
            
            batch_reps = [self._generate_representation(doc) for doc in batch]
            all_representations.extend(batch_reps)
        
        # Embed all representations
        embeddings = self.embedding_model.encode(
            all_representations,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress
        )
        
        return {
            'embeddings': np.array(embeddings),
            'representations': all_representations,
            'original': documents
        }
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about the representation cache."""
        if self.cache is None:
            return {"cache_enabled": False}
        return {
            "cache_enabled": True,
            "cached_items": len(self.cache)
        }