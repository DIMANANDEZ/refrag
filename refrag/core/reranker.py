# refrag/core/reranker.py
"""
REFRAG Reranker: Optional LLM-based reranking of results
"""

from typing import List, Dict, Any, Optional
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


class REFRAGReranker:
    """
    Optional reranking using LLM to score relevance.
    
    Use this when you need highest precision for top results.
    """
    
    RERANK_PROMPT = """Given this query and document, rate the relevance on a scale of 0-10.
Only return the number.

Query: {query}

Document: {document}

Relevance score (0-10):"""
    
    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: Optional[str] = None,
        api_key: Optional[str] = None,
        rerank_prompt: Optional[str] = None
    ):
        """
        Initialize REFRAG Reranker.
        
        Args:
            llm_provider: "openai" or "anthropic"
            llm_model: LLM model name
            api_key: API key for LLM
            rerank_prompt: Custom reranking prompt
        """
        self.llm_provider = llm_provider.lower()
        
        # Set default model
        if llm_model is None:
            if self.llm_provider == "openai":
                self.llm_model = "gpt-4o-mini"
            elif self.llm_provider == "anthropic":
                self.llm_model = "claude-3-5-haiku-20241022"
        else:
            self.llm_model = llm_model
        
        # Initialize client
        self._init_llm_client(api_key)
        
        self.rerank_prompt = rerank_prompt or self.RERANK_PROMPT
    
    def _init_llm_client(self, api_key: Optional[str]):
        """Initialize the LLM client."""
        if self.llm_provider == "openai":
            if not HAS_OPENAI:
                raise ImportError("OpenAI package not installed")
            self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        
        elif self.llm_provider == "anthropic":
            if not HAS_ANTHROPIC:
                raise ImportError("Anthropic package not installed")
            self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
    
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank results using LLM scoring.
        
        Args:
            query: Original search query
            results: List of retrieval results
            top_k: Return only top-k after reranking (optional)
            
        Returns:
            Reranked results with 'rerank_score' field added
        """
        scored_results = []
        
        for result in results:
            score = self._score_relevance(query, result['text'])
            result_copy = result.copy()
            result_copy['rerank_score'] = score
            scored_results.append(result_copy)
        
        # Sort by rerank score
        scored_results.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        if top_k is not None:
            scored_results = scored_results[:top_k]
        
        return scored_results
    
    def _score_relevance(self, query: str, document: str) -> float:
        """Score relevance of document to query."""
        prompt = self.rerank_prompt.format(query=query, document=document)
        
        if self.llm_provider == "openai":
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            score_text = response.choices[0].message.content.strip()
        
        elif self.llm_provider == "anthropic":
            response = self.client.messages.create(
                model=self.llm_model,
                max_tokens=10,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )
            score_text = response.content[0].text.strip()
        
        # Parse score
        try:
            score = float(score_text)
            return max(0.0, min(10.0, score))  # Clamp to 0-10
        except ValueError:
            return 5.0  # Default middle score if parsing fails