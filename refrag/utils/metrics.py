# refrag/utils/metrics.py
"""
Evaluation metrics for RAG systems
"""

from typing import List, Dict, Any
import numpy as np


def calculate_metrics(
    retrieved_docs: List[str],
    relevant_docs: List[str],
    k: int = 5
) -> Dict[str, float]:
    """
    Calculate retrieval metrics.
    
    Args:
        retrieved_docs: List of retrieved document IDs
        relevant_docs: List of relevant document IDs
        k: Top-k to consider
        
    Returns:
        Dict of metrics (precision, recall, f1)
    """
    retrieved_set = set(retrieved_docs[:k])
    relevant_set = set(relevant_docs)
    
    if len(retrieved_set) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    if len(relevant_set) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    true_positives = len(retrieved_set & relevant_set)
    
    precision = true_positives / len(retrieved_set)
    recall = true_positives / len(relevant_set)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return {
        "precision@k": precision,
        "recall@k": recall,
        "f1@k": f1,
        "k": k
    }


def mean_reciprocal_rank(retrieved_lists: List[List[str]], relevant_docs: List[str]) -> float:
    """
    Calculate Mean Reciprocal Rank.
    
    Args:
        retrieved_lists: List of retrieved document lists
        relevant_docs: List of relevant documents
        
    Returns:
        MRR score
    """
    reciprocal_ranks = []
    
    for retrieved in retrieved_lists:
        for i, doc in enumerate(retrieved):
            if doc in relevant_docs:
                reciprocal_ranks.append(1.0 / (i + 1))
                break
        else:
            reciprocal_ranks.append(0.0)
    
    return float(np.mean(reciprocal_ranks))