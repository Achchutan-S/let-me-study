"""
Embedding utilities for semantic search functionality.
"""
import os
from typing import List, Optional
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# Configure Gemini API
EMBEDDING_API_KEY = os.getenv('EMBEDDING_API_KEY')
if not EMBEDDING_API_KEY:
    raise RuntimeError('EMBEDDING_API_KEY not configured')

genai.configure(api_key=EMBEDDING_API_KEY)


def get_embedding(text: str, task_type: str = "retrieval_query") -> List[float]:
    """Get embedding vector for text using Gemini API.

    Args:
        text: The text to embed
        task_type: Either "retrieval_query" for search queries or "retrieval_document" for stored content

    Returns:
        List[float]: The embedding vector. Empty list if embedding fails.
    """
    model = 'models/embedding-001'
    try:
        result = genai.embed_content(
            model=model,
            content=text,
            task_type=task_type
        )
        return result['embedding']
    except Exception as e:
        print(f"[Embedding Error] {e}")
        return []


def compute_cosine_similarities(
    query_embedding: List[float],
    stored_embeddings: List[List[float]]
) -> List[float]:
    """Compute cosine similarities between query and stored embeddings."""
    if not stored_embeddings:
        return []

    # Convert to numpy arrays
    query_np = np.array(query_embedding).reshape(1, -1)
    stored_np = np.array(stored_embeddings)

    # Compute similarities
    similarities = cosine_similarity(query_np, stored_np).flatten()
    return similarities.tolist()


def get_top_k_indices(similarities: List[float], k: int = 5) -> List[int]:
    """Get indices of top k similar items."""
    if not similarities:
        return []
    return np.argsort(similarities)[-k:].tolist()[::-1]
