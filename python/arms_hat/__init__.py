"""
ARMS-HAT: Hierarchical Attention Tree for AI memory retrieval.

A semantic memory index optimized for LLM conversation history.

Example:
    >>> from arms_hat import HatIndex
    >>>
    >>> # Create index for OpenAI embeddings (1536 dims)
    >>> index = HatIndex.cosine(1536)
    >>>
    >>> # Add embeddings
    >>> id1 = index.add([0.1] * 1536)
    >>>
    >>> # Query
    >>> results = index.near([0.1] * 1536, k=10)
    >>> for r in results:
    ...     print(f"{r.id}: {r.score}")
    >>>
    >>> # Session management
    >>> index.new_session()
    >>>
    >>> # Persistence
    >>> index.save("memory.hat")
    >>> loaded = HatIndex.load("memory.hat")
"""

from .arms_hat import (
    HatIndex,
    HatConfig,
    SearchResult,
    SessionSummary,
    DocumentSummary,
    HatStats,
)

__all__ = [
    "HatIndex",
    "HatConfig",
    "SearchResult",
    "SessionSummary",
    "DocumentSummary",
    "HatStats",
]

__version__ = "0.1.0"
