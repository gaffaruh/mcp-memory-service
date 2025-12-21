# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Cross-encoder reranking for improved retrieval precision.

Based on AnythingLLM best practices:
1. Initial vector search returns top N candidates (e.g., 20)
2. Cross-encoder reranks using pairwise query-document scoring
3. Return top K results (e.g., 5) with improved precision

Configuration via environment variables:
- MCP_MEMORY_RERANK_ENABLED: Enable reranking (default: false)
- MCP_MEMORY_RERANK_MODEL: Cross-encoder model name (default: cross-encoder/ms-marco-MiniLM-L-6-v2)
- MCP_MEMORY_RERANK_TOP_K: Number of results to return after reranking (default: 5)
- MCP_MEMORY_RERANK_CANDIDATES: Number of candidates to fetch for reranking (default: 20)
"""

import os
import logging
from typing import List, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Safe int parsing helper
def _safe_int(env_var: str, default: int) -> int:
    """Parse int from env with fallback to default on error."""
    try:
        return int(os.getenv(env_var, str(default)))
    except (ValueError, TypeError):
        logger.warning(f"Invalid value for {env_var}, using default {default}")
        return default


# Configuration from environment
RERANK_ENABLED = os.getenv("MCP_MEMORY_RERANK_ENABLED", "false").lower() == "true"
RERANK_MODEL = os.getenv("MCP_MEMORY_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_TOP_K = _safe_int("MCP_MEMORY_RERANK_TOP_K", 5)
RERANK_CANDIDATES = _safe_int("MCP_MEMORY_RERANK_CANDIDATES", 20)


@dataclass
class RerankResult:
    """Result of reranking operation."""
    content: str
    original_score: float
    rerank_score: float
    original_data: Any  # Original memory/result object
    was_reranked: bool = True  # Whether this result was actually reranked


class CrossEncoderReranker:
    """
    Cross-encoder based reranker for semantic search results.

    Uses sentence-transformers CrossEncoder for pairwise query-document scoring.
    Model is lazy-loaded on first use to avoid startup overhead when disabled.
    """

    def __init__(self, model_name: str = None):
        """
        Initialize the reranker.

        Args:
            model_name: Cross-encoder model name (default from env or ms-marco-MiniLM-L-6-v2)
        """
        self.model_name = model_name or RERANK_MODEL
        self._model = None
        self._initialized = False

    def _ensure_model(self) -> bool:
        """
        Lazy-load the cross-encoder model.

        Returns:
            True if model is ready, False if loading failed
        """
        if self._initialized:
            return self._model is not None

        self._initialized = True

        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
            logger.info(f"Cross-encoder model loaded successfully")
            return True
        except ImportError:
            logger.warning("sentence-transformers not available for cross-encoder reranking")
            return False
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")
            return False

    def rerank(
        self,
        query: str,
        results: List[Tuple[str, float, Any]],
        top_k: int = None,
        force: bool = False
    ) -> List[RerankResult]:
        """
        Rerank search results using cross-encoder.

        Args:
            query: Search query string
            results: List of (content, original_score, original_data) tuples
            top_k: Number of top results to return (default from env)
            force: If True, attempt reranking even if globally disabled

        Returns:
            List of RerankResult sorted by rerank_score descending.
            Each result includes was_reranked=True/False to indicate if
            cross-encoder scoring was actually applied.
        """
        if not results:
            return []

        top_k = top_k or RERANK_TOP_K

        # Check if reranking should be attempted
        should_rerank = force or RERANK_ENABLED

        # If reranking is disabled or model unavailable, return original order
        if not should_rerank or not self._ensure_model():
            return [
                RerankResult(
                    content=content,
                    original_score=score,
                    rerank_score=score,  # Use original score
                    original_data=data,
                    was_reranked=False
                )
                for content, score, data in results[:top_k]
            ]

        try:
            # Create query-document pairs for cross-encoder
            pairs = [(query, content) for content, _, _ in results]

            # Get cross-encoder scores
            scores = self._model.predict(pairs)

            # Combine with original data and sort by rerank score
            reranked = []
            for (content, orig_score, data), rerank_score in zip(results, scores):
                reranked.append(RerankResult(
                    content=content,
                    original_score=orig_score,
                    rerank_score=float(rerank_score),
                    original_data=data,
                    was_reranked=True
                ))

            # Sort by rerank score descending
            reranked.sort(key=lambda x: x.rerank_score, reverse=True)

            logger.debug(f"Reranked {len(results)} results, returning top {top_k}")
            return reranked[:top_k]

        except Exception as e:
            logger.error(f"Reranking failed: {e}, falling back to original order")
            return [
                RerankResult(
                    content=content,
                    original_score=score,
                    rerank_score=score,
                    original_data=data,
                    was_reranked=False  # Fallback - not actually reranked
                )
                for content, score, data in results[:top_k]
            ]


# Singleton instance for global access
_reranker_instance: Optional[CrossEncoderReranker] = None


def get_reranker() -> CrossEncoderReranker:
    """
    Get the global reranker instance.

    Returns:
        CrossEncoderReranker singleton instance
    """
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker()
    return _reranker_instance


def is_rerank_enabled() -> bool:
    """Check if reranking is enabled via configuration."""
    return RERANK_ENABLED


def get_rerank_candidates() -> int:
    """Get the number of candidates to fetch for reranking."""
    return RERANK_CANDIDATES


def get_rerank_top_k() -> int:
    """Get the number of results to return after reranking."""
    return RERANK_TOP_K
