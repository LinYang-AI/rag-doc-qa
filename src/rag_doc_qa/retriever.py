"""
Hybrid retrieval module combining vector search and BM25.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from rank_bm25 import BM25Okapi
import re

from rag_doc_qa.config import RetrieverConfig
from rag_doc_qa.indexer import FAISSIndex

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Combines dense vector retrieval (FAISS) with sparse lexical retrieval (BM25).
    """

    def __init__(
        self,
        config: Optional[RetrieverConfig] = None,
        index: Optional[FAISSIndex] = None,
    ):
        self.config = config or RetrieverConfig()
        self.index = index
        self.bm25 = None
        self.corpus = []

        # Initialize BM25 if index has data
        if self.index and self.index.metadata:
            self._initialize_bm25()

    def _initialize_bm25(self):
        """Initialize BM25 index from metadata."""
        if not self.index or not self.index.metadata:
            logger.warning("No metadata available for BM25 initialization")
            return

        # Tokenize documents for BM25
        self.corpus = []
        for meta in self.index.metadata:
            tokens = self._tokenize(meta["text"])
            self.corpus.append(tokens)

        self.bm25 = BM25Okapi(self.corpus)
        logger.info(f"BM25 index initialized with {len(self.corpus)} documents")

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        # Convert to lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r"\w+", text)
        return tokens

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using hybrid search.

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            List of retrieved chunks with scores
        """
        if top_k is None:
            top_k = self.config.top_k

        if not self.index or self.index.index is None:
            logger.error("Index not initialized")
            return []

        # Get vector search results
        vector_results = self.index.search(query, top_k * 2)  # Get more for reranking

        # Get BM25 results
        bm25_results = []
        if self.bm25:
            query_tokens = self._tokenize(query)
            bm25_scores = self.bm25.get_scores(query_tokens)

            # Get top BM25 results
            top_indices = np.argsort(bm25_scores)[-top_k * 2 :][::-1]

            for idx in top_indices:
                if idx < len(self.index.metadata):
                    result = self.index.metadata[idx].copy()
                    result["bm25_score"] = float(bm25_scores[idx])
                    bm25_results.append(result)

        # Combine and rerank results
        combined_results = self._combine_results(vector_results, bm25_results)

        # Apply minimum score threshold
        if self.config.min_score > 0:
            combined_results = [
                r
                for r in combined_results
                if r.get("combined_score", 0) >= self.config.min_score
            ]

        # Return top-k results
        return combined_results[:top_k]

    def _combine_results(
        self, vector_results: List[Dict], bm25_results: List[Dict]
    ) -> List[Dict]:
        """
        Combine and rerank results from vector and BM25 search.

        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search

        Returns:
            Combined and reranked results
        """
        # Create result dictionary keyed by chunk_id
        combined = {}

        # Normalize vector scores (assuming cosine similarity in [-1, 1])
        vector_scores = {}
        if vector_results:
            max_vector_score = max(r.get("score", 0) for r in vector_results)
            min_vector_score = min(r.get("score", 0) for r in vector_results)

            for result in vector_results:
                chunk_id = result["chunk_id"]
                if max_vector_score > min_vector_score:
                    normalized_score = (result["score"] - min_vector_score) / (
                        max_vector_score - min_vector_score
                    )
                else:
                    normalized_score = 1.0

                vector_scores[chunk_id] = normalized_score
                combined[chunk_id] = result.copy()

        # Normalize BM25 scores
        bm25_scores = {}
        if bm25_results:
            max_bm25_score = max(r.get("bm25_score", 0) for r in bm25_results)
            min_bm25_score = min(r.get("bm25_score", 0) for r in bm25_results)

            for result in bm25_results:
                chunk_id = result["chunk_id"]
                if max_bm25_score > min_bm25_score:
                    normalized_score = (result["bm25_score"] - min_bm25_score) / (
                        max_bm25_score - min_bm25_score
                    )
                else:
                    normalized_score = 1.0

                bm25_scores[chunk_id] = normalized_score

                if chunk_id not in combined:
                    combined[chunk_id] = result.copy()

        # Calculate combined scores
        for chunk_id, result in combined.items():
            vector_score = vector_scores.get(chunk_id, 0)
            bm25_score = bm25_scores.get(chunk_id, 0)

            # Weighted combination
            combined_score = (
                self.config.vector_weight * vector_score
                + self.config.bm25_weight * bm25_score
            )

            result["vector_score"] = vector_score
            result["bm25_score"] = bm25_score
            result["combined_score"] = combined_score

        # Sort by combined score
        sorted_results = sorted(
            combined.values(), key=lambda x: x["combined_score"], reverse=True
        )

        return sorted_results

    def rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Rerank results using additional signals.
        Can be extended with cross-encoders or other reranking models.

        Args:
            query: Original query
            results: Initial retrieval results

        Returns:
            Reranked results
        """
        if not self.config.rerank:
            return results

        # Simple reranking based on query term overlap
        query_terms = set(self._tokenize(query))

        for result in results:
            text_terms = set(self._tokenize(result["text"]))
            overlap = (
                len(query_terms & text_terms) / len(query_terms) if query_terms else 0
            )

            # Boost score based on term overlap
            result["rerank_boost"] = overlap * 0.1
            result["final_score"] = (
                result.get("combined_score", 0) + result["rerank_boost"]
            )

        # Sort by final score
        reranked = sorted(results, key=lambda x: x.get("final_score", 0), reverse=True)

        return reranked

    def get_context(self, results: List[Dict], max_length: int = 3000) -> str:
        """
        Format retrieval results as context for generation.

        Args:
            results: Retrieved chunks
            max_length: Maximum context length in characters

        Returns:
            Formatted context string
        """
        if not results:
            return ""

        context_parts = []
        current_length = 0

        for i, result in enumerate(results, 1):
            # Format chunk with metadata
            source = result.get("metadata", {}).get("source", "Unknown")
            page = result.get("metadata", {}).get("page", "N/A")

            chunk_text = f"[Source {i}: {source}, Page {page}]\n{result['text']}\n"

            # Check if adding this chunk exceeds max length
            if current_length + len(chunk_text) > max_length:
                # Add truncated version if there's room
                remaining = max_length - current_length
                if remaining > 100:  # Only add if meaningful amount remains
                    truncated = chunk_text[:remaining] + "..."
                    context_parts.append(truncated)
                break

            context_parts.append(chunk_text)
            current_length += len(chunk_text)

        return "\n".join(context_parts)
