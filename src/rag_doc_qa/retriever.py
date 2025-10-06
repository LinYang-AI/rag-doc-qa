"""
Advanced hybrid retrieval module combining multiple retrieval strategies.
Includes vector search, BM25, and reranking capabilities.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag_doc_qa.config import RetrieverConfig
from rag_doc_qa.indexer import FAISSIndex

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Container for retrieval results with detailed scoring."""
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any]
    vector_score: float = 0.0
    bm25_score: float = 0.0
    combined_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    rank: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "metadata": self.metadata,
            "scores": {
                "vector": self.vector_score,
                "bm25": self.bm25_score,
                "combined": self.combined_score,
                "rerank": self.rerank_score,
                "final": self.final_score
            },
            "rank": self.rank
        }

class HybridRetriever:
    """
    Advanced retriever combining dense and sparse retrieval methods.
    """
    
    def __init__(self,
                 config: Optional[RetrieverConfig] = None,
                 index: Optional[FAISSIndex] = None):
        self.config = config or RetrieverConfig()
        self.index = index
        self.bm25 = None
        self.tfidf_vectorizer = None
        self.corpus = []
        self.corpus_embeddings = None
        
        # Statistics
        self.stats = {
            "total_retrievals": 0,
            "avg_retrieval_time": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Query cache
        self.query_cache = {}
        
        # Initialize retrieval components
        if self.index and self.index.metadata:
            self._initialize_sparse_retrievers()
    
    def _initialize_sparse_retrievers(self):
        """Initialize BM25 and TF-IDF for sparse retrieval."""
        if not self.index or not self.index.metadata:
            logger.warning("No metadata available for sparse retriever initialization")
            return
        
        # Prepare corpus
        self.corpus = []
        for meta in self.index.metadata:
            tokens = self._tokenize(meta.text)
            self.corpus.append(tokens)
        
        # Initialize BM25
        self.bm25 = BM25Okapi(self.corpus)
        
        # Initialize TF-IDF for reranking
        corpus_texts = [meta.text for meta in self.index.metadata]
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus_texts)
        
        logger.info(f"Initialized sparse retrievers with {len(self.corpus)} documents")
    
    def _tokenize(self, text: str) -> List[str]:
        """Advanced tokenization for BM25."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\!\?\-]', ' ', text)
        
        # Split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text)
        
        # Remove stop words (simplified - in production use NLTK or spaCy)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were'}
        tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
        
        return tokens
    
    def retrieve(self,
                query: str,
                top_k: Optional[int] = None,
                filter_metadata: Optional[Dict[str, Any]] = None,
                use_cache: bool = True) -> List[RetrievalResult]:
        """
        Main retrieval method using hybrid search.
        
        Args:
            query: Query text
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            use_cache: Whether to use cached results
            
        Returns:
            List of RetrievalResult objects
        """
        start_time = datetime.now()
        
        if top_k is None:
            top_k = self.config.top_k
        
        # Check cache
        cache_key = f"{query}:{top_k}:{str(filter_metadata)}"
        if use_cache and cache_key in self.query_cache:
            self.stats["cache_hits"] += 1
            return self.query_cache[cache_key]
        
        self.stats["cache_misses"] += 1
        
        if not self.index or self.index.index is None:
            logger.error("Index not initialized")
            return []
        
        # Get candidates from different methods
        candidates = self._get_candidates(query, top_k * 3, filter_metadata)
        
        # Combine and rerank
        results = self._combine_and_rerank(query, candidates, top_k)
        
        # Apply MMR if configured
        if self.config.use_mmr:
            results = self._apply_mmr(results, top_k)
        
        # Cache results
        if use_cache:
            self.query_cache[cache_key] = results
        
        # Update statistics
        elapsed = (datetime.now() - start_time).total_seconds()
        self.stats["total_retrievals"] += 1
        self.stats["avg_retrieval_time"] = (
            (self.stats["avg_retrieval_time"] * (self.stats["total_retrievals"] - 1) + elapsed)
            / self.stats["total_retrievals"]
        )
        
        return results
    
    def _get_candidates(self,
                       query: str,
                       num_candidates: int,
                       filter_metadata: Optional[Dict[str, Any]]) -> Dict[str, RetrievalResult]:
        """
        Get candidate chunks from different retrieval methods.
        
        Args:
            query: Query text
            num_candidates: Number of candidates to retrieve
            filter_metadata: Optional metadata filters
            
        Returns:
            Dictionary of candidates keyed by chunk_id
        """
        candidates = {}
        
        # Vector search
        vector_results = self.index.search(query, num_candidates, filter_metadata)
        for result in vector_results:
            chunk_id = result["chunk_id"]
            candidates[chunk_id] = RetrievalResult(
                chunk_id=chunk_id,
                doc_id=result["doc_id"],
                text=result["text"],
                metadata=result["metadata"],
                vector_score=result["score"]
            )
        
        # BM25 search
        if self.bm25:
            query_tokens = self._tokenize(query)
            bm25_scores = self.bm25.get_scores(query_tokens)
            
            # Get top BM25 results
            top_indices = np.argsort(bm25_scores)[-num_candidates:][::-1]
            
            for idx in top_indices:
                if idx < len(self.index.metadata):
                    meta = self.index.metadata[idx]
                    
                    # Apply filters
                    if filter_metadata:
                        match = all(
                            meta.metadata.get(key) == value
                            for key, value in filter_metadata.items()
                        )
                        if not match:
                            continue
                    
                    chunk_id = meta.chunk_id
                    if chunk_id in candidates:
                        candidates[chunk_id].bm25_score = float(bm25_scores[idx])
                    else:
                        candidates[chunk_id] = RetrievalResult(
                            chunk_id=chunk_id,
                            doc_id=meta.doc_id,
                            text=meta.text,
                            metadata=meta.metadata,
                            bm25_score=float(bm25_scores[idx])
                        )
        
        return candidates
    
    def _combine_and_rerank(self,
                           query: str,
                           candidates: Dict[str, RetrievalResult],
                           top_k: int) -> List[RetrievalResult]:
        """
        Combine scores and rerank candidates.
        
        Args:
            query: Query text
            candidates: Dictionary of candidate results
            top_k: Number of final results
            
        Returns:
            List of reranked results
        """
        if not candidates:
            return []
        
        # Normalize scores
        self._normalize_scores(candidates)
        
        # Combine scores
        for result in candidates.values():
            result.combined_score = (
                self.config.vector_weight * result.vector_score +
                self.config.bm25_weight * result.bm25_score
            )
        
        # Apply reranking if configured
        if self.config.rerank:
            self._rerank_results(query, list(candidates.values()))
        
        # Calculate final scores
        for result in candidates.values():
            if self.config.rerank:
                result.final_score = result.combined_score * 0.7 + result.rerank_score * 0.3
            else:
                result.final_score = result.combined_score
        
        # Sort by final score
        sorted_results = sorted(
            candidates.values(),
            key=lambda x: x.final_score,
            reverse=True
        )
        
        # Apply minimum score threshold
        if self.config.min_score > 0:
            sorted_results = [
                r for r in sorted_results
                if r.final_score >= self.config.min_score
            ]
        
        # Assign ranks
        for i, result in enumerate(sorted_results):
            result.rank = i + 1
        
        return sorted_results[:top_k]
    
    def _normalize_scores(self, candidates: Dict[str, RetrievalResult]):
        """Normalize scores to [0, 1] range."""
        # Normalize vector scores
        vector_scores = [r.vector_score for r in candidates.values() if r.vector_score > 0]
        if vector_scores:
            max_vector = max(vector_scores)
            min_vector = min(vector_scores)
            range_vector = max_vector - min_vector
            
            for result in candidates.values():
                if range_vector > 0:
                    result.vector_score = (result.vector_score - min_vector) / range_vector
                else:
                    result.vector_score = 1.0 if result.vector_score > 0 else 0.0
        
        # Normalize BM25 scores
        bm25_scores = [r.bm25_score for r in candidates.values() if r.bm25_score > 0]
        if bm25_scores:
            max_bm25 = max(bm25_scores)
            min_bm25 = min(bm25_scores)
            range_bm25 = max_bm25 - min_bm25
            
            for result in candidates.values():
                if range_bm25 > 0:
                    result.bm25_score = (result.bm25_score - min_bm25) / range_bm25
                else:
                    result.bm25_score = 1.0 if result.bm25_score > 0 else 0.0
    
    def _rerank_results(self, query: str, results: List[RetrievalResult]):
        """
        Rerank results using advanced methods.
        
        Args:
            query: Query text
            results: List of results to rerank
        """
        if not results:
            return
        
        # TF-IDF similarity reranking
        if self.tfidf_vectorizer and self.tfidf_matrix is not None:
            query_tfidf = self.tfidf_vectorizer.transform([query])
            
            for result in results:
                # Find the index of this chunk in the corpus
                chunk_idx = None
                for i, meta in enumerate(self.index.metadata):
                    if meta.chunk_id == result.chunk_id:
                        chunk_idx = i
                        break
                
                if chunk_idx is not None:
                    # Calculate TF-IDF similarity
                    similarity = cosine_similarity(
                        query_tfidf,
                        self.tfidf_matrix[chunk_idx]
                    )[0, 0]
                    result.rerank_score = float(similarity)
        
        # Additional reranking based on query term overlap
        query_terms = set(self._tokenize(query))
        for result in results:
            text_terms = set(self._tokenize(result.text))
            
            # Calculate Jaccard similarity
            intersection = query_terms & text_terms
            union = query_terms | text_terms
            jaccard = len(intersection) / len(union) if union else 0
            
            # Combine with existing rerank score
            result.rerank_score = (result.rerank_score + jaccard) / 2
    
    def _apply_mmr(self,
                   results: List[RetrievalResult],
                   top_k: int) -> List[RetrievalResult]:
        """
        Apply Maximal Marginal Relevance for diversity.
        
        Args:
            results: List of results
            top_k: Number of results to return
            
        Returns:
            Diversified results
        """
        if not results:
            return []
        
        # Start with the top result
        selected = [results[0]]
        candidates = results[1:]
        
        # Get embeddings for all results
        texts = [r.text for r in results]
        embeddings = self.index.embedding_model.embed_texts(texts, show_progress=False)
        
        while len(selected) < top_k and candidates:
            # Calculate MMR scores
            mmr_scores = []
            
            for i, candidate in enumerate(candidates):
                # Relevance score
                relevance = candidate.final_score
                
                # Calculate maximum similarity to selected documents
                max_similarity = 0
                candidate_idx = results.index(candidate)
                
                for selected_result in selected:
                    selected_idx = results.index(selected_result)
                    similarity = cosine_similarity(
                        embeddings[candidate_idx:candidate_idx+1],
                        embeddings[selected_idx:selected_idx+1]
                    )[0, 0]
                    max_similarity = max(max_similarity, similarity)
                
                # MMR score
                mmr_score = (
                    self.config.mmr_lambda * relevance -
                    (1 - self.config.mmr_lambda) * max_similarity
                )
                mmr_scores.append(mmr_score)
            
            # Select document with highest MMR score
            best_idx = np.argmax(mmr_scores)
            selected.append(candidates[best_idx])
            candidates.pop(best_idx)
        
        return selected
    
    def get_context(self,
                   results: List[RetrievalResult],
                   max_length: int = 3000,
                   include_metadata: bool = True) -> str:
        """
        Format retrieval results as context for generation.
        
        Args:
            results: Retrieved results
            max_length: Maximum context length
            include_metadata: Whether to include source metadata
            
        Returns:
            Formatted context string
        """
        if not results:
            return ""
        
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(results, 1):
            # Format metadata
            metadata_str = ""
            if include_metadata:
                source = result.metadata.get("source", "Unknown")
                page = result.metadata.get("page", "N/A")
                metadata_str = f"[Source {i}: {source}, Page {page}]\n"
            
            # Format chunk
            chunk_text = f"{metadata_str}{result.text}\n"
            
            # Check length
            if current_length + len(chunk_text) > max_length:
                remaining = max_length - current_length
                if remaining > 100:  # Only add if meaningful
                    truncated = chunk_text[:remaining] + "..."
                    context_parts.append(truncated)
                break
            
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        return "\n".join(context_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        cache_size = len(self.query_cache)
        hit_rate = (
            self.stats["cache_hits"] / 
            max(1, self.stats["cache_hits"] + self.stats["cache_misses"])
        )
        
        return {
            **self.stats,
            "cache_size": cache_size,
            "cache_hit_rate": hit_rate,
            "num_documents": len(self.corpus) if self.corpus else 0,
            "retrieval_method": "hybrid",
            "vector_weight": self.config.vector_weight,
            "bm25_weight": self.config.bm25_weight
        }
    
    def clear_cache(self):
        """Clear the query cache."""
        self.query_cache.clear()
        self.stats["cache_hits"] = 0
        self.stats["cache_misses"] = 0
        logger.info("Query cache cleared")