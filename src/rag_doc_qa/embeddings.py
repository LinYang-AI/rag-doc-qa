"""
Embeddings module with support for multiple backends and optimizations.
Includes caching, batching, and async support.
"""

import asyncio
import json
import hashlib
import logging
import os
import pickle
import random
import string
import time
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from datetime import datetime, timedelta

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from rag_doc_qa.config import EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Advanced disk-based cache with TTL and size management."""

    def __init__(
        self, cache_dir: Path, ttl_seconds: int = 86400, max_size_gb: float = 10.0
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(seconds=ttl_seconds)
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        self.memory_cache = {}  # In-memory LRU cache
        self.stats = {"hits": 0, "misses": 0, "errors": 0}

        # Index file for cache management
        self.index_file = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_index()

    def _load_index(self) -> Dict[str, Dict]:
        """Load cache index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, "r") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_index(self):
        """Save cache index to disk."""
        with open(self.index_file, "w") as f:
            json.dump(self.cache_index, f)

    def get_cache_key(self, text: str, model_name: str) -> str:
        """Generate cache key for text and model."""
        content = f"{model_name}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """Retrieve embedding from cache with TTL check."""
        key = self.get_cache_key(text, model_name)

        # Check memory cache first
        if key in self.memory_cache:
            self.stats["hits"] += 1
            return self.memory_cache[key]

        # Check disk cache
        cache_file = self.cache_dir / f"{key}.npy"
        if cache_file.exists():
            try:
                # Check TTL
                if key in self.cache_index:
                    cached_time = datetime.fromisoformat(
                        self.cache_index[key]["timestamp"]
                    )
                    if datetime.now() - cached_time > self.ttl:
                        # Expired
                        cache_file.unlink()
                        del self.cache_index[key]
                        self.stats["misses"] += 1
                        return None

                # Load from disk
                embedding = np.load(cache_file)
                self.memory_cache[key] = embedding
                self.stats["hits"] += 1
                return embedding

            except Exception as e:
                logger.warning(f"Error loading cached embedding: {e}")
                self.stats["errors"] += 1

        self.stats["misses"] += 1
        return None

    def set(self, text: str, model_name: str, embedding: np.ndarray):
        """Store embedding in cache with metadata."""
        key = self.get_cache_key(text, model_name)

        # Store in memory
        self.memory_cache[key] = embedding

        # Check cache size before storing
        if self._get_cache_size() > self.max_size_bytes:
            self._cleanup_old_entries()

        # Store on disk
        cache_file = self.cache_dir / f"{key}.npy"
        try:
            np.save(cache_file, embedding)

            # Update index
            self.cache_index[key] = {
                "timestamp": datetime.now().isoformat(),
                "size": cache_file.stat().st_size,
                "model": model_name,
            }
            self._save_index()

        except Exception as e:
            logger.warning(f"Error caching embedding: {e}")
            self.stats["errors"] += 1

    def _get_cache_size(self) -> int:
        """Get total cache size in bytes."""
        return sum(f.stat().st_size for f in self.cache_dir.glob("*.npy"))

    def _cleanup_old_entries(self, keep_fraction: float = 0.7):
        """Remove old cache entries to free space."""
        # Sort by timestamp
        sorted_entries = sorted(
            self.cache_index.items(), key=lambda x: x[1]["timestamp"]
        )

        # Remove oldest entries
        num_to_remove = int(len(sorted_entries) * (1 - keep_fraction))
        for key, _ in sorted_entries[:num_to_remove]:
            cache_file = self.cache_dir / f"{key}.npy"
            if cache_file.exists():
                cache_file.unlink()
            if key in self.cache_index:
                del self.cache_index[key]
            if key in self.memory_cache:
                del self.memory_cache[key]

        self._save_index()
        logger.info(f"Cleaned up {num_to_remove} cache entries")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        cache_size = self._get_cache_size()
        hit_rate = self.stats["hits"] / max(
            1, self.stats["hits"] + self.stats["misses"]
        )

        return {
            **self.stats,
            "hit_rate": hit_rate,
            "cache_size_mb": cache_size / (1024 * 1024),
            "num_entries": len(self.cache_index),
            "memory_entries": len(self.memory_cache),
        }

    def clear(self):
        """Clear all cache entries."""
        # Clear memory cache
        self.memory_cache.clear()

        # Clear disk cache
        for cache_file in self.cache_dir.glob("*.npy"):
            cache_file.unlink()

        # Clear index
        self.cache_index.clear()
        self._save_index()

        # Reset stats
        self.stats = {"hits": 0, "misses": 0, "errors": 0}
        logger.info("Cache cleared")


class EmbeddingModel:
    """
    Universal embedding model wrapper with multiple backend support.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.cache = EmbeddingCache(
            self.config.cache_dir, ttl_seconds=3600 * 24 * 7  # 7 days TTL
        )
        self.model = None
        self.tokenizer = None
        self.openai_client = None

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Initialize model
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the appropriate embedding model."""
        if self.config.use_openai:
            self._initialize_openai()
        else:
            self._initialize_local_model()

    def _initialize_local_model(self):
        """Initialize local sentence-transformers model."""
        try:
            logger.info(f"Loading embedding model: {self.config.model_name}")

            # Load model with optimization settings
            self.model = SentenceTransformer(
                self.config.model_name, device=self.config.device
            )

            # Set max sequence length
            self.model.max_seq_length = self.config.max_seq_length

            # Enable mixed precision for GPU
            if self.config.device == "cuda":
                self.model = self.model.half()

            # Update dimension from model
            actual_dimension = self.model.get_sentence_embedding_dimension()
            if actual_dimension is not None:
                self.config.set_dimension(actual_dimension)

            logger.info(
                f"Model loaded. Dimension: {self.config.dimension}, Device: {self.config.device}"
            )

        except Exception as e:
            logger.error(f"Error loading model {self.config.model_name}: {e}")

            # Try fallback to CPU
            if self.config.device != "cpu":
                logger.info("Falling back to CPU...")
                self.config.device = "cpu"
                self._initialize_local_model()
            else:
                raise

    def _initialize_openai(self):
        """Initialize OpenAI client for embeddings."""
        try:
            import openai

            if not self.config.openai_api_key:
                self.config.openai_api_key = os.getenv("OPENAI_API_KEY")

            if not self.config.openai_api_key:
                raise ValueError("OpenAI API key not provided")

            openai.api_key = self.config.openai_api_key
            self.openai_client = openai

            logger.info(
                f"OpenAI embeddings initialized with model: {self.config.openai_model}"
            )

        except ImportError:
            logger.error(
                "OpenAI package not installed. Install with: pip install openai"
            )
            raise

    def embed_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        Embed multiple texts with optimizations.

        Args:
            texts: List of texts to embed
            batch_size: Override batch size
            show_progress: Show progress bar
            use_cache: Use caching

        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.array([])

        batch_size = batch_size or self.config.batch_size
        embeddings = []
        texts_to_embed = []
        text_indices = []

        # Check cache first
        if use_cache:
            for i, text in enumerate(texts):
                cached = self.cache.get(text, self.config.model_name)
                if cached is not None:
                    embeddings.append((i, cached))
                else:
                    texts_to_embed.append(text)
                    text_indices.append(i)
        else:
            texts_to_embed = texts
            text_indices = list(range(len(texts)))

        # Embed uncached texts
        new_embeddings = np.array([])  # Initialize as empty array
        if texts_to_embed:
            logger.info(
                f"Embedding {len(texts_to_embed)} texts (cached: {len(embeddings)})"
            )

            if self.config.use_openai:
                new_embeddings = self._embed_with_openai(
                    texts_to_embed, batch_size, show_progress
                )
            else:
                new_embeddings = self._embed_with_local_model(
                    texts_to_embed, batch_size, show_progress
                )

            # Cache new embeddings
            if use_cache:
                for text, embedding, idx in zip(
                    texts_to_embed, new_embeddings, text_indices
                ):
                    self.cache.set(text, self.config.model_name, embedding)
                    embeddings.append((idx, embedding))
            else:
                embeddings = [(i, emb) for i, emb in enumerate(new_embeddings)]

        # Sort by original index and return
        if use_cache:
            embeddings.sort(key=lambda x: x[0])
            return np.array([emb for _, emb in embeddings])
        else:
            return (
                new_embeddings
                if isinstance(new_embeddings, np.ndarray)
                else np.array(new_embeddings)
            )

    def _embed_with_local_model(
        self, texts: List[str], batch_size: int, show_progress: bool
    ) -> np.ndarray:
        """Embed texts using local sentence-transformers model."""

        # Process in batches
        all_embeddings = []

        for i in tqdm(
            range(0, len(texts), batch_size),
            desc="Embedding batches",
            disable=not show_progress,
        ):
            batch = texts[i : i + batch_size]

            # Embed batch
            with torch.no_grad():
                batch_embeddings = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    normalize_embeddings=self.config.normalize_embeddings,
                    show_progress_bar=False,
                )

            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings) if all_embeddings else np.array([])

    def _embed_with_openai(
        self, texts: List[str], batch_size: int, show_progress: bool
    ) -> np.ndarray:
        """Embed texts using OpenAI API with retry logic."""
        import openai
        from openai import OpenAI

        client = OpenAI(api_key=self.config.openai_api_key)
        embeddings = []

        # OpenAI has a limit of ~8000 tokens per batch
        max_batch_size = min(batch_size, 100)
        max_retries = 3

        for i in tqdm(
            range(0, len(texts), max_batch_size),
            desc="OpenAI embeddings",
            disable=not show_progress,
        ):
            batch = texts[i : i + max_batch_size]

            for attempt in range(max_retries):
                try:
                    response = client.embeddings.create(
                        model=self.config.openai_model, input=batch
                    )

                    batch_embeddings = [item.embedding for item in response.data]
                    embeddings.extend(batch_embeddings)
                    break

                except Exception as e:
                    logger.warning(f"OpenAI API attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2**attempt)  # Exponential backoff
                    else:
                        raise

        return np.array(embeddings)

    def embed_query(self, query: str, use_cache: bool = True) -> np.ndarray:
        """
        Embed a single query text.

        Args:
            query: Query text to embed
            use_cache: Whether to use cache

        Returns:
            Numpy array of embedding
        """
        return self.embed_texts([query], use_cache=use_cache, show_progress=False)[0]

    async def embed_texts_async(
        self, texts: List[str], batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Asynchronously embed texts for better performance.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing

        Returns:
            Numpy array of embeddings
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.embed_texts,
            texts,
            batch_size,
            False,  # Don't show progress in async mode
        )

    def embed_documents_batch(
        self,
        documents: List[Any],
        text_field: str = "content",
        batch_size: Optional[int] = None,
    ) -> List[Tuple[Any, np.ndarray]]:
        """
        Embed a batch of documents.

        Args:
            documents: List of document objects
            text_field: Field name containing text to embed
            batch_size: Batch size for processing

        Returns:
            List of (document, embedding) tuples
        """
        texts = [
            getattr(doc, text_field) if hasattr(doc, text_field) else doc[text_field]
            for doc in documents
        ]

        embeddings = self.embed_texts(texts, batch_size)

        return list(zip(documents, embeddings))

    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        embeddings: np.ndarray,
        metric: str = "cosine",
    ) -> np.ndarray:
        """
        Compute similarity between query and multiple embeddings.

        Args:
            query_embedding: Query embedding vector
            embeddings: Matrix of embeddings to compare against
            metric: Similarity metric (cosine, euclidean, dot)

        Returns:
            Array of similarity scores
        """
        if metric == "cosine":
            # Normalize embeddings
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            embeddings_norm = embeddings / np.linalg.norm(
                embeddings, axis=1, keepdims=True
            )
            return np.dot(embeddings_norm, query_norm)

        elif metric == "euclidean":
            return -np.linalg.norm(embeddings - query_embedding, axis=1)

        elif metric == "dot":
            return np.dot(embeddings, query_embedding)

        else:
            raise ValueError(f"Unknown metric: {metric}")

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.config.dimension

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model_name": (
                self.config.model_name
                if not self.config.use_openai
                else self.config.openai_model
            ),
            "dimension": self.config.dimension,
            "device": self.config.device,
            "backend": "openai" if self.config.use_openai else "local",
            "max_seq_length": self.config.max_seq_length,
            "normalize": self.config.normalize_embeddings,
            "cache_stats": self.cache.get_stats(),
        }

    def optimize_for_inference(self):
        """Optimize model for inference (production use)."""
        if self.model and hasattr(self.model, "eval"):
            self.model.eval()

            # Enable torch compile for faster inference (PyTorch 2.0+)
            if self.config.device == "cuda" and hasattr(torch, "compile"):
                try:
                    self.model = torch.compile(self.model)
                    logger.info("Model compiled for faster inference")
                except Exception as e:
                    logger.warning(f"Could not compile model: {e}")

    def warmup(self, sample_texts: Optional[List[str]] = None):
        """
        Warm up the model with sample texts for consistent latency.

        Args:
            sample_texts: Optional sample texts, otherwise uses defaults
        """
        if sample_texts is None:
            sample_texts = [
                "This is a warmup text.",
                "Another sample for warming up the model.",
                "Third warmup text to ensure consistent performance.",
            ]

        logger.info("Warming up embedding model...")
        _ = self.embed_texts(sample_texts, show_progress=False, use_cache=False)
        logger.info("Model warmup complete")

    def benchmark(
        self, num_texts: int = 100, text_length: int = 200
    ) -> Dict[str, float]:
        """
        Benchmark embedding performance.

        Args:
            num_texts: Number of texts to benchmark
            text_length: Approximate length of each text

        Returns:
            Dictionary with benchmark metrics
        """
        import random
        import string

        # Generate random texts
        texts = [
            "".join(random.choices(string.ascii_letters + " ", k=text_length))
            for _ in range(num_texts)
        ]

        # Measure time
        start_time = time.time()
        embeddings = self.embed_texts(texts, show_progress=False, use_cache=False)
        total_time = time.time() - start_time

        return {
            "total_texts": float(num_texts),
            "total_time": total_time,
            "texts_per_second": num_texts / total_time,
            "avg_time_per_text": total_time / num_texts,
            "embedding_dimensions": (
                float(embeddings.shape[1]) if len(embeddings.shape) > 1 else 0.0
            ),
        }

    def clear_cache(self):
        """Clear the embedding cache."""
        self.cache.clear()

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)
