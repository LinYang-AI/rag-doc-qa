"""
Embeddings module for RAG system.
Supports both local sentence-transformers and OpenAI embeddings.
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Union, Dict
import numpy as np
import pickle
from functools import lru_cache
import time

from sentence_transformers import SentenceTransformer
import torch

from rag_doc_qa.config import EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Simple disk-based cache for embeddings."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}  # In-memory LRU cache

    def get_cache_key(self, text: str, model_name: str) -> str:
        """Generate cache key for text and model."""
        content = f"{model_name}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """Retrieve embedding from cache."""
        key = self.get_cache_key(text, model_name)

        # Check memory cache first
        if key in self.memory_cache:
            return self.memory_cache[key]

        # Check disk cache
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    embedding = pickle.load(f)
                    self.memory_cache[key] = embedding
                    return embedding
            except Exception as e:
                logger.warning(f"Error loading cached embedding: {e}")

        return None

    def set(self, text: str, model_name: str, embedding: np.ndarray):
        """Store embedding in cache."""
        key = self.get_cache_key(text, model_name)

        # Store in memory
        self.memory_cache[key] = embedding

        # Store on disk
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Error caching embedding: {e}")


class EmbeddingModel:
    """
    Wrapper for embedding models with caching and batch processing.
    Supports both sentence-transformers and OpenAI embeddings.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.cache = EmbeddingCache(self.config.cache_dir)
        self.model = None
        self.openai_client = None

        # Initialize model based on configuration
        if not self.config.use_openai:
            self._initialize_local_model()
        else:
            self._initialize_openai()

    def _initialize_local_model(self):
        """Initialize local sentence-transformers model."""
        try:
            logger.info(f"Loading embedding model: {self.config.model_name}")
            self.model = SentenceTransformer(
                self.config.model_name, device=self.config.device
            )

            # Update dimension based on loaded model
            self.config.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Embedding dimension: {self.config.dimension}")

        except Exception as e:
            logger.error(f"Error loading model {self.config.model_name}: {e}")
            logger.info("Falling back to CPU...")
            self.config.device = "cpu"
            self.model = SentenceTransformer(self.config.model_name, device="cpu")

    def _initialize_openai(self):
        """Initialize OpenAI client for embeddings."""
        try:
            import openai
            from openai import OpenAI

            if not self.config.embedding.openai_api_key:
                raise ValueError("OpenAI API key not provided in config")

            self.openai_client = OpenAI(api_key=self.config.embedding.openai_api_key)
            self.config.dimension = 1536  # OpenAI ada-002 dimension
            logger.info("OpenAI embeddings initialized")

        except ImportError:
            logger.error(
                "OpenAI package not installed. Install with: pip install openai"
            )
            raise
        except Exception as e:
            logger.error(f"Error initializing OpenAI: {e}")
            raise

    def embed_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Embed multiple texts with caching and batch processing.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar

        Returns:
            Numpy array of embeddings (n_texts, embedding_dim)
        """
        embeddings = []
        texts_to_embed = []
        text_indices = []

        # Check cache first
        for i, text in enumerate(texts):
            cached = self.cache.get(text, self.config.model_name)
            if cached is not None:
                embeddings.append((i, cached))
            else:
                texts_to_embed.append(text)
                text_indices.append(i)

        # Embed uncached texts
        if texts_to_embed:
            if self.config.use_openai:
                new_embeddings = self._embed_with_openai(texts_to_embed)
            else:
                new_embeddings = self._embed_with_sentence_transformers(
                    texts_to_embed, show_progress
                )

            # Cache and collect new embeddings
            for text, embedding, idx in zip(
                texts_to_embed, new_embeddings, text_indices
            ):
                self.cache.set(text, self.config.model_name, embedding)
                embeddings.append((idx, embedding))

        # Sort by original index and return
        embeddings.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in embeddings])

    def _embed_with_sentence_transformers(
        self, texts: List[str], show_progress: bool
    ) -> np.ndarray:
        """Embed texts using sentence-transformers."""
        logger.info(f"Embedding {len(texts)} texts with {self.config.model_name}")

        # Batch processing
        embeddings = []
        batch_size = self.config.batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            with torch.no_grad():
                batch_embeddings = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=show_progress and len(texts) > 10,
                )

            embeddings.append(batch_embeddings)

        return np.vstack(embeddings) if embeddings else np.array([])

    def _embed_with_openai(self, texts: List[str]) -> np.ndarray:
        """Embed texts using OpenAI API with retry logic."""
        embeddings = []
        batch_size = 100  # OpenAI batch limit
        max_retries = 3

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            for attempt in range(max_retries):
                try:
                    response = self.openai_client.embeddings.create(
                        model="text-embedding-ada-002", input=batch
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

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query text.

        Args:
            query: Query text to embed

        Returns:
            Numpy array of embedding
        """
        return self.embed_texts([query], show_progress=False)[0]

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.config.dimension

    def clear_cache(self):
        """Clear the embedding cache."""
        self.cache.memory_cache.clear()

        # Clear disk cache
        for cache_file in self.cache.cache_dir.glob("*.pkl"):
            cache_file.unlink()

        logger.info("Embedding cache cleared")
