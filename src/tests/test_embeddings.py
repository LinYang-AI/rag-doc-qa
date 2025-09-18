"""
Unit tests for embedding functionality.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag_doc_qa.embeddings import EmbeddingModel, EmbeddingCache
from rag_doc_qa.config import EmbeddingConfig

class TestEmbeddingCache:
    """Test embedding cache functionality."""
    
    @pytest.fixture
    def cache(self):
        """Create a temporary cache instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield EmbeddingCache(Path(tmpdir))
    
    def test_cache_key_generation(self, cache):
        """Test cache key generation."""
        key1 = cache.get_cache_key("test text", "model1")
        key2 = cache.get_cache_key("test text", "model1")
        key3 = cache.get_cache_key("different text", "model1")
        
        assert key1 == key2  # Same text and model
        assert key1 != key3  # Different text
    
    def test_cache_storage_retrieval(self, cache):
        """Test storing and retrieving from cache."""
        text = "test text"
        model = "test_model"
        embedding = np.random.randn(384)
        
        # Store embedding
        cache.set(text, model, embedding)
        
        # Retrieve embedding
        retrieved = cache.get(text, model)
        
        assert retrieved is not None
        assert np.array_equal(embedding, retrieved)
    
    def test_cache_miss(self, cache):
        """Test cache miss returns None."""
        result = cache.get("non_existent", "model")
        assert result is None

class TestEmbeddingModel:
    """Test embedding model functionality."""
    
    @pytest.fixture
    def embedding_model(self):
        """Create embedding model instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EmbeddingConfig(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=Path(tmpdir) / "cache",
                device="cpu",
                use_openai=False
            )
            return EmbeddingModel(config)
    
    def test_embed_single_text(self, embedding_model):
        """Test embedding a single text."""
        text = "This is a test sentence."
        embedding = embedding_model.embed_query(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (embedding_model.config.dimension,)
        assert not np.all(embedding == 0)  # Should not be all zeros
    
    def test_embed_multiple_texts(self, embedding_model):
        """Test embedding multiple texts."""
        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence."
        ]
        embeddings = embedding_model.embed_texts(texts, show_progress=False)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, embedding_model.config.dimension)
        
        # Embeddings should be different
        assert not np.allclose(embeddings[0], embeddings[1])
        assert not np.allclose(embeddings[1], embeddings[2])
    
    def test_embedding_caching(self, embedding_model):
        """Test that embeddings are cached."""
        text = "Cache test sentence."
        
        # First embedding (not cached)
        embedding1 = embedding_model.embed_query(text)
        
        # Second embedding (should be from cache)
        embedding2 = embedding_model.embed_query(text)
        
        # Should be identical
        assert np.array_equal(embedding1, embedding2)
    
    def test_batch_processing(self, embedding_model):
        """Test batch processing of embeddings."""
        # Create more texts than batch size
        texts = [f"Test sentence {i}" for i in range(50)]
        
        embeddings = embedding_model.embed_texts(texts, show_progress=False)
        
        assert embeddings.shape == (50, embedding_model.config.dimension)
        
        # Check that all embeddings are unique
        unique_embeddings = np.unique(embeddings, axis=0)
        assert len(unique_embeddings) == 50
    
    def test_empty_text_handling(self, embedding_model):
        """Test handling of empty text."""
        texts = ["", "Non-empty text"]
        embeddings = embedding_model.embed_texts(texts, show_progress=False)
        
        assert embeddings.shape == (2, embedding_model.config.dimension)
        
        # Empty text should still produce an embedding
        assert not np.all(embeddings[0] == 0)
    
    def test_dimension_property(self, embedding_model):
        """Test dimension property."""
        dimension = embedding_model.get_dimension()
        
        assert isinstance(dimension, int)
        assert dimension > 0
        assert dimension == embedding_model.config.dimension
    
    def test_clear_cache(self, embedding_model):
        """Test cache clearing."""
        text = "Clear cache test"
        
        # Create cached embedding
        _ = embedding_model.embed_query(text)
        
        # Clear cache
        embedding_model.clear_cache()
        
        # Cache should be empty
        cached = embedding_model.cache.get(text, embedding_model.config.model_name)
        assert cached is None