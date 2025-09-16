"""
Unit tests for FAISS indexing functionality.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import tempfile
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag_doc_qa.indexer import FAISSIndex
from rag_doc_qa.splitter import Chunk
from rag_doc_qa.config import IndexConfig
from rag_doc_qa.embeddings import EmbeddingModel

class TestFAISSIndex:
    """Test FAISS index functionality."""
    
    @pytest.fixture
    def index_config(self):
        """Create index configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = IndexConfig(
                index_path=Path(tmpdir) / "test.index",
                metadata_path=Path(tmpdir) / "test_metadata.json",
                similarity_metric="cosine"
            )
            yield config
    
    @pytest.fixture
    def faiss_index(self, index_config):
        """Create FAISS index instance."""
        # Mock embedding model for testing
        class MockEmbeddingModel:
            def get_dimension(self):
                return 384
            
            def embed_texts(self, texts, **kwargs):
                # Return random embeddings for testing
                return np.random.randn(len(texts), 384).astype('float32')
            
            def embed_query(self, query):
                return np.random.randn(384).astype('float32')
        
        return FAISSIndex(index_config, MockEmbeddingModel())
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        chunks = []
        for i in range(5):
            chunk = Chunk(
                text=f"This is test chunk number {i}.",
                chunk_id=f"chunk_{i}",
                doc_id=f"doc_{i % 2}",  # Two different documents
                metadata={"page": i, "source": f"doc_{i % 2}.txt"},
                start_char=i * 100,
                end_char=(i + 1) * 100
            )
            chunks.append(chunk)
        return chunks
    
    def test_create_index(self, faiss_index, sample_chunks):
        """Test index creation."""
        faiss_index.create_index(sample_chunks, rebuild=True)
        
        assert faiss_index.index is not None
        assert faiss_index.index.ntotal == len(sample_chunks)
        assert len(faiss_index.metadata) == len(sample_chunks)
    
    def test_search(self, faiss_index, sample_chunks):
        """Test searching the index."""
        faiss_index.create_index(sample_chunks, rebuild=True)
        
        results = faiss_index.search("test query", top_k=3)
        
        assert len(results) <= 3
        assert all("score" in result for result in results)
        assert all("text" in result for result in results)
        assert all("metadata" in result for result in results)
    
    def test_save_load_index(self, faiss_index, sample_chunks):
        """Test saving and loading index."""
        faiss_index.create_index(sample_chunks, rebuild=True)
        
        # Save index
        faiss_index.save()
        
        assert faiss_index.config.index_path.exists()
        assert faiss_index.config.metadata_path.exists()
        
        # Create new index and load
        new_index = FAISSIndex(faiss_index.config, faiss_index.embedding_model)
        new_index.load()
        
        assert new_index.index.ntotal == len(sample_chunks)
        assert len(new_index.metadata) == len(sample_chunks)
    
    def test_duplicate_detection(self, faiss_index, sample_chunks):
        """Test that duplicates are not added."""
        faiss_index.create_index(sample_chunks, rebuild=True)
        initial_count = faiss_index.index.ntotal
        
        # Try to add the same chunks again
        faiss_index.create_index(sample_chunks, rebuild=False)
        
        # Count should remain the same
        assert faiss_index.index.ntotal == initial_count
    
    def test_empty_index_search(self, faiss_index):
        """Test searching empty index."""
        results = faiss_index.search("test query", top_k=5)
        assert results == []
    
    def test_index_stats(self, faiss_index, sample_chunks):
        """Test index statistics."""
        faiss_index.create_index(sample_chunks, rebuild=True)
        faiss_index.save()
        
        stats = faiss_index.get_stats()
        
        assert stats["num_vectors"] == len(sample_chunks)
        assert stats["dimension"] == 384
        assert stats["num_documents"] == 2  # We have 2 unique doc_ids
        assert "index_size_bytes" in stats
        assert "metadata_size_bytes" in stats
    
    def test_clear_index(self, faiss_index, sample_chunks):
        """Test clearing the index."""
        faiss_index.create_index(sample_chunks, rebuild=True)
        faiss_index.save()
        
        faiss_index.clear()
        
        assert faiss_index.index is None
        assert len(faiss_index.metadata) == 0
        assert not faiss_index.config.index_path.exists()
        assert not faiss_index.config.metadata_path.exists()
    
    def test_index_exists(self, faiss_index, sample_chunks):
        """Test checking index existence."""
        assert not faiss_index.index_exists()
        
        faiss_index.create_index(sample_chunks, rebuild=True)
        faiss_index.save()
        
        assert faiss_index.index_exists()
    
    def test_metadata_preservation(self, faiss_index, sample_chunks):
        """Test that metadata is properly preserved."""
        faiss_index.create_index(sample_chunks, rebuild=True)
        
        for i, chunk in enumerate(sample_chunks):
            # Find corresponding metadata
            meta = next((m for m in faiss_index.metadata if m["chunk_id"] == chunk.chunk_id), None)
            assert meta is not None
            assert meta["text"] == chunk.text
            assert meta["doc_id"] == chunk.doc_id
            assert meta["metadata"]["page"] == chunk.metadata["page"]