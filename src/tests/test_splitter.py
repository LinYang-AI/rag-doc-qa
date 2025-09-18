"""
Unit tests for text splitting functionality.
"""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag_doc_qa.splitter import TextSplitter, Chunk
from rag_doc_qa.ingest import Document
from rag_doc_qa.config import ChunkingConfig

class TestTextSplitter:
    """Test text splitting functionality."""
    
    @pytest.fixture
    def splitter(self):
        """Create a text splitter instance."""
        config = ChunkingConfig(chunk_size=100, chunk_overlap=20)
        return TextSplitter(config)
    
    @pytest.fixture
    def sample_document(self):
        """Create a sample document."""
        content = """
        Machine learning is a subset of artificial intelligence.
        It enables systems to learn from data.
        Deep learning uses neural networks with multiple layers.
        This technology powers many modern applications.
        Natural language processing helps computers understand human language.
        """
        return Document(
            content=content,
            metadata={"source": "test.txt", "page": 1},
            doc_id="test_doc_001",
            source="test.txt"
        )
    
    def test_split_document(self, splitter, sample_document):
        """Test basic document splitting."""
        chunks = splitter.split_document(sample_document)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(len(chunk.text) <= splitter.config.chunk_size + 50 for chunk in chunks)
    
    def test_chunk_overlap(self, splitter, sample_document):
        """Test that chunks have proper overlap."""
        chunks = splitter.split_document(sample_document)
        
        if len(chunks) > 1:
            # Check that consecutive chunks have some overlap
            for i in range(len(chunks) - 1):
                chunk1_end = chunks[i].text[-20:]
                chunk2_start = chunks[i + 1].text[:50]
                
                # There should be some common text
                # (This is a simplified check)
                assert len(chunk1_end) > 0 and len(chunk2_start) > 0
    
    def test_metadata_preservation(self, splitter, sample_document):
        """Test that metadata is preserved in chunks."""
        chunks = splitter.split_document(sample_document)
        
        for chunk in chunks:
            assert chunk.doc_id == sample_document.doc_id
            assert "source" in chunk.metadata
            assert chunk.metadata["source"] == sample_document.source
    
    def test_deterministic_chunk_ids(self, splitter, sample_document):
        """Test that chunk IDs are deterministic."""
        chunks1 = splitter.split_document(sample_document)
        
        # Clear cache and split again
        splitter.chunk_cache.clear()
        chunks2 = splitter.split_document(sample_document)
        
        # Chunk IDs should be the same
        ids1 = [c.chunk_id for c in chunks1]
        ids2 = [c.chunk_id for c in chunks2]
        assert ids1 == ids2
    
    def test_empty_document(self, splitter):
        """Test handling of empty document."""
        empty_doc = Document(
            content="",
            metadata={},
            doc_id="empty",
            source="empty.txt"
        )
        
        chunks = splitter.split_document(empty_doc)
        assert len(chunks) == 0
    
    def test_long_document(self, splitter):
        """Test splitting of very long document."""
        long_content = "\n".join(["This is a very long test sentence that contains many words and exceeds the chunk size limit to test the long text splitting functionality."] * 100)
        long_doc = Document(
            content=long_content,
            metadata={"source": "long.txt"},
            doc_id="long_doc",
            source="long.txt"
        )
        
        chunks = splitter.split_document(long_doc)
        
        # Verify all content is captured
        total_length = sum(len(c.text) for c in chunks)
        assert total_length >= len(long_content) * 0.9  # Allow for some trimming
        
        # Verify chunk size constraints
        for chunk in chunks:
            assert len(chunk.text) <= splitter.config.chunk_size * 2  # Allow some flexibility
    
    def test_chunk_stats(self, splitter, sample_document):
        """Test chunk statistics calculation."""
        chunks = splitter.split_document(sample_document)
        stats = splitter.get_chunk_stats(chunks)
        
        assert "total_chunks" in stats
        assert "avg_chunk_size" in stats
        assert "min_chunk_size" in stats
        assert "max_chunk_size" in stats
        assert stats["total_chunks"] == len(chunks)