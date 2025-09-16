"""
FAISS vector index module for RAG system.
Handles index creation, persistence, and management.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
import pickle

from rag_doc_qa.config import IndexConfig
from rag_doc_qa.splitter import Chunk
from rag_doc_qa.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


class FAISSIndex:
    """
    FAISS vector index with metadata storage.
    """

    def __init__(
        self,
        config: Optional[IndexConfig] = None,
        embedding_model: Optional[EmbeddingModel] = None,
    ):
        self.config = config or IndexConfig()
        self.embedding_model = embedding_model or EmbeddingModel()
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self.dimension = self.embedding_model.get_dimension()

    def create_index(self, chunks: List[Chunk], rebuild: bool = False):
        """
        Create or update FAISS index from chunks.

        Args:
            chunks: List of Chunk objects to index
            rebuild: Whether to rebuild index from scratch
        """
        if not rebuild and self.index_exists():
            logger.info("Loading existing index...")
            self.load()
            logger.info(f"Existing index has {self.index.ntotal} vectors")

        # Get texts for embedding
        texts = [chunk.text for chunk in chunks]

        # Check for duplicates
        existing_ids = (
            {m["chunk_id"] for m in self.metadata} if self.metadata else set()
        )
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]

        if not new_chunks:
            logger.info("No new chunks to index")
            return

        logger.info(f"Indexing {len(new_chunks)} new chunks...")

        # Embed new texts
        new_texts = [chunk.text for chunk in new_chunks]
        embeddings = self.embedding_model.embed_texts(new_texts)

        # Initialize index if needed
        if self.index is None:
            self._initialize_index()

        # Add to index
        self.index.add(embeddings.astype("float32"))

        # Update metadata
        for chunk in new_chunks:
            self.metadata.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                }
            )

        logger.info(f"Index now contains {self.index.ntotal} vectors")

        # Save index
        self.save()

    def _initialize_index(self):
        """Initialize FAISS index based on configuration."""
        if self.config.similarity_metric == "cosine":
            # Use Inner Product with normalized vectors for cosine similarity
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.config.similarity_metric == "l2":
            self.index = faiss.IndexFlatL2(self.dimension)
        else:
            self.index = faiss.IndexFlatIP(self.dimension)

        # Optionally use IVF for larger datasets
        if self.config.nlist > 0:
            quantizer = self.index
            self.index = faiss.IndexIVFFlat(
                quantizer, self.dimension, self.config.nlist
            )
            # Train with dummy data if empty
            if self.index.ntotal == 0:
                dummy_data = np.random.random(
                    (self.config.nlist * 10, self.dimension)
                ).astype("float32")
                self.index.train(dummy_data)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the index for similar chunks.

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            List of search results with scores and metadata
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []

        # Embed query
        query_embedding = self.embedding_model.embed_query(query)

        # Normalize for cosine similarity if needed
        if self.config.similarity_metric == "cosine":
            query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Search
        query_embedding = query_embedding.reshape(1, -1).astype("float32")
        scores, indices = self.index.search(
            query_embedding, min(top_k, self.index.ntotal)
        )

        # Compile results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result["score"] = float(score)
                results.append(result)

        return results

    def save(self):
        """Save index and metadata to disk."""
        if self.index is None:
            logger.warning("No index to save")
            return

        # Create directories
        self.config.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(self.config.index_path))

        # Save metadata
        with open(self.config.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        logger.info(f"Index saved to {self.config.index_path}")
        logger.info(f"Metadata saved to {self.config.metadata_path}")

    def load(self):
        """Load index and metadata from disk."""
        if not self.index_exists():
            raise FileNotFoundError(f"Index not found at {self.config.index_path}")

        # Load FAISS index
        self.index = faiss.read_index(str(self.config.index_path))

        # Load metadata
        with open(self.config.metadata_path, "r") as f:
            self.metadata = json.load(f)

        logger.info(f"Loaded index with {self.index.ntotal} vectors")

    def index_exists(self) -> bool:
        """Check if index files exist."""
        return self.config.index_path.exists() and self.config.metadata_path.exists()

    def delete_chunk(self, chunk_id: str):
        """
        Delete a chunk from the index.
        Note: FAISS doesn't support deletion, so this rebuilds the index.
        """
        # Remove from metadata
        self.metadata = [m for m in self.metadata if m["chunk_id"] != chunk_id]

        # Rebuild index
        if self.metadata:
            texts = [m["text"] for m in self.metadata]
            embeddings = self.embedding_model.embed_texts(texts)

            self._initialize_index()
            self.index.add(embeddings.astype("float32"))

            self.save()
        else:
            # Clear index if no metadata left
            self.clear()

    def clear(self):
        """Clear the index and metadata."""
        self.index = None
        self.metadata = []

        # Delete files if they exist
        if self.config.index_path.exists():
            self.config.index_path.unlink()
        if self.config.metadata_path.exists():
            self.config.metadata_path.unlink()

        logger.info("Index cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        if self.index is None:
            return {"status": "not_initialized"}

        index_size = 0
        metadata_size = 0

        if self.config.index_path.exists():
            index_size = self.config.index_path.stat().st_size
        if self.config.metadata_path.exists():
            metadata_size = self.config.metadata_path.stat().st_size

        return {
            "num_vectors": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "index_size_bytes": index_size,
            "metadata_size_bytes": metadata_size,
            "total_size_mb": (index_size + metadata_size) / (1024 * 1024),
            "num_documents": len(set(m["doc_id"] for m in self.metadata)),
            "similarity_metric": self.config.similarity_metric,
        }
