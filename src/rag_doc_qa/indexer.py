"""
Advanced FAISS vector index module with multiple index types and optimizations.
"""

import json
import logging
import hashlib  # Added missing import
import os  # Added missing import
import pickle
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, asdict, field

import numpy as np
import faiss
from tqdm import tqdm

from rag_doc_qa.config import IndexConfig
from rag_doc_qa.splitter import Chunk
from rag_doc_qa.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)

@dataclass
class IndexMetadata:
    """Metadata for indexed chunks."""
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any]
    start_char: int
    end_char: int
    timestamp: str
    vector_id: int  # Position in FAISS index

    @classmethod
    def from_chunk(cls, chunk: Chunk, vector_id: int) -> "IndexMetadata":
        """Create IndexMetadata from a Chunk object."""
        return cls(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            text=chunk.text,
            metadata=chunk.metadata,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            timestamp=chunk.timestamp.isoformat() if hasattr(chunk, 'timestamp') else datetime.now().isoformat(),
            vector_id=vector_id
        )

class FAISSIndex:
    """
    Advanced FAISS vector index with multiple backend support and optimizations.
    """
    
    def __init__(self, 
                 config: Optional[IndexConfig] = None,
                 embedding_model: Optional[EmbeddingModel] = None):
        self.config = config or IndexConfig()
        self.embedding_model = embedding_model or EmbeddingModel()
        self.index = None
        self.index_map = None  # Separate ID map for tracking
        self.metadata: List[IndexMetadata] = []
        self.dimension = self.embedding_model.get_dimension()
        self.id_to_metadata: Dict[str, IndexMetadata] = {}
        self.doc_id_to_chunks: Dict[str, List[str]] = {}
        
        # Vector ID counter (for maintaining unique IDs)
        self.next_vector_id = 0
        
        # Statistics
        self.stats = {
            "total_indexed": 0,
            "total_searches": 0,
            "avg_search_time": 0,
            "index_build_time": 0
        }
    
    def create_index(self,
                    chunks: List[Chunk],
                    rebuild: bool = False,
                    show_progress: bool = True) -> None:
        """
        Create or update FAISS index from chunks.
        
        Args:
            chunks: List of Chunk objects to index
            rebuild: Whether to rebuild index from scratch
            show_progress: Show progress bar
        """
        start_time = datetime.now()
        
        # Load existing index if not rebuilding
        if not rebuild and self.index_exists():
            logger.info("Loading existing index...")
            self.load()
            logger.info(f"Existing index has {self.index.ntotal if self.index else 0} vectors")
        
        # Filter out already indexed chunks
        existing_ids = {m.chunk_id for m in self.metadata}
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
        
        if not new_chunks:
            logger.info("No new chunks to index")
            return
        
        logger.info(f"Indexing {len(new_chunks)} new chunks...")
        
        # Embed new chunks
        texts = [chunk.text for chunk in new_chunks]
        embeddings = self.embedding_model.embed_texts(
            texts,
            show_progress=show_progress
        )
        
        # Initialize index if needed
        if self.index is None:
            self._initialize_index()
        
        # Add to index
        self._add_vectors(embeddings, new_chunks)
        
        # Update statistics
        elapsed = (datetime.now() - start_time).total_seconds()
        self.stats["index_build_time"] = elapsed
        self.stats["total_indexed"] = self.index.ntotal if self.index else 0
        
        logger.info(f"Index now contains {self.index.ntotal if self.index else 0} vectors")
        logger.info(f"Indexing took {elapsed:.2f} seconds")
        
        # Save index
        self.save()
    
    def _initialize_index(self) -> None:
        """Initialize FAISS index based on configuration."""
        logger.info(f"Initializing {self.config.index_type} index with {self.config.similarity_metric} metric")
        
        # Create base index based on similarity metric
        if self.config.similarity_metric == "cosine":
            # Use Inner Product with normalized vectors for cosine similarity
            base_index = faiss.IndexFlatIP(self.dimension)
        elif self.config.similarity_metric == "l2":
            base_index = faiss.IndexFlatL2(self.dimension)
        elif self.config.similarity_metric == "inner_product":
            base_index = faiss.IndexFlatIP(self.dimension)
        else:
            raise ValueError(f"Unknown similarity metric: {self.config.similarity_metric}")
        
        # Apply index type
        if self.config.index_type == "flat":
            # Simple flat index
            self.index = base_index
            
        elif self.config.index_type == "ivf":
            # IVF index for larger datasets
            # Calculate reasonable nlist based on expected data size
            nlist = min(self.config.nlist, 100)  # Start with reasonable default
            
            if self.config.similarity_metric == "cosine":
                quantizer = faiss.IndexFlatIP(self.dimension)
            else:
                quantizer = faiss.IndexFlatL2(self.dimension)
            
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            
            # IVF indices need training - we'll train when we have data
            self.index.is_trained = False
            
        elif self.config.index_type == "hnsw":
            # HNSW for fast approximate search
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            if hasattr(self.index, 'hnsw'):
                self.index.hnsw.efConstruction = 40
            
        else:
            raise ValueError(f"Unknown index type: {self.config.index_type}")
        
        # Create ID map for tracking (separate from main index)
        self.index_map = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
        
        # Move to GPU if configured and available
        if self.config.use_gpu:
            self._move_to_gpu()
    
    def _move_to_gpu(self):
        """Move index to GPU if available."""
        # Check if faiss-gpu is installed
        if not hasattr(faiss, 'StandardGpuResources'):
            logger.warning("faiss-gpu not installed. Install with: pip install faiss-gpu")
            self.config.use_gpu = False
            return
        
        try:
            if faiss.get_num_gpus() > 0:
                logger.info("Moving index to GPU...")
                res = faiss.StandardGpuResources()
                
                # Don't wrap GPU index with IDMap, handle IDs separately
                if self.config.index_type == "flat":
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                else:
                    # For complex indices, keep on CPU for now
                    logger.warning(f"GPU support for {self.config.index_type} index is limited, keeping on CPU")
                    self.config.use_gpu = False
            else:
                logger.warning("No GPUs available, keeping index on CPU")
                self.config.use_gpu = False
        except Exception as e:
            logger.warning(f"Could not move index to GPU: {e}")
            self.config.use_gpu = False
    
    def _add_vectors(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        """Add vectors to index with metadata."""
        # Normalize embeddings for cosine similarity
        if self.config.similarity_metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            # Avoid division by zero
            norms[norms == 0] = 1
            embeddings = embeddings / norms
        
        # Convert to float32 for FAISS
        embeddings = embeddings.astype('float32')
        
        # Train IVF index if needed
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            logger.info("Training IVF index...")
            if len(embeddings) >= self.config.nlist:
                # Use actual data for training
                self.index.train(embeddings)
            else:
                # Need more data for training, use what we have repeated
                training_data = np.vstack([embeddings] * (self.config.nlist // len(embeddings) + 1))
                self.index.train(training_data[:self.config.nlist * 10])
            self.index.is_trained = True
        
        # Add to main index
        self.index.add(embeddings)
        
        # Store metadata with proper vector IDs
        start_id = self.next_vector_id
        for i, chunk in enumerate(chunks):
            metadata = IndexMetadata.from_chunk(chunk, start_id + i)
            
            self.metadata.append(metadata)
            self.id_to_metadata[chunk.chunk_id] = metadata
            
            # Track document to chunks mapping
            if chunk.doc_id not in self.doc_id_to_chunks:
                self.doc_id_to_chunks[chunk.doc_id] = []
            self.doc_id_to_chunks[chunk.doc_id].append(chunk.chunk_id)
        
        # Update vector ID counter
        self.next_vector_id = start_id + len(chunks)
    
    def search(self,
              query: Union[str, np.ndarray],
              top_k: int = 5,
              filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search the index for similar chunks.
        
        Args:
            query: Query text or embedding
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of search results with scores and metadata
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []
        
        start_time = datetime.now()
        
        # Get query embedding
        if isinstance(query, str):
            query_embedding = self.embedding_model.embed_query(query)
        else:
            query_embedding = query
        
        # Normalize for cosine similarity
        if self.config.similarity_metric == "cosine":
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm
        
        # Prepare query
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Set search parameters for IVF
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = min(self.config.nprobe, self.index.nlist)
        
        # Search (get more results if filtering)
        k = min(top_k * 3 if filter_metadata else top_k, self.index.ntotal)
        
        try:
            scores, indices = self.index.search(query_embedding, k)
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
        
        # Compile results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            # FAISS returns -1 for empty slots
            if idx < 0 or idx >= len(self.metadata):
                continue
            
            metadata_entry = self.metadata[idx]
            
            # Apply metadata filters
            if filter_metadata:
                match = all(
                    metadata_entry.metadata.get(key) == value
                    for key, value in filter_metadata.items()
                )
                if not match:
                    continue
            
            result = {
                "chunk_id": metadata_entry.chunk_id,
                "doc_id": metadata_entry.doc_id,
                "text": metadata_entry.text,
                "metadata": metadata_entry.metadata,
                "score": float(score),
                "start_char": metadata_entry.start_char,
                "end_char": metadata_entry.end_char
            }
            results.append(result)
            
            if len(results) >= top_k:
                break
        
        # Update statistics
        elapsed = (datetime.now() - start_time).total_seconds()
        self.stats["total_searches"] += 1
        self.stats["avg_search_time"] = (
            (self.stats["avg_search_time"] * (self.stats["total_searches"] - 1) + elapsed)
            / self.stats["total_searches"]
        )
        
        return results
    
    def save(self) -> None:
        """Save index and metadata to disk."""
        if self.index is None:
            logger.warning("No index to save")
            return
        
        # Create directories
        self.config.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Handle GPU index
        index_to_save = self.index
        if self.config.use_gpu and hasattr(faiss, 'index_gpu_to_cpu'):
            try:
                index_to_save = faiss.index_gpu_to_cpu(self.index)
            except Exception as e:
                logger.warning(f"Could not move index from GPU: {e}")
                index_to_save = self.index
        
        # Save FAISS index
        faiss.write_index(index_to_save, str(self.config.index_path))
        
        # Save metadata - ensure it's always a dictionary structure
        metadata_dict = {
            "metadata": [asdict(m) for m in self.metadata],
            "config": {
                "dimension": self.dimension,
                "similarity_metric": self.config.similarity_metric,
                "index_type": self.config.index_type,
                "next_vector_id": self.next_vector_id
            },
            "stats": self.stats,
            "doc_mapping": self.doc_id_to_chunks,
            "version": "1.0"  # Add version for future compatibility
        }
        
        with open(self.config.metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2, default=str)
        
        logger.info(f"Index saved to {self.config.index_path}")
        logger.info(f"Metadata saved to {self.config.metadata_path}")
    
    def load(self) -> None:
        """Load index and metadata from disk."""
        if not self.index_exists():
            raise FileNotFoundError(f"Index not found at {self.config.index_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(str(self.config.index_path))
        
        # Load metadata with proper error handling
        with open(self.config.metadata_path, 'r') as f:
            data = json.load(f)
        
        # Handle different metadata formats for backward compatibility
        if isinstance(data, dict):
            # Correct format
            metadata_dict = data
        elif isinstance(data, list):
            # Old format or corrupted - try to recover
            logger.warning("Detected old metadata format, attempting recovery...")
            # Assume it's a list of metadata entries
            metadata_dict = {
                "metadata": data if all(isinstance(item, dict) for item in data) else [],
                "config": {},
                "stats": {},
                "doc_mapping": {}
            }
        else:
            raise ValueError(f"Invalid metadata format in {self.config.metadata_path}")
        
        # Load metadata entries
        try:
            self.metadata = [
                IndexMetadata(**m) for m in metadata_dict.get("metadata", [])
            ]
        except Exception as e:
            logger.error(f"Error loading metadata entries: {e}")
            # Try to recover with minimal metadata
            self.metadata = []
            for i, m in enumerate(metadata_dict.get("metadata", [])):
                try:
                    # Ensure required fields exist
                    if not all(k in m for k in ["chunk_id", "doc_id", "text"]):
                        logger.warning(f"Skipping invalid metadata entry {i}")
                        continue
                    
                    # Add missing fields with defaults
                    m.setdefault("metadata", {})
                    m.setdefault("start_char", 0)
                    m.setdefault("end_char", len(m.get("text", "")))
                    m.setdefault("timestamp", datetime.now().isoformat())
                    m.setdefault("vector_id", i)
                    
                    self.metadata.append(IndexMetadata(**m))
                except Exception as e2:
                    logger.warning(f"Could not load metadata entry {i}: {e2}")
        
        # Load other components
        self.stats = metadata_dict.get("stats", {})
        self.doc_id_to_chunks = metadata_dict.get("doc_mapping", {})
        
        # Load or infer next_vector_id
        config = metadata_dict.get("config", {})
        self.next_vector_id = config.get("next_vector_id", len(self.metadata))
        
        # Rebuild ID mapping
        self.id_to_metadata = {m.chunk_id: m for m in self.metadata}
        
        # Verify index consistency
        if self.index.ntotal != len(self.metadata):
            logger.warning(
                f"Index vector count ({self.index.ntotal}) doesn't match "
                f"metadata count ({len(self.metadata)}). Index may be corrupted."
            )
        
        # Move to GPU if configured
        if self.config.use_gpu:
            self._move_to_gpu()
        
        logger.info(f"Loaded index with {self.index.ntotal} vectors and {len(self.metadata)} metadata entries")
    
    def index_exists(self) -> bool:
        """Check if index files exist."""
        return (
            self.config.index_path.exists() and
            self.config.metadata_path.exists()
        )
    
    def clear(self) -> None:
        """Clear the index and metadata."""
        self.index = None
        self.index_map = None
        self.metadata = []
        self.id_to_metadata = {}
        self.doc_id_to_chunks = {}
        self.next_vector_id = 0
        
        # Delete files if they exist
        if self.config.index_path.exists():
            self.config.index_path.unlink()
        if self.config.metadata_path.exists():
            self.config.metadata_path.unlink()
        
        # Reset statistics
        self.stats = {
            "total_indexed": 0,
            "total_searches": 0,
            "avg_search_time": 0,
            "index_build_time": 0
        }
        
        logger.info("Index cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive index statistics."""
        index_size = 0
        metadata_size = 0
        
        if self.config.index_path.exists():
            index_size = self.config.index_path.stat().st_size
        if self.config.metadata_path.exists():
            metadata_size = self.config.metadata_path.stat().st_size
        
        return {
            **self.stats,
            "num_vectors": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "index_size_bytes": index_size,
            "metadata_size_bytes": metadata_size,
            "total_size_mb": (index_size + metadata_size) / (1024 * 1024),
            "num_documents": len(self.doc_id_to_chunks),
            "num_chunks": len(self.metadata),
            "similarity_metric": self.config.similarity_metric,
            "index_type": self.config.index_type,
            "use_gpu": self.config.use_gpu,
            "next_vector_id": self.next_vector_id
        }