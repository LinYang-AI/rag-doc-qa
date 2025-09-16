"""
RAG Document QA System

A production-ready Retrieval-Augmented Generation system for document question answering.
"""

__version__ = "1.0.0"

from .ingest import DocumentIngestor, Document
from .splitter import TextSplitter, Chunk
from .embeddings import EmbeddingModel
from .indexer import FAISSIndex
from .retriever import HybridRetriever
from .generator import LLMGenerator, GenerationResult
from .evaluate import RAGEvaluator, EvaluationResult
from .web_app import RAGPipeline, create_app
from .config import (
    SystemConfig,
    EmbeddingConfig,
    ChunkingConfig,
    IndexConfig,
    RetrieverConfig,
    GeneratorConfig,
)

__all__ = [
    # Core classes
    "DocumentIngestor",
    "Document",
    "TextSplitter",
    "Chunk",
    "EmbeddingModel",
    "FAISSIndex",
    "HybridRetriever",
    "LLMGenerator",
    "GenerationResult",
    "RAGEvaluator",
    "EvaluationResult",
    "RAGPipeline",
    "create_app",
    # Config classes
    "SystemConfig",
    "EmbeddingConfig",
    "ChunkingConfig",
    "IndexConfig",
    "RetrieverConfig",
    "GeneratorConfig",
    # Version
    "__version__",
]
