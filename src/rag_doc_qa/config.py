"""
Configuration module for RAG Document QA System.
Handles environment variables, model settings, and system parameters.
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
INDEX_DIR = DATA_DIR / "index"

# Create directories if they don't exist
for dir_path in [DATA_DIR, CACHE_DIR, INDEX_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""

    model_name: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    use_openai: bool = os.getenv("USE_OPENAI_EMBEDDING", "false").lower() == "true"
    cache_dir: Path = CACHE_DIR / "embeddings"
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dimension: int = 384  # Default for all-MiniLM-L6-v2


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    min_chunk_size: int = 100
    metadata_fields: list = None

    def __post_init__(self):
        if self.metadata_fields is None:
            self.metadata_fields = ["source", "page", "chunk_id"]


@dataclass
class IndexConfig:
    """Configuration for FAISS index."""

    index_path: Path = Path(os.getenv("INDEX_PATH", str(INDEX_DIR / "faiss.index")))
    metadata_path: Path = Path(
        os.getenv("METADATA_PATH", str(INDEX_DIR / "metadata.json"))
    )
    similarity_metric: str = "cosine"  # Options: cosine, l2, inner_product
    nlist: int = 100  # Number of clusters for IVF index
    nprobe: int = 10  # Number of clusters to search


@dataclass
class RetrieverConfig:
    """Configuration for retrieval."""

    top_k: int = int(os.getenv("TOP_K_RETRIEVAL", "5"))
    vector_weight: float = 0.7  # Weight for vector similarity
    bm25_weight: float = 0.3  # Weight for BM25 lexical search
    rerank: bool = True
    min_score: float = 0.3  # Minimum similarity score threshold


@dataclass
class GeneratorConfig:
    """Configuration for LLM generation."""

    backend: str = os.getenv("LLM_BACKEND", "hf")  # Options: hf, openai
    model_name: str = "google/flan-t5-base"  # Default for HF
    openai_model: str = "gpt-3.5-turbo"
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    max_context_length: int = int(os.getenv("MAX_CONTEXT_LENGTH", "3000"))
    max_new_tokens: int = 300
    temperature: float = 0.7
    top_p: float = 0.9
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class SystemConfig:
    """Overall system configuration."""

    embedding: EmbeddingConfig = None
    chunking: ChunkingConfig = None
    index: IndexConfig = None
    retriever: RetrieverConfig = None
    generator: GeneratorConfig = None

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Performance
    num_workers: int = int(os.getenv("NUM_WORKERS", "4"))
    batch_processing: bool = True

    def __post_init__(self):
        if self.embedding is None:
            self.embedding = EmbeddingConfig()
        if self.chunking is None:
            self.chunking = ChunkingConfig()
        if self.index is None:
            self.index = IndexConfig()
        if self.retriever is None:
            self.retriever = RetrieverConfig()
        if self.generator is None:
            self.generator = GeneratorConfig()


# Global config instance
config = SystemConfig()

# Prompt templates
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on provided documents. 
Always cite your sources by mentioning the document name and page number.
If you cannot find relevant information in the provided context, say so clearly.
Be concise and factual in your responses."""

QA_PROMPT_TEMPLATE = """Context information from documents:
{context}

Question: {question}

Instructions:
1. Answer based ONLY on the provided context
2. Cite sources (document name and page) for each claim
3. If uncertain or information is not in context, state this clearly
4. Keep the answer concise and relevant

Answer:"""

# Supported file types
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".html", ".htm"}

# Model download settings
AUTO_DOWNLOAD_MODELS = os.getenv("AUTO_DOWNLOAD_MODELS", "false").lower() == "true"
MAX_MODEL_SIZE_GB = 1  # Don't auto-download models larger than this
