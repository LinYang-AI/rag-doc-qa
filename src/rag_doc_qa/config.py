"""
Configuration module for RAG Document QA System.
Handles environment variables, model settings, and system parameters.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
INDEX_DIR = DATA_DIR / "index"
UPLOAD_DIR = DATA_DIR / "uploads"

# Create directories if they don't exist
for dir_path in [DATA_DIR, CACHE_DIR, INDEX_DIR, UPLOAD_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Detect hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    logger.info("CUDA not available. Using CPU")


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""

    model_name: str = field(
        default_factory=lambda: os.getenv(
            "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    use_openai: bool = field(
        default_factory=lambda: os.getenv("USE_OPENAI_EMBEDDING", "false").lower()
        == "true"
    )
    openai_model: str = "text-embedding-3-small"  # Updated OpenAI model
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    cache_dir: Path = CACHE_DIR / "embeddings"
    batch_size: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    )
    device: str = DEVICE
    max_seq_length: int = 512
    normalize_embeddings: bool = True
    show_progress_bar: bool = True
    _actual_dimension: Optional[int] = None  # Store actual dimension from model

    @property
    def dimension(self) -> int:
        """Get embedding dimension based on model."""
        # If we have the actual dimension from the model, use it
        if self._actual_dimension is not None:
            return self._actual_dimension

        # Otherwise, use model-based defaults
        if self.use_openai:
            return 1536 if "ada" in self.openai_model else 3072
        elif "all-MiniLM-L6-v2" in self.model_name:
            return 384
        elif "all-mpnet-base-v2" in self.model_name:
            return 768
        else:
            return 384  # Default

    def set_dimension(self, dimension: int):
        """Set the actual embedding dimension."""
        self._actual_dimension = dimension


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""

    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "800")))
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200"))
    )
    min_chunk_size: int = field(
        default_factory=lambda: int(os.getenv("MIN_CHUNK_SIZE", "100"))
    )
    separator: str = "\n\n"
    length_function: str = "chars"  # "chars" or "tokens"
    metadata_fields: list = field(
        default_factory=lambda: ["source", "page", "chunk_id", "timestamp"]
    )
    keep_separator: bool = True
    strip_whitespace: bool = True


@dataclass
class IndexConfig:
    """Configuration for FAISS index."""

    index_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("INDEX_PATH", str(INDEX_DIR / "faiss.index"))
        )
    )
    metadata_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("METADATA_PATH", str(INDEX_DIR / "metadata.json"))
        )
    )
    similarity_metric: str = "cosine"  # Options: cosine, l2, inner_product
    index_type: str = "flat"  # Options: flat, ivf, hnsw
    nlist: int = field(default_factory=lambda: int(os.getenv("FAISS_NLIST", "100")))
    nprobe: int = field(default_factory=lambda: int(os.getenv("FAISS_NPROBE", "10")))
    use_gpu: bool = (
        DEVICE == "cuda" and os.getenv("USE_GPU_INDEX", "true").lower() == "true"
    )


@dataclass
class RetrieverConfig:
    """Configuration for retrieval."""

    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K_RETRIEVAL", "5")))
    vector_weight: float = field(
        default_factory=lambda: float(os.getenv("VECTOR_WEIGHT", "0.7"))
    )
    bm25_weight: float = field(
        default_factory=lambda: float(os.getenv("BM25_WEIGHT", "0.3"))
    )
    rerank: bool = field(
        default_factory=lambda: os.getenv("ENABLE_RERANK", "true").lower() == "true"
    )
    min_score: float = field(
        default_factory=lambda: float(os.getenv("MIN_SCORE", "0.3"))
    )
    max_candidates: int = 20  # Maximum candidates to consider before reranking
    use_mmr: bool = False  # Maximal Marginal Relevance for diversity
    mmr_lambda: float = 0.5  # Diversity parameter for MMR


@dataclass
class GeneratorConfig:
    """Configuration for LLM generation."""

    backend: str = field(default_factory=lambda: os.getenv("LLM_BACKEND", "hf"))

    # Hugging Face settings
    hf_model_name: str = field(
        default_factory=lambda: os.getenv(
            "HF_MODEL", "google/flan-t5-small"  # Smaller default for CPU
        )
    )

    # OpenAI settings
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )

    # Generation parameters
    max_context_length: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONTEXT_LENGTH", "3000"))
    )
    max_new_tokens: int = field(
        default_factory=lambda: int(os.getenv("MAX_NEW_TOKENS", "300"))
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("TEMPERATURE", "0.7"))
    )
    top_p: float = field(default_factory=lambda: float(os.getenv("TOP_P", "0.9")))
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True

    # Performance
    device: str = DEVICE
    use_8bit: bool = False  # For quantization
    use_flash_attention: bool = False

    # Streaming
    stream: bool = field(
        default_factory=lambda: os.getenv("ENABLE_STREAMING", "false").lower() == "true"
    )


@dataclass
class SystemConfig:
    """Overall system configuration."""

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)

    # System settings
    num_workers: int = field(default_factory=lambda: int(os.getenv("NUM_WORKERS", "4")))
    batch_processing: bool = True
    enable_caching: bool = True
    cache_ttl: int = 3600  # Cache time-to-live in seconds

    # API settings
    api_rate_limit: int = 100  # Requests per minute
    max_file_size_mb: int = 10
    allowed_file_types: set = field(
        default_factory=lambda: {
            ".pdf",
            ".txt",
            ".md",
            ".html",
            ".htm",
            ".docx",
            ".csv",
        }
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "embedding": {
                "model": self.embedding.model_name,
                "dimension": self.embedding.dimension,
                "device": self.embedding.device,
            },
            "chunking": {
                "size": self.chunking.chunk_size,
                "overlap": self.chunking.chunk_overlap,
            },
            "retriever": {
                "top_k": self.retriever.top_k,
                "weights": f"vector={self.retriever.vector_weight}, bm25={self.retriever.bm25_weight}",
            },
            "generator": {
                "backend": self.generator.backend,
                "model": (
                    self.generator.hf_model_name
                    if self.generator.backend == "hf"
                    else self.generator.openai_model
                ),
            },
            "system": {"device": DEVICE, "workers": self.num_workers},
        }


# Global config instance
config = SystemConfig()

# Prompt templates
SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on provided documents. 
Follow these guidelines:
1. Base your answers ONLY on the provided context
2. Cite sources with [Source: filename, Page: X] format
3. If information is not in the context, clearly state this
4. Be concise and factual
5. Highlight any uncertainties or conflicting information"""

QA_PROMPT_TEMPLATE = """Context from documents:
{context}

Question: {question}

Please provide a comprehensive answer based on the context above. Include source citations for each claim.

Answer:"""

CHAT_PROMPT_TEMPLATE = """Previous conversation:
{chat_history}

Context from documents:
{context}

Current question: {question}

Provide an answer that considers both the conversation history and the document context. Cite sources where applicable.

Answer:"""

# Model recommendations based on hardware
MODEL_RECOMMENDATIONS = {
    "cpu": {
        "embedding": "sentence-transformers/all-MiniLM-L6-v2",
        "generation": "google/flan-t5-small",
    },
    "cuda": {"embedding": "BAAI/bge-base-en-v1.5", "generation": "google/flan-t5-base"},
}


# Export configuration
def get_config() -> SystemConfig:
    """Get the global configuration instance."""
    return config


def update_config(**kwargs) -> SystemConfig:
    """Update configuration with new values."""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def save_config(path: Path):
    """Save configuration to JSON file."""
    import json

    path = Path(path)
    with open(path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    logger.info(f"Configuration saved to {path}")


def load_config(path: Path) -> SystemConfig:
    """Load configuration from JSON file."""
    import json

    global config
    path = Path(path)
    with open(path, "r") as f:
        data = json.load(f)
    # Update config with loaded data
    # This is simplified; in production you'd properly deserialize
    logger.info(f"Configuration loaded from {path}")
    return config
