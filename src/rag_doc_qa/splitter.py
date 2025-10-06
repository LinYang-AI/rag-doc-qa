"""
Text splitting and chunking module with advanced strategies.
Implements context-aware chunking with overlap and metadata preservation.
"""

import hashlib
import re
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np
from tqdm import tqdm

from rag_doc_qa.config import ChunkingConfig
from rag_doc_qa.ingest import Document

logger = logging.getLogger(__name__)

class ChunkingStrategy(Enum):
    """Enumeration of chunking strategies."""
    FIXED = "fixed"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"

@dataclass
class Chunk:
    """Enhanced chunk with rich metadata and utility methods."""
    text: str
    chunk_id: str = ""
    doc_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_char: int = 0
    end_char: int = 0
    chunk_index: int = 0
    embedding: Optional[np.ndarray] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Generate chunk_id if not provided."""
        if not self.chunk_id:
            self.chunk_id = self._generate_id()
            
    def _generate_id(self) -> str:
        """Generate deterministic chunk ID."""
        content = f"{self.doc_id}:{self.chunk_index}:{self.text[:50]}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "metadata": self.metadata,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "chunk_index": self.chunk_index,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """Create Chunk from dictionary."""
        if "timestamp" in data:
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)
    
    @property
    def length(self) -> int:
        """Get chunk length in characters."""
        return len(self.text)
    
    def get_context_window(self, window_size: int = 100) -> Tuple[str, str]:
        """Get surrounding context from original document."""
        # This would need access to original document
        # Placeholder for demonstration
        return "", ""

class TextSplitter:
    """
    Advanced text splitter with multiple strategies and optimizations.
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self.chunk_cache: Dict[str, List[Chunk]] = {}
        self._sentence_splitter = None
        self._init_nlp_components()
    
    def _init_nlp_components(self):
        """Initialize NLP components for advanced splitting."""
        try:
            # Try to use spaCy for better sentence splitting
            import spacy
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except:
                # Fallback to basic model
                logger.info("spaCy model not found, using regex-based splitting")
                self._nlp = None
        except ImportError:
            logger.info("spaCy not installed, using regex-based splitting")
            self._nlp = None
    
    def split_documents(self, 
                       documents: List[Document],
                       strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
                       show_progress: bool = True) -> List[Chunk]:
        """
        Split multiple documents into chunks using specified strategy.
        
        Args:
            documents: List of Document objects
            strategy: Chunking strategy to use
            show_progress: Show progress bar
            
        Returns:
            List of Chunk objects
        """
        all_chunks = []
        
        for doc in tqdm(documents, desc="Splitting documents", disable=not show_progress):
            chunks = self.split_document(doc, strategy)
            all_chunks.extend(chunks)
        
        logger.info(f"Split {len(documents)} documents into {len(all_chunks)} chunks")
        logger.info(f"Average chunks per document: {len(all_chunks) / len(documents):.1f}")
        
        return all_chunks
    
    def split_document(self,
                      document: Document,
                      strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE) -> List[Chunk]:
        """
        Split a single document into chunks using specified strategy.
        
        Args:
            document: Document to split
            strategy: Chunking strategy
            
        Returns:
            List of Chunk objects
        """
        # Check cache
        cache_key = f"{document.doc_id}:{strategy.value}"
        if cache_key in self.chunk_cache:
            return self.chunk_cache[cache_key]
        
        # Select splitting method based on strategy
        if strategy == ChunkingStrategy.FIXED:
            chunks = self._split_fixed(document)
        elif strategy == ChunkingStrategy.SENTENCE:
            chunks = self._split_by_sentence(document)
        elif strategy == ChunkingStrategy.PARAGRAPH:
            chunks = self._split_by_paragraph(document)
        elif strategy == ChunkingStrategy.SEMANTIC:
            chunks = self._split_semantic(document)
        else:  # RECURSIVE (default)
            chunks = self._split_recursive(document)
        
        # Cache results
        if self.config.keep_separator:
            self.chunk_cache[cache_key] = chunks
        
        return chunks
    
    def _split_recursive(self, document: Document) -> List[Chunk]:
        """
        Recursively split text using multiple separators.
        Best general-purpose strategy.
        """
        text = document.content
        
        # Define separators in order of preference
        separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
        
        chunks = []
        current_chunks = [text]
        
        for separator in separators:
            if not current_chunks:
                break
            
            new_chunks = []
            for chunk_text in current_chunks:
                if len(chunk_text) <= self.config.chunk_size:
                    new_chunks.append(chunk_text)
                else:
                    # Split by current separator
                    if separator:
                        parts = chunk_text.split(separator)
                        
                        current_part = []
                        current_length = 0
                        
                        for part in parts:
                            part_length = len(part) + len(separator)
                            
                            if current_length + part_length > self.config.chunk_size:
                                if current_part:
                                    combined = separator.join(current_part)
                                    if self.config.keep_separator and separator:
                                        combined += separator
                                    new_chunks.append(combined)
                                    
                                    # Handle overlap
                                    if self.config.chunk_overlap > 0:
                                        overlap_text = self._get_overlap_text(current_part, separator)
                                        current_part = [overlap_text] if overlap_text else []
                                        current_length = len(overlap_text) if overlap_text else 0
                                    else:
                                        current_part = []
                                        current_length = 0
                                
                                current_part.append(part)
                                current_length = part_length
                            else:
                                current_part.append(part)
                                current_length += part_length
                        
                        # Add remaining
                        if current_part:
                            combined = separator.join(current_part)
                            if combined.strip():
                                new_chunks.append(combined)
                    else:
                        # Character-level splitting as last resort
                        for i in range(0, len(chunk_text), self.config.chunk_size - self.config.chunk_overlap):
                            new_chunks.append(chunk_text[i:i + self.config.chunk_size])
            
            current_chunks = new_chunks
        
        # Convert text chunks to Chunk objects
        return self._create_chunks_from_texts(current_chunks, document)
    
    def _split_fixed(self, document: Document) -> List[Chunk]:
        """Simple fixed-size chunking."""
        text = document.content
        chunks = []
        
        for i in range(0, len(text), self.config.chunk_size - self.config.chunk_overlap):
            chunk_text = text[i:i + self.config.chunk_size]
            if len(chunk_text.strip()) >= self.config.min_chunk_size:
                chunks.append(chunk_text)
        
        return self._create_chunks_from_texts(chunks, document)
    
    def _split_by_sentence(self, document: Document) -> List[Chunk]:
        """Split by sentences using NLP or regex."""
        text = document.content
        
        if self._nlp:
            # Use spaCy for sentence segmentation
            doc = self._nlp(text[:1000000])  # Limit to 1M chars for spaCy
            sentences = [sent.text.strip() for sent in doc.sents]
        else:
            # Fallback to regex
            sentences = self._split_sentences_regex(text)
        
        # Group sentences into chunks
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence) + 1  # +1 for space
            
            if current_length + sentence_length > self.config.chunk_size:
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(chunk_text)
                    
                    # Handle overlap
                    if self.config.chunk_overlap > 0:
                        overlap_sentences = self._get_overlap_sentences(
                            current_chunk, self.config.chunk_overlap
                        )
                        current_chunk = overlap_sentences
                        current_length = sum(len(s) + 1 for s in overlap_sentences)
                    else:
                        current_chunk = []
                        current_length = 0
                
                current_chunk.append(sentence)
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add remaining
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return self._create_chunks_from_texts(chunks, document)
    
    def _split_by_paragraph(self, document: Document) -> List[Chunk]:
        """Split by paragraphs."""
        text = document.content
        
        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n+', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_length = len(para) + 2  # +2 for paragraph break
            
            if para_length > self.config.chunk_size:
                # Paragraph too long, split it further
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split long paragraph by sentences
                sub_doc = Document(content=para, doc_id=document.doc_id, source=document.source)
                sub_chunks = self._split_by_sentence(sub_doc)
                chunks.extend([c.text for c in sub_chunks])
            elif current_length + para_length > self.config.chunk_size:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    
                    # Handle overlap
                    if self.config.chunk_overlap > 0:
                        # Keep last paragraph for overlap
                        current_chunk = [current_chunk[-1]] if len(current_chunk) > 0 else []
                        current_length = len(current_chunk[0]) if current_chunk else 0
                    else:
                        current_chunk = []
                        current_length = 0
                
                current_chunk.append(para)
                current_length = para_length
            else:
                current_chunk.append(para)
                current_length += para_length
        
        # Add remaining
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        
        return self._create_chunks_from_texts(chunks, document)
    
    def _split_semantic(self, document: Document) -> List[Chunk]:
        """
        Split based on semantic similarity (requires embeddings).
        This is a simplified version - full implementation would use
        embedding model to find semantic boundaries.
        """
        # For now, fall back to recursive splitting
        # Full implementation would:
        # 1. Embed sentences
        # 2. Find semantic boundaries using embedding similarity
        # 3. Group similar sentences into chunks
        logger.info("Semantic splitting not fully implemented, using recursive strategy")
        return self._split_recursive(document)
    
    def _split_sentences_regex(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        # Improved sentence splitting regex
        sentence_endings = re.compile(
            r'(?<=[.!?])\s+'  # Standard sentence endings
            r'|(?<=[.!?]["\'])\s+'  # Quoted sentence endings  
            r'|(?<=\n)\s*\n+'  # Paragraph breaks
        )
        
        sentences = sentence_endings.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_text(self, parts: List[str], separator: str) -> str:
        """Get text for overlap from parts list."""
        if not parts:
            return ""
        
        total_length = 0
        overlap_parts = []
        
        # Work backwards to get overlap text
        for part in reversed(parts):
            part_length = len(part) + len(separator)
            total_length += part_length
            overlap_parts.insert(0, part)
            
            if total_length >= self.config.chunk_overlap:
                break
        
        return separator.join(overlap_parts)
    
    def _get_overlap_sentences(self, sentences: List[str], overlap_size: int) -> List[str]:
        """Get sentences for overlap."""
        if not sentences:
            return []
        
        total_length = 0
        overlap_sentences = []
        
        for sent in reversed(sentences):
            total_length += len(sent) + 1
            overlap_sentences.insert(0, sent)
            
            if total_length >= overlap_size:
                break
        
        return overlap_sentences
    
    def _create_chunks_from_texts(self, texts: List[str], document: Document) -> List[Chunk]:
        """Convert text pieces to Chunk objects with metadata."""
        chunks = []
        current_pos = 0
        
        for i, text in enumerate(texts):
            if self.config.strip_whitespace:
                text = text.strip()
            
            if len(text) < self.config.min_chunk_size:
                continue
            
            # Find position in original document
            start_pos = document.content.find(text, current_pos)
            if start_pos == -1:
                start_pos = current_pos
            
            end_pos = start_pos + len(text)
            current_pos = end_pos
            
            # Create chunk with metadata
            chunk = Chunk(
                text=text,
                doc_id=document.doc_id,
                metadata={
                    **document.metadata,
                    "chunk_index": i,
                    "source": document.source,
                    "chunking_strategy": "recursive"
                },
                start_char=start_pos,
                end_char=end_pos,
                chunk_index=i
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def get_chunk_stats(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Get detailed statistics about chunks."""
        if not chunks:
            return {"error": "No chunks provided"}
        
        lengths = [chunk.length for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "total_characters": sum(lengths),
            "avg_chunk_size": np.mean(lengths),
            "median_chunk_size": np.median(lengths),
            "std_chunk_size": np.std(lengths),
            "min_chunk_size": min(lengths),
            "max_chunk_size": max(lengths),
            "chunks_below_min": sum(1 for l in lengths if l < self.config.min_chunk_size),
            "chunks_above_max": sum(1 for l in lengths if l > self.config.chunk_size),
        }
    
    def visualize_chunks(self, chunks: List[Chunk], max_display: int = 5):
        """Visualize chunk distribution for debugging."""
        import matplotlib.pyplot as plt
        
        lengths = [chunk.length for chunk in chunks]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram
        ax1.hist(lengths, bins=30, edgecolor='black')
        ax1.axvline(self.config.chunk_size, color='r', linestyle='--', label='Target size')
        ax1.axvline(self.config.min_chunk_size, color='g', linestyle='--', label='Min size')
        ax1.set_xlabel('Chunk Size (characters)')
        ax1.set_ylabel('Count')
        ax1.set_title('Chunk Size Distribution')
        ax1.legend()
        
        # Box plot
        ax2.boxplot(lengths)
        ax2.set_ylabel('Chunk Size (characters)')
        ax2.set_title('Chunk Size Statistics')
        
        plt.tight_layout()
        plt.show()
        
        # Display sample chunks
        print(f"\nFirst {max_display} chunks:")
        for i, chunk in enumerate(chunks[:max_display]):
            print(f"\nChunk {i+1} (length: {chunk.length}):")
            print(f"  {chunk.text[:100]}..." if chunk.length > 100 else f"  {chunk.text}")