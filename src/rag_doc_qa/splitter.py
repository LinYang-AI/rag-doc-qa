"""
Text splitting and chunking module.
Implements context-aware chunking with overlap and metadata preservation.
"""

import hashlib
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from rag_doc_qa.config import ChunkingConfig
from rag_doc_qa.ingest import Document

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""

    text: str
    chunk_id: str
    doc_id: str
    metadata: Dict[str, Any]
    start_char: int
    end_char: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "metadata": self.metadata,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


class TextSplitter:
    """
    Context-aware text splitter with configurable chunk size and overlap.
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self.chunk_cache: Dict[str, List[Chunk]] = {}

    def split_documents(self, documents: List[Document]) -> List[Chunk]:
        """
        Split multiple documents into chunks.

        Args:
            documents: List of Document objects

        Returns:
            List of Chunk objects
        """
        all_chunks = []

        for doc in documents:
            chunks = self.split_document(doc)
            all_chunks.extend(chunks)

        logger.info(f"Split {len(documents)} documents into {len(all_chunks)} chunks")
        return all_chunks

    def split_document(self, document: Document) -> List[Chunk]:
        """
        Split a single document into chunks.

        Args:
            document: Document object to split

        Returns:
            List of Chunk objects
        """
        # Check cache
        logger.debug(f"Checking cache for document: {document.doc_id}")
        logger.debug(f"Cache keys: {list(self.chunk_cache.keys())}")
        
        if document.doc_id in self.chunk_cache:
            logger.debug(f"Found in cache, returning {len(self.chunk_cache[document.doc_id])} chunks")
            return self.chunk_cache[document.doc_id]

        text = document.content
        chunks = []

        # Split by paragraphs first for better context preservation
        paragraphs = self._split_paragraphs(text)
        current_chunk = []
        current_length = 0
        start_char = 0

        for para in paragraphs:
            para_length = len(para)

            logger.debug(f"Paragraph length: {para_length}")
            # If paragraph is too long, split it further
            if para_length > self.config.chunk_size:
                # Flush current chunk if not empty
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    logger.debug(f"Final chunk length: {len(chunk_text)}")
                    chunks.append(
                        self._create_chunk(
                            chunk_text,
                            document,
                            start_char,
                            start_char + len(chunk_text),
                        )
                    )
                    current_chunk = []
                    current_length = 0
                    start_char += len(chunk_text) + 2  # Account for \n\n

                # Split long paragraph
                sub_chunks = self._split_long_text(para, document, start_char)
                chunks.extend(sub_chunks)
                start_char += para_length + 2

            # If adding paragraph exceeds chunk size, start new chunk with overlap
            elif current_length + para_length > self.config.chunk_size:
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append(
                        self._create_chunk(
                            chunk_text,
                            document,
                            start_char,
                            start_char + len(chunk_text),
                        )
                    )

                    # Handle overlap
                    if self.config.chunk_overlap > 0:
                        overlap_text = self._get_overlap_text(current_chunk)
                        current_chunk = [overlap_text] if overlap_text else []
                        current_length = len(overlap_text) if overlap_text else 0
                        start_char = start_char + len(chunk_text) - len(overlap_text)
                    else:
                        current_chunk = []
                        current_length = 0
                        start_char += len(chunk_text) + 2

                current_chunk.append(para)
                current_length += para_length

            else:
                current_chunk.append(para)
                current_length += para_length

        # Add remaining chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(
                self._create_chunk(
                    chunk_text, document, start_char, start_char + len(chunk_text)
                )
            )

        # Cache the chunks
        self.chunk_cache[document.doc_id] = chunks
        logger.info(f"Total chunks: {len(chunks)}")
        logger.info(
            f"Total captured length: {sum(len(c.text) for c in chunks)} / {len(text)}"
        )
        return chunks

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split by double newline or single newline followed by indentation
        paragraphs = re.split(r"\n\n+|\n(?=\s{2,})", text)
        result = [p.strip() for p in paragraphs if p.strip()]
        logger.debug(
            f"Paragraph splitting: input length {len(text)}, found {len(result)} paragraphs."
        )
        if result:
            logger.debug(f"First paragraph length: {len(result[0])}")
        return result
    # def _split_paragraphs(self, text: str) -> List[str]:
    #     """Split text into paragraphs."""
    #     # Split by double newline or single newline followed by indentation
    #     paragraphs = re.split(r"\n\n+|\n(?=\s{2,})", text)
    #     return [p.strip() for p in paragraphs if p.strip()]

    def _split_long_text(
        self, text: str, document: Document, start_offset: int
    ) -> List[Chunk]:
        """Split text that's longer than chunk_size."""
        chunks = []
        sentences = self._split_sentences(text)

        # Fallback: if sentence splitting fails, treat the whole text as one chunk
        if not sentences:
            logger.warning("No sentences found, using whole text as one chunk.")
            chunks.append(
                self._create_chunk(
                    text, document, start_offset, start_offset + len(text)
                )
            )
            return chunks

        current_chunk = []
        current_length = 0
        chunk_start = start_offset

        for sentence in sentences:
            sent_length = len(sentence)
            logger.debug(f"Sentence length: {sent_length}")
            if current_length + sent_length > self.config.chunk_size:
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(
                        self._create_chunk(
                            chunk_text,
                            document,
                            chunk_start,
                            chunk_start + len(chunk_text),
                        )
                    )

                    # Handle overlap at sentence level
                    if self.config.chunk_overlap > 0:
                        overlap_sents = self._get_overlap_sentences(current_chunk)
                        current_chunk = overlap_sents
                        current_length = sum(len(s) for s in overlap_sents)
                        chunk_start = chunk_start + len(chunk_text) - current_length
                    else:
                        current_chunk = []
                        current_length = 0
                        chunk_start += len(chunk_text) + 1

                current_chunk.append(sentence)
                current_length += sent_length

        # Add remaining sentences
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            logger.debug(f"Final long chunk length: {len(chunk_text)}")
            chunks.append(
                self._create_chunk(
                    chunk_text, document, chunk_start, chunk_start + len(chunk_text)
                )
            )
        logger.info(f"long text split into {len(chunks)} chunks")
        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting (can be improved with spaCy or NLTK)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _get_overlap_text(self, chunks: List[str]) -> str:
        """Get overlap text from the end of chunks list."""
        if not chunks:
            return ""

        overlap_chars = self.config.chunk_overlap
        full_text = "\n\n".join(chunks)

        if len(full_text) <= overlap_chars:
            return full_text

        # Find a good break point near the overlap size
        overlap_text = full_text[-overlap_chars:]

        # Try to break at paragraph or sentence boundary
        para_break = overlap_text.find("\n\n")
        if para_break > 0:
            overlap_text = overlap_text[para_break + 2 :]
        else:
            sent_break = overlap_text.find(". ")
            if sent_break > 0 and sent_break < len(overlap_text) - 2:
                overlap_text = overlap_text[sent_break + 2 :]

        return overlap_text.strip()

    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get overlap sentences based on overlap size."""
        if not sentences:
            return []

        total_length = 0
        overlap_sents = []

        # Work backwards to get sentences for overlap
        for sent in reversed(sentences):
            total_length += len(sent)
            overlap_sents.insert(0, sent)

            if total_length >= self.config.chunk_overlap:
                break

        return overlap_sents

    def _create_chunk(
        self, text: str, document: Document, start_char: int, end_char: int
    ) -> Chunk:
        """Create a Chunk object with metadata."""
        # Generate deterministic chunk ID
        chunk_id = self._generate_chunk_id(document.doc_id, text)

        # Combine document metadata with chunk-specific metadata
        metadata = document.metadata.copy()
        metadata.update(
            {
                "chunk_index": len(self.chunk_cache.get(document.doc_id, [])),
                "source": document.source,
                "char_range": f"{start_char}-{end_char}",
            }
        )

        return Chunk(
            text=text,
            chunk_id=chunk_id,
            doc_id=document.doc_id,
            metadata=metadata,
            start_char=start_char,
            end_char=end_char,
        )

    def _generate_chunk_id(self, doc_id: str, text: str) -> str:
        """Generate deterministic chunk ID."""
        content = f"{doc_id}:{text[:100]}"  # Use first 100 chars for uniqueness
        return hashlib.md5(content.encode()).hexdigest()

    def get_chunk_stats(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Get statistics about chunks."""
        if not chunks:
            return {}

        lengths = [len(chunk.text) for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(lengths) / len(lengths),
            "min_chunk_size": min(lengths),
            "max_chunk_size": max(lengths),
            "total_characters": sum(lengths),
        }

    
