"""
Document ingestion module for RAG system.
Handles PDF, TXT, and HTML document processing.
"""

import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json

import PyPDF2
from bs4 import BeautifulSoup
import requests
import chardet

from rag_doc_qa.config import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a processed document."""

    content: str
    metadata: Dict[str, Any]
    doc_id: str
    source: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "doc_id": self.doc_id,
            "source": self.source,
        }


class DocumentIngestor:
    """Handles ingestion of various document formats."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.processed_docs: List[Document] = []

    def ingest_file(self, file_path: Path) -> List[Document]:
        """
        Ingest a single file and return processed documents.

        Args:
            file_path: Path to the file to ingest

        Returns:
            List of Document objects
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        logger.info(f"Ingesting file: {file_path}")

        try:
            if file_path.suffix.lower() == ".pdf":
                return self._ingest_pdf(file_path)
            elif file_path.suffix.lower() in [".txt", ".md"]:
                return self._ingest_text(file_path)
            elif file_path.suffix.lower() in [".html", ".htm"]:
                return self._ingest_html(file_path)
        except Exception as e:
            logger.error(f"Error ingesting {file_path}: {e}")
            raise

    def ingest_directory(self, dir_path: Path) -> List[Document]:
        """
        Recursively ingest all supported files in a directory.

        Args:
            dir_path: Path to directory

        Returns:
            List of all processed documents
        """
        dir_path = Path(dir_path)
        all_docs = []

        for file_path in dir_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    docs = self.ingest_file(file_path)
                    all_docs.extend(docs)
                except Exception as e:
                    logger.warning(f"Skipping {file_path}: {e}")

        logger.info(f"Ingested {len(all_docs)} documents from {dir_path}")
        return all_docs

    def ingest_url(self, url: str) -> List[Document]:
        """
        Ingest content from a URL.

        Args:
            url: URL to fetch and ingest

        Returns:
            List of Document objects
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            # Detect content type
            content_type = response.headers.get("content-type", "").lower()

            if "html" in content_type:
                soup = BeautifulSoup(response.text, "html.parser")
                content = self._extract_text_from_html(soup)
            else:
                content = response.text

            doc_id = self._generate_doc_id(content)

            return [
                Document(
                    content=content,
                    metadata={"source_type": "url", "url": url},
                    doc_id=doc_id,
                    source=url,
                )
            ]
        except Exception as e:
            logger.error(f"Error ingesting URL {url}: {e}")
            raise

    def _ingest_pdf(self, file_path: Path) -> List[Document]:
        """Extract text from PDF file."""
        docs = []

        try:
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)

                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        text = page.extract_text()

                        if text.strip():  # Only add non-empty pages
                            doc_id = self._generate_doc_id(f"{file_path}:{page_num}")

                            doc = Document(
                                content=text,
                                metadata={
                                    "source_type": "pdf",
                                    "file_name": file_path.name,
                                    "page": page_num,
                                    "total_pages": len(pdf_reader.pages),
                                },
                                doc_id=doc_id,
                                source=str(file_path),
                            )
                            docs.append(doc)
                    except Exception as e:
                        logger.warning(
                            f"Error extracting page {page_num} from {file_path}: {e}"
                        )
        except PyPDF2.errors.PdfReadError as e:
            logger.error(f"Corrupted PDF file {file_path}: {e}")
            raise ValueError(f"Cannot read PDF file: {file_path}")

        return docs

    def _ingest_text(self, file_path: Path) -> List[Document]:
        """Extract text from plain text file."""
        try:
            # Detect encoding
            with open(file_path, "rb") as file:
                raw_data = file.read()
                detected = chardet.detect(raw_data)
                encoding = detected.get("encoding", "utf-8")

            # Read with detected encoding
            with open(file_path, "r", encoding=encoding) as file:
                content = file.read()

            doc_id = self._generate_doc_id(str(file_path))

            return [
                Document(
                    content=content,
                    metadata={
                        "source_type": "text",
                        "file_name": file_path.name,
                        "encoding": encoding,
                    },
                    doc_id=doc_id,
                    source=str(file_path),
                )
            ]
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {e}")
            raise

    def _ingest_html(self, file_path: Path) -> List[Document]:
        """Extract text from HTML file."""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                soup = BeautifulSoup(file.read(), "html.parser")

            content = self._extract_text_from_html(soup)
            doc_id = self._generate_doc_id(str(file_path))

            # Extract title if available
            title = soup.find("title")
            title_text = title.string if title else file_path.name

            return [
                Document(
                    content=content,
                    metadata={
                        "source_type": "html",
                        "file_name": file_path.name,
                        "title": title_text,
                    },
                    doc_id=doc_id,
                    source=str(file_path),
                )
            ]
        except Exception as e:
            logger.error(f"Error parsing HTML file {file_path}: {e}")
            raise

    def _extract_text_from_html(self, soup: BeautifulSoup) -> str:
        """Extract clean text from BeautifulSoup object."""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text and clean it
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = " ".join(chunk for chunk in chunks if chunk)

        return text

    def _generate_doc_id(self, content: str) -> str:
        """Generate deterministic document ID."""
        return hashlib.md5(content.encode()).hexdigest()

    def save_documents(self, docs: List[Document], output_path: Path):
        """Save processed documents to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump([doc.to_dict() for doc in docs], f, indent=2)

        logger.info(f"Saved {len(docs)} documents to {output_path}")

    @staticmethod
    def load_documents(input_path: Path) -> List[Document]:
        """Load documents from JSON."""
        with open(input_path, "r") as f:
            data = json.load(f)

        return [
            Document(
                content=item["content"],
                metadata=item["metadata"],
                doc_id=item["doc_id"],
                source=item["source"],
            )
            for item in data
        ]
