#!/bin/bash

# Build index for RAG Document QA System

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}RAG Document QA - Index Builder${NC}"
echo "================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

# Set paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_DIR}/data"
DOCS_DIR="${PROJECT_DIR}/examples/sample_docs"

# Create data directories
echo -e "${YELLOW}Creating data directories...${NC}"
mkdir -p "${DATA_DIR}/index"
mkdir -p "${DATA_DIR}/cache"

# Check for documents
if [ -z "$(ls -A ${DOCS_DIR} 2>/dev/null)" ]; then
    echo -e "${YELLOW}No documents found in ${DOCS_DIR}${NC}"
    echo "Please add documents to index or specify a different directory"
    echo "Usage: $0 [document_directory]"
    exit 0
fi

# Use provided directory or default
if [ $# -eq 1 ]; then
    DOCS_DIR="$1"
fi

echo -e "${GREEN}Indexing documents from: ${DOCS_DIR}${NC}"

# Set Python path
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH}"

# Run indexing
python3 -c "
import sys
import os
sys.path.insert(0, '${PROJECT_DIR}/src')
os.chdir('${PROJECT_DIR}')

from rag_doc_qa import DocumentIngestor, TextSplitter, EmbeddingModel, FAISSIndex
from rag_doc_qa.config import SystemConfig
from pathlib import Path

# Initialize components
print('Initializing components...')
config = SystemConfig()
ingestor = DocumentIngestor()
splitter = TextSplitter(config.chunking)
embedding_model = EmbeddingModel(config.embedding)
index = FAISSIndex(config.index, embedding_model)

# Ingest documents
print(f'Ingesting documents from {Path(\"${DOCS_DIR}\")}...')
documents = ingestor.ingest_directory(Path('${DOCS_DIR}'))
print(f'Ingested {len(documents)} documents')

# Split into chunks
print('Splitting documents into chunks...')
chunks = splitter.split_documents(documents)
print(f'Created {len(chunks)} chunks')

# Build index
print('Building FAISS index...')
index.create_index(chunks, rebuild=True)

# Get stats
stats = index.get_stats()
print(f'Index built successfully!')
print(f'  - Vectors: {stats[\"num_vectors\"]}')
print(f'  - Dimension: {stats[\"dimension\"]}')
print(f'  - Size: {stats[\"total_size_mb\"]:.2f} MB')
print(f'  - Documents: {stats[\"num_documents\"]}')
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Index built successfully!${NC}"
    echo -e "Index location: ${DATA_DIR}/index/"
else
    echo -e "${RED}✗ Index building failed${NC}"
    exit 1
fi