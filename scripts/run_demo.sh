#!/bin/bash

# Run the RAG Document QA Demo

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔═══════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   RAG Document QA System - Web Demo   ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════╝${NC}"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

# Set paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_DIR}/data"

# Check for .env file
if [ ! -f "${PROJECT_DIR}/.env" ]; then
    echo -e "${YELLOW}Warning: No .env file found${NC}"
    echo "Creating default .env file..."
    cat > "${PROJECT_DIR}/.env" << EOF
# RAG Configuration
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_BACKEND=hf
CHUNK_SIZE=800
CHUNK_OVERLAP=200
TOP_K_RETRIEVAL=5
MAX_CONTEXT_LENGTH=3000

# Paths
INDEX_PATH=./data/index/faiss.index
METADATA_PATH=./data/index/metadata.json
CACHE_DIR=./data/cache

# Optional: OpenAI Configuration
# OPENAI_API_KEY=your-key-here
# USE_OPENAI_EMBEDDING=false
EOF
    echo -e "${GREEN}✓ Created default .env file${NC}"
fi

# Check if index exists
if [ ! -f "${DATA_DIR}/index/faiss.index" ]; then
    echo -e "${YELLOW}No index found. Building index first...${NC}"
    bash "${SCRIPT_DIR}/build_index.sh"
fi

# Set Python path
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH}"

# Parse command line arguments
PORT=${1:-7860}
HOST=${2:-0.0.0.0}

echo -e "${BLUE}Starting web server...${NC}"
echo -e "  Host: ${HOST}"
echo -e "  Port: ${PORT}"
echo ""

# Check if port is already in use
if lsof -Pi :${PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${RED}Error: Port ${PORT} is already in use${NC}"
    echo "Please specify a different port: $0 [port]"
    exit 1
fi

# Run the web app
cd "${PROJECT_DIR}"

echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo -e "${GREEN}Server starting...${NC}"
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo ""
echo -e "📚 ${BLUE}Access the UI at:${NC}"
echo -e "   ${GREEN}http://localhost:${PORT}${NC}"
echo ""
echo -e "🔌 ${BLUE}API Endpoints:${NC}"
echo -e "   POST ${GREEN}http://localhost:${PORT}/api/query${NC}"
echo -e "   POST ${GREEN}http://localhost:${PORT}/api/upload${NC}"
echo -e "   GET  ${GREEN}http://localhost:${PORT}/api/stats${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo ""

# Start the application
python3 -m rag_doc_qa.web_app

# Cleanup on exit
echo ""
echo -e "${YELLOW}Server stopped${NC}"