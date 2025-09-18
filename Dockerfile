# Multi-stage build for smaller image
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime stage
FROM python:3.10-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 raguser

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/raguser/.local

# Copy application code
COPY --chown=raguser:raguser src/ ./src/
COPY --chown=raguser:raguser examples/ ./examples/
COPY --chown=raguser:raguser scripts/ ./scripts/

# Create data directories
RUN mkdir -p /app/data/cache /app/data/index && \
    chown -R raguser:raguser /app/data

# Switch to non-root user
USER raguser

# Update PATH
ENV PATH=/home/raguser/.local/bin:$PATH
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Environment variables
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"

# CPU-optimized settings
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Default command
CMD ["python", "-m", "rag_doc_qa.web_app"]

# GPU Support (uncomment for GPU builds)
# FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as gpu-base
# ... (add GPU-specific configuration)
# RUN pip install faiss-gpu torch --index-url https://download.pytorch.org/whl/cu118