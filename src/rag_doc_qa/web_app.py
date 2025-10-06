"""
Modern Gradio web interface for RAG Document QA system.
Includes REST API, async support, and real-time features.
"""

import gradio as gr
import logging
import json
import time
import asyncio
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from rag_doc_qa.ingest import DocumentIngestor
from rag_doc_qa.splitter import TextSplitter, ChunkingStrategy
from rag_doc_qa.embeddings import EmbeddingModel
from rag_doc_qa.indexer import FAISSIndex
from rag_doc_qa.retriever import HybridRetriever
from rag_doc_qa.generator import LLMGenerator
from rag_doc_qa.evaluate import RAGEvaluator
from rag_doc_qa.config import SystemConfig, get_config, save_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """Complete RAG pipeline with async support."""

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()

        # Initialize components
        logger.info("Initializing RAG pipeline components...")
        self.ingestor = DocumentIngestor()
        self.splitter = TextSplitter(self.config.chunking)
        self.embedding_model = EmbeddingModel(self.config.embedding)
        self.index = FAISSIndex(self.config.index, self.embedding_model)
        self.retriever = HybridRetriever(self.config.retriever, self.index)
        self.generator = LLMGenerator(self.config.generator)
        self.evaluator = RAGEvaluator()

        # Session management
        self.sessions = {}
        self.current_session_id = None

        # Load existing index if available
        if self.index.index_exists():
            try:
                self.index.load()
                self.retriever._initialize_sparse_retrievers()
                logger.info("Loaded existing index")
            except Exception as e:
                logger.warning(f"Could not load index: {e}")

    def ingest_documents(self, file_paths, chunking_strategy: str = "recursive"):
        """
        Ingest documents and update index.

        Args:
            file_paths: List of file paths
            chunking_strategy: Strategy for chunking

        Returns:
            Ingestion report
        """
        try:
            start_time = datetime.now()

            # Ingest documents
            all_docs = []
            failed_files = []

            for file_path in file_paths:
                try:
                    docs = self.ingestor.ingest_file(Path(file_path))
                    all_docs.extend(docs)
                except Exception as e:
                    logger.error(f"Failed to ingest {file_path}: {e}")
                    failed_files.append({"file": file_path, "error": str(e)})

            if not all_docs:
                return {
                    "status": "error",
                    "message": "No documents were successfully ingested",
                    "failed_files": failed_files,
                }

            # Split into chunks
            strategy = ChunkingStrategy[chunking_strategy.upper()]
            chunks = self.splitter.split_documents(all_docs, strategy)

            # Index chunks
            self.index.create_index(chunks)

            # Reinitialize sparse retrievers
            self.retriever._initialize_sparse_retrievers()

            # Calculate statistics
            elapsed = (datetime.now() - start_time).total_seconds()
            chunk_stats = self.splitter.get_chunk_stats(chunks)

            return {
                "status": "success",
                "documents_ingested": len(all_docs),
                "chunks_created": len(chunks),
                "failed_files": failed_files,
                "processing_time": elapsed,
                "chunk_statistics": chunk_stats,
            }

        except Exception as e:
            logger.error(f"Error in document ingestion: {e}")
            return {"status": "error", "message": str(e)}

    def query(
        self,
        question: str,
        top_k: int = 5,
        use_chat_history: bool = False,
        session_id: Optional[str] = None,
    ):
        """
        Process a query and return answer with sources.

        Args:
            question: User question
            top_k: Number of chunks to retrieve
            use_chat_history: Whether to use conversation history
            session_id: Session ID for conversation tracking

        Returns:
            Tuple of (answer, sources, metadata)
        """
        try:
            start_time = datetime.now()

            # Retrieve relevant chunks
            retrieval_results = self.retriever.retrieve(question, top_k)

            if not retrieval_results:
                return (
                    "I couldn't find relevant information in the documents to answer your question.",
                    [],
                    {"status": "no_results"},
                )

            # Format context
            context = self.retriever.get_context(retrieval_results)

            # Get chat history if requested
            chat_history = None
            if use_chat_history and session_id and session_id in self.sessions:
                chat_history = self._format_chat_history(self.sessions[session_id])

            # Generate answer
            generation_result = self.generator.generate(
                question, context, retrieval_results, chat_history=chat_history
            )

            # Update session
            if session_id:
                if session_id not in self.sessions:
                    self.sessions[session_id] = []
                self.sessions[session_id].append(
                    {
                        "question": question,
                        "answer": generation_result.text,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            # Format sources
            sources = []
            for i, result in enumerate(retrieval_results[:3], 1):
                sources.append(
                    {
                        "rank": i,
                        "text": (
                            result.text[:300] + "..."
                            if len(result.text) > 300
                            else result.text
                        ),
                        "source": result.metadata.get("source", "Unknown"),
                        "page": result.metadata.get("page", "N/A"),
                        "score": result.final_score,
                    }
                )

            # Compile metadata
            elapsed = (datetime.now() - start_time).total_seconds()
            metadata = {
                "status": "success",
                "retrieval_time": self.retriever.stats["avg_retrieval_time"],
                "generation_time": generation_result.generation_time,
                "total_time": elapsed,
                "tokens_used": generation_result.tokens_used,
                "confidence": generation_result.confidence_score,
                "chunks_retrieved": len(retrieval_results),
                "model": generation_result.metadata.get("model", "unknown"),
            }

            return generation_result.text, sources, metadata

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return (
                f"An error occurred: {str(e)}",
                [],
                {"status": "error", "error": str(e)},
            )

    async def query_async(self, question: str, **kwargs):
        """Async version of query for better concurrency."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.query, question, **kwargs)

    def _format_chat_history(self, session: List[Dict]) -> str:
        """Format session history for context."""
        history = []
        for entry in session[-5:]:  # Last 5 exchanges
            history.append(f"Q: {entry['question']}")
            history.append(f"A: {entry['answer']}")
        return "\n".join(history)

    def get_statistics(self):
        """Get comprehensive system statistics."""
        return {
            "index": self.index.get_stats(),
            "retriever": self.retriever.get_stats(),
            "generator": self.generator.get_stats(),
            "embeddings": self.embedding_model.get_model_info(),
            "config": self.config.to_dict(),
        }


# Global pipeline instance
pipeline = None
app = FastAPI(title="RAG Document QA API")


def initialize_pipeline():
    """Initialize the global RAG pipeline."""
    global pipeline
    if pipeline is None:
        pipeline = RAGPipeline()
        # Warm up models
        pipeline.embedding_model.warmup()
    return "✅ Pipeline initialized successfully!"


# FastAPI endpoints
@app.post("/api/ingest")
async def api_ingest(files: List[UploadFile] = File(...)):
    """Ingest uploaded documents."""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    file_paths = []
    for file in files:
        path = Path(f"/tmp/{file.filename}")
        content = await file.read()
        path.write_bytes(content)
        file_paths.append(str(path))

    result = pipeline.ingest_documents(file_paths)
    return JSONResponse(content=result)


@app.post("/api/query")
async def api_query(question: str, top_k: int = 5, session_id: Optional[str] = None):
    """Process a query."""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    answer, sources, metadata = await pipeline.query_async(
        question, top_k, session_id=session_id
    )

    return JSONResponse(
        content={"answer": answer, "sources": sources, "metadata": metadata}
    )


@app.get("/api/stats")
async def api_stats():
    """Get system statistics."""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    return JSONResponse(content=pipeline.get_statistics())


# Gradio interface functions
def upload_and_index(files, strategy):
    """Handle file upload and indexing."""
    if not pipeline:
        return "⚠️ Please initialize the pipeline first!", ""

    if not files:
        return "⚠️ No files uploaded", ""

    # Normalize Gradio File input to a list of file paths
    def _to_paths(fs):
        """Convert various gr.File outputs into a list of file path strings."""
        if fs is None:
            return []
        # Single string or Path
        if isinstance(fs, (str, Path)):
            return [str(fs)]
        # If it's not a list/tuple, wrap it
        items = fs if isinstance(fs, (list, tuple)) else [fs]
        paths = []
        for f in items:
            if f is None:
                continue
            if isinstance(f, (str, Path)):
                paths.append(str(f))
                continue
            if isinstance(f, dict):
                # Possible dict structure from some gradio configs
                for key in ("path", "name", "orig_name"):
                    v = f.get(key)
                    if isinstance(v, str) and v:
                        paths.append(v)
                        break
                continue
            # Fallback: objects with a .name attribute
            if hasattr(f, "name"):
                name = getattr(f, "name")
                if isinstance(name, str) and name:
                    paths.append(name)
        return paths

    file_paths = _to_paths(files)
    if not file_paths:
        return "⚠️ No valid files uploaded", ""

    result = pipeline.ingest_documents(file_paths, strategy)

    # Create summary
    if result["status"] == "success":
        summary = f"""✅ **Ingestion Complete**
        
**Documents processed:** {result['documents_ingested']}
**Chunks created:** {result['chunks_created']}
**Processing time:** {result['processing_time']:.2f} seconds

**Chunk Statistics:**
- Average size: {result['chunk_statistics']['avg_chunk_size']:.0f} chars
- Min size: {result['chunk_statistics']['min_chunk_size']} chars
- Max size: {result['chunk_statistics']['max_chunk_size']} chars
"""

        if result.get("failed_files"):
            summary += f"\n⚠️ Failed files: {len(result['failed_files'])}"

        # Return simple success message for now to avoid visualization issues
        return summary, ""
    else:
        return f"❌ Error: {result.get('message', 'Unknown error')}", ""


def process_query(question, top_k, temperature, max_tokens, use_history, session_id):
    """Process user query with enhanced options."""
    if not pipeline:
        return "⚠️ Please initialize the pipeline first!", "", "", ""

    if not question:
        return "⚠️ Please enter a question", "", "", ""

    # Override generation parameters
    pipeline.generator.config.temperature = temperature
    pipeline.generator.config.max_new_tokens = max_tokens

    # Process query
    answer, sources, metadata = pipeline.query(
        question,
        top_k=top_k,
        use_chat_history=use_history,
        session_id=session_id if session_id else "default",
    )

    # Format sources
    sources_md = ""
    for source in sources:
        sources_md += f"""
### Source {source['rank']} (Score: {source['score']:.3f})
**File:** {source['source']} | **Page:** {source['page']}

{source['text']}

---
"""

    # Format metadata
    metadata_md = f"""
**Performance Metrics:**
- Retrieval time: {metadata.get('retrieval_time', 0):.3f}s
- Generation time: {metadata.get('generation_time', 0):.3f}s
- Total time: {metadata.get('total_time', 0):.3f}s
- Tokens used: {metadata.get('tokens_used', 0)}
- Confidence: {metadata.get('confidence', 0):.2%}
- Model: {metadata.get('model', 'unknown')}
"""

    # Skip chart generation for now to avoid type issues
    return answer, sources_md, metadata_md, ""


def create_chunk_visualization(stats):
    """Create visualization for chunk statistics."""
    # Return simple text summary instead of complex matplotlib object
    total = stats.get("total_chunks", 0)
    below_min = stats.get("chunks_below_min", 0)
    above_max = stats.get("chunks_above_max", 0)
    normal = total - below_min - above_max

    summary = f"""📊 Chunk Statistics:
    
• Total Chunks: {total}
• Min Size: {stats.get('min_chunk_size', 0)} chars
• Avg Size: {stats.get('avg_chunk_size', 0)} chars  
• Max Size: {stats.get('max_chunk_size', 0)} chars

📈 Size Distribution:
• Below Min: {below_min} ({(below_min/max(total,1)*100):.1f}%)
• Normal: {normal} ({(normal/max(total,1)*100):.1f}%)
• Above Max: {above_max} ({(above_max/max(total,1)*100):.1f}%)"""

    return summary


def create_performance_chart(metadata):
    """Create performance visualization."""
    # Return simple text summary instead of complex matplotlib object
    retrieval_time = metadata.get("retrieval_time", 0)
    generation_time = metadata.get("generation_time", 0)
    total_time = metadata.get("total_time", 0)
    other_time = max(0, total_time - retrieval_time - generation_time)

    summary = f"""⏱️ Performance Breakdown:
    
• Total Time: {total_time:.3f}s
• Retrieval: {retrieval_time:.3f}s ({(retrieval_time/max(total_time,0.001)*100):.1f}%)
• Generation: {generation_time:.3f}s ({(generation_time/max(total_time,0.001)*100):.1f}%)
• Other: {other_time:.3f}s ({(other_time/max(total_time,0.001)*100):.1f}%)

🎯 Retrieved Docs: {metadata.get('num_retrieved', 0)}
📄 Context Length: {metadata.get('context_length', 0)} chars"""

    return summary


def create_app():
    """Create and configure the Gradio application."""

    with gr.Blocks(
        title="RAG Document QA System",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        """,
    ) as demo:

        gr.Markdown(
            """
        # 📚 RAG Document QA System
        
        Advanced Retrieval-Augmented Generation for intelligent document question answering.
        """
        )

        with gr.Tab("🚀 Setup"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### System Initialization")
                    init_btn = gr.Button("Initialize Pipeline")
                    init_output = gr.Textbox(label="Status", interactive=False)

                    gr.Markdown("### Configuration")
                    with gr.Accordion("Advanced Settings", open=False):
                        embedding_model = gr.Dropdown(
                            choices=[
                                "sentence-transformers/all-MiniLM-L6-v2",
                                "sentence-transformers/all-mpnet-base-v2",
                                "BAAI/bge-small-en-v1.5",
                            ],
                            value="sentence-transformers/all-MiniLM-L6-v2",
                            label="Embedding Model",
                        )
                        llm_backend = gr.Radio(
                            choices=["hf", "openai"], value="hf", label="LLM Backend"
                        )

                with gr.Column(scale=2):
                    gr.Markdown("### Document Upload")
                    file_upload = gr.File(
                        label="Upload Documents",
                        file_types=[".pdf", ".txt", ".html", ".md", ".docx", ".csv"],
                        file_count="multiple",
                        type="filepath",
                    )

                    chunking_strategy = gr.Dropdown(
                        choices=["recursive", "fixed", "sentence", "paragraph"],
                        value="recursive",
                        label="Chunking Strategy",
                    )

                    upload_btn = gr.Button("Index Documents")

                    with gr.Row():
                        upload_output = gr.Textbox(
                            label="Indexing Status", lines=5, interactive=False
                        )
                        chunk_viz = gr.Textbox(
                            label="Chunk Statistics", lines=10, interactive=False
                        )

        with gr.Tab("💬 Query"):
            with gr.Row():
                with gr.Column(scale=2):
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask anything about your documents...",
                        lines=2,
                    )

                    with gr.Row():
                        query_btn = gr.Button("🔍 Search & Generate", variant="primary")
                        clear_btn = gr.Button("🗑️ Clear")

                    with gr.Accordion("⚙️ Advanced Options", open=False):
                        with gr.Row():
                            top_k_slider = gr.Slider(
                                minimum=1,
                                maximum=20,
                                value=5,
                                step=1,
                                label="Number of sources",
                            )
                            temperature_slider = gr.Slider(
                                minimum=0,
                                maximum=1,
                                value=0.7,
                                step=0.1,
                                label="Temperature",
                            )
                        with gr.Row():
                            max_tokens_slider = gr.Slider(
                                minimum=50,
                                maximum=1000,
                                value=300,
                                step=50,
                                label="Max response length",
                            )
                            use_history = gr.Checkbox(
                                label="Use conversation history", value=False
                            )
                        session_id = gr.Textbox(
                            label="Session ID (for conversation tracking)",
                            value="default",
                            visible=False,
                        )

                with gr.Column(scale=3):
                    answer_output = gr.Textbox(
                        label="Answer", lines=8, interactive=False
                    )

                    with gr.Tab("📖 Sources"):
                        sources_output = gr.Markdown()

                    with gr.Tab("📊 Metadata"):
                        metadata_output = gr.Markdown()
                        performance_chart = gr.Textbox(
                            label="Performance Chart", lines=8, interactive=False
                        )

        with gr.Tab("📈 Analytics"):
            gr.Markdown("### System Analytics & Performance")

            with gr.Row():
                refresh_btn = gr.Button("Refresh Statistics")
                export_btn = gr.Button("Export Report")

            stats_output = gr.Textbox(
                label="System Statistics", lines=10, interactive=False
            )

            with gr.Row():
                index_chart = gr.Textbox(
                    label="Index Statistics", lines=6, interactive=False
                )
                performance_metrics = gr.Textbox(
                    label="Performance Trends", lines=6, interactive=False
                )

        with gr.Tab("🧪 Evaluation"):
            gr.Markdown(
                """
            ### Evaluation Suite
            Test your RAG system with predefined or custom evaluation datasets.
            """
            )

            eval_dataset = gr.File(
                label="Upload Evaluation Dataset (JSON)", file_types=[".json"]
            )

            run_eval_btn = gr.Button("Run Evaluation")
            eval_output = gr.Textbox(
                label="Evaluation Report", lines=10, interactive=False
            )
            eval_charts = gr.Textbox(
                label="Evaluation Metrics", lines=8, interactive=False
            )

        with gr.Tab("ℹ️ Help"):
            gr.Markdown(
                """
            ### How to Use
            
            1. **Initialize**: Click "Initialize Pipeline" to load models
            2. **Upload**: Add your documents (PDF, TXT, HTML, etc.)
            3. **Index**: Process documents with your chosen strategy
            4. **Query**: Ask questions about your documents
            5. **Analyze**: Review performance metrics and sources
            
            ### API Endpoints
            
            - `POST /api/ingest` - Upload and index documents
            - `POST /api/query` - Submit queries
            - `GET /api/stats` - Get system statistics
            
            ### Tips
            
            - Use "recursive" chunking for best results
            - Adjust temperature for more/less creative responses
            - Enable conversation history for follow-up questions
            - Check sources to verify answer accuracy
            """
            )

        # Connect event handlers
        init_btn.click(initialize_pipeline, outputs=init_output)

        upload_btn.click(
            upload_and_index,
            inputs=[file_upload, chunking_strategy],
            outputs=[upload_output, chunk_viz],
        )

        query_btn.click(
            process_query,
            inputs=[
                question_input,
                top_k_slider,
                temperature_slider,
                max_tokens_slider,
                use_history,
                session_id,
            ],
            outputs=[answer_output, sources_output, metadata_output, performance_chart],
        )

        clear_btn.click(
            lambda: ("", "", "", ""),
            outputs=[answer_output, sources_output, metadata_output, performance_chart],
        )

        refresh_btn.click(
            lambda: (
                json.dumps(pipeline.get_statistics(), indent=2)
                if pipeline
                else "No statistics available"
            ),
            outputs=stats_output,
        )

    return demo


def main():
    """Main entry point for the application."""
    global app

    # Initialize
    initialize_pipeline()

    # Create Gradio app
    demo = create_app()

    # Mount FastAPI
    demo.queue()
    app = gr.mount_gradio_app(app, demo, path="/")

    # Run with Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")


if __name__ == "__main__":
    main()
