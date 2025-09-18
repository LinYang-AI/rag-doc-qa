"""
Gradio web interface for RAG Document QA system.
"""

import gradio as gr
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import json
import time

from rag_doc_qa.ingest import DocumentIngestor
from rag_doc_qa.splitter import TextSplitter
from rag_doc_qa.embeddings import EmbeddingModel
from rag_doc_qa.indexer import FAISSIndex
from rag_doc_qa.retriever import HybridRetriever
from rag_doc_qa.generator import LLMGenerator
from rag_doc_qa.config import SystemConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """Complete RAG pipeline orchestration."""

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()

        # Initialize components
        self.ingestor = DocumentIngestor()
        self.splitter = TextSplitter(self.config.chunking)
        self.embedding_model = EmbeddingModel(self.config.embedding)
        self.index = FAISSIndex(self.config.index, self.embedding_model)
        self.retriever = HybridRetriever(self.config.retriever, self.index)
        self.generator = LLMGenerator(self.config.generator)

        # Load existing index if available
        if self.index.index_exists():
            self.index.load()
            self.retriever._initialize_bm25()

    def ingest_documents(self, file_paths: List[str]) -> str:
        """Ingest documents and update index."""
        try:
            all_docs = []
            for file_path in file_paths:
                docs = self.ingestor.ingest_file(Path(file_path))
                all_docs.extend(docs)

            # Split into chunks
            chunks = self.splitter.split_documents(all_docs)

            # Index chunks
            self.index.create_index(chunks)

            # Reinitialize BM25
            self.retriever._initialize_bm25()

            return f"Successfully ingested {len(all_docs)} documents, created {len(chunks)} chunks"
        except Exception as e:
            logger.error(f"Error ingesting documents: {e}")
            return f"Error: {e}"

    def query(self, question: str, top_k: int = 5) -> Tuple[str, List[dict]]:
        """Process a query and return answer with sources."""
        try:
            # Retrieve relevant chunks
            retrieved_chunks = self.retriever.retrieve(question, top_k)

            if not retrieved_chunks:
                return "No relevant information found in the documents.", []

            # Format context
            context = self.retriever.get_context(retrieved_chunks)

            # Generate answer
            result = self.generator.generate(question, context, retrieved_chunks)

            # Format sources
            sources = []
            for chunk in retrieved_chunks[:3]:
                sources.append(
                    {
                        "text": chunk["text"][:200] + "...",
                        "source": chunk["metadata"].get("source", "Unknown"),
                        "page": chunk["metadata"].get("page", "N/A"),
                        "score": chunk.get("combined_score", 0),
                    }
                )

            return result.text, sources
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Error: {e}", []


# Global pipeline instance
pipeline = None


def initialize_pipeline():
    """Initialize the RAG pipeline."""
    global pipeline
    if pipeline is None:
        pipeline = RAGPipeline()
    return "Pipeline initialized successfully!"


def upload_and_index(files):
    """Handle file upload and indexing."""
    if pipeline is None:
        return "Please initialize the pipeline first!"

    if not files:
        return "No files uploaded"

    file_paths = [f.name for f in files]
    return pipeline.ingest_documents(file_paths)


def process_query(
    question: str, top_k: int, temperature: float, max_tokens: int, system_prompt: str
):
    """Process user query."""
    if pipeline is None:
        return "Please initialize the pipeline first!", []

    if not question:
        return "Please enter a question", []

    # Override system prompt if provided
    if system_prompt:
        pipeline.generator.config.system_prompt = system_prompt

    # Set generation parameters
    answer, sources = pipeline.query(question, top_k=top_k)

    # Format sources for display
    sources_display = []
    for i, source in enumerate(sources, 1):
        sources_display.append(
            f"**Source {i}** (Score: {source['score']:.3f})\n"
            f"File: {source['source']}, Page: {source['page']}\n"
            f"Preview: {source['text']}\n"
        )

    return answer, "\n---\n".join(sources_display) if sources_display else "No sources"


def get_index_stats():
    """Get current index statistics."""
    if pipeline is None or pipeline.index is None:
        return "Pipeline not initialized"

    stats = pipeline.index.get_stats()
    return json.dumps(stats, indent=2)


# Create Gradio interface
def create_app():
    """Create and configure Gradio app."""

    with gr.Blocks(title="RAG Document QA System") as app:
        gr.Markdown(
            """
        # 📚 RAG Document QA System
        
        Upload documents and ask questions to get AI-powered answers with source citations.
        """
        )

        with gr.Tab("Setup"):
            gr.Markdown("### Initialize System")
            init_btn = gr.Button("Initialize Pipeline", variant="primary")
            init_output = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("### Upload Documents")
            file_upload = gr.File(
                label="Upload Documents (PDF, TXT, HTML)",
                file_count="multiple",
                file_types=[".pdf", ".txt", ".html", ".md"],
            )
            upload_btn = gr.Button("Index Documents")
            upload_output = gr.Textbox(label="Indexing Status", interactive=False)

            gr.Markdown("### Index Statistics")
            stats_btn = gr.Button("Get Stats")
            stats_output = gr.Textbox(label="Index Stats", interactive=False, lines=8)

        with gr.Tab("Query"):
            gr.Markdown("### Ask Questions")

            with gr.Row():
                with gr.Column(scale=2):
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="What would you like to know?",
                        lines=2,
                    )

                    with gr.Accordion("Advanced Settings", open=False):
                        top_k_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Number of sources to retrieve",
                        )
                        temperature_slider = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0.7,
                            step=0.1,
                            label="Temperature (creativity)",
                        )
                        max_tokens_slider = gr.Slider(
                            minimum=50,
                            maximum=500,
                            value=300,
                            step=50,
                            label="Max response length",
                        )
                        system_prompt_input = gr.Textbox(
                            label="System Prompt (optional)",
                            placeholder="Override default system prompt...",
                            lines=3,
                        )

                    query_btn = gr.Button("Get Answer", variant="primary")

                with gr.Column(scale=3):
                    answer_output = gr.Textbox(
                        label="Answer", lines=8, interactive=False
                    )
                    sources_output = gr.Markdown(label="Sources")

        with gr.Tab("API Examples"):
            gr.Markdown(
                """
            ### cURL Examples
            
            ```bash
            # Query endpoint
            curl -X POST http://localhost:7860/api/query \\
                -H "Content-Type: application/json" \\
                -d '{"question": "What is RAG?", "top_k": 5}'
            
            # Upload documents
            curl -X POST http://localhost:7860/api/upload \\
                -F "file=@document.pdf"
            ```
            
            ### Python SDK
            
            ```python
            from rag_doc_qa import RAGPipeline
            
            # Initialize
            rag = RAGPipeline()
            
            # Ingest documents
            rag.ingest_documents(["document.pdf"])
            
            # Query
            answer, sources = rag.query("Your question here")
            print(answer)
            ```
            """
            )

        # Connect handlers
        init_btn.click(initialize_pipeline, outputs=init_output)
        upload_btn.click(upload_and_index, inputs=file_upload, outputs=upload_output)
        stats_btn.click(get_index_stats, outputs=stats_output)
        query_btn.click(
            process_query,
            inputs=[
                question_input,
                top_k_slider,
                temperature_slider,
                max_tokens_slider,
                system_prompt_input,
            ],
            outputs=[answer_output, sources_output],
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
