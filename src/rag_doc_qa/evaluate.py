"""
Evaluation module for RAG system.
Implements Recall@k and answer similarity metrics.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
import time

from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""

    recall_at_k: Dict[int, float]
    answer_similarity: float
    latency_ms: float
    num_queries: int
    detailed_results: List[Dict[str, Any]]


class RAGEvaluator:
    """
    Evaluates RAG system performance.
    """

    def __init__(
        self, similarity_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.similarity_model = SentenceTransformer(similarity_model)

    def evaluate(
        self,
        rag_pipeline,
        eval_dataset: List[Dict[str, Any]],
        k_values: List[int] = [1, 3, 5, 10],
    ) -> EvaluationResult:
        """
        Evaluate RAG pipeline on a dataset.

        Args:
            rag_pipeline: RAG pipeline instance
            eval_dataset: List of evaluation examples
            k_values: Values of k for Recall@k

        Returns:
            EvaluationResult object
        """
        recall_scores = {k: [] for k in k_values}
        answer_similarities = []
        latencies = []
        detailed_results = []

        for example in eval_dataset:
            query = example["query"]
            ground_truth_docs = example.get("relevant_docs", [])
            ground_truth_answer = example.get("answer", "")

            # Measure latency
            start_time = time.time()

            # Get retrieval results
            retrieved_chunks = rag_pipeline.retriever.retrieve(query, max(k_values))

            # Generate answer
            context = rag_pipeline.retriever.get_context(retrieved_chunks[:5])
            generation_result = rag_pipeline.generator.generate(
                query, context, retrieved_chunks[:5]
            )

            latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(latency)

            # Calculate Recall@k
            retrieved_doc_ids = [chunk["doc_id"] for chunk in retrieved_chunks]

            for k in k_values:
                if ground_truth_docs:
                    relevant_found = sum(
                        1
                        for doc_id in retrieved_doc_ids[:k]
                        if doc_id in ground_truth_docs
                    )
                    recall = relevant_found / len(ground_truth_docs)
                    recall_scores[k].append(recall)

            # Calculate answer similarity
            if ground_truth_answer:
                similarity = self._calculate_answer_similarity(
                    generation_result.text, ground_truth_answer
                )
                answer_similarities.append(similarity)

            # Store detailed result
            detailed_results.append(
                {
                    "query": query,
                    "generated_answer": generation_result.text,
                    "ground_truth_answer": ground_truth_answer,
                    "retrieved_docs": retrieved_doc_ids[:5],
                    "ground_truth_docs": ground_truth_docs,
                    "latency_ms": latency,
                    "answer_similarity": similarity if ground_truth_answer else None,
                }
            )

        # Calculate averages
        avg_recall = {
            k: np.mean(scores) if scores else 0 for k, scores in recall_scores.items()
        }
        avg_similarity = np.mean(answer_similarities) if answer_similarities else 0
        avg_latency = np.mean(latencies) if latencies else 0

        return EvaluationResult(
            recall_at_k=avg_recall,
            answer_similarity=avg_similarity,
            latency_ms=avg_latency,
            num_queries=len(eval_dataset),
            detailed_results=detailed_results,
        )

    def _calculate_answer_similarity(self, generated: str, ground_truth: str) -> float:
        """
        Calculate semantic similarity between generated and ground truth answers.

        Args:
            generated: Generated answer
            ground_truth: Ground truth answer

        Returns:
            Similarity score between 0 and 1
        """
        # Encode both answers
        embeddings = self.similarity_model.encode([generated, ground_truth])

        # Calculate cosine similarity
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()

        # Ensure score is between 0 and 1
        return max(0, min(1, similarity))

    def generate_report(
        self, results: EvaluationResult, output_path: Optional[Path] = None
    ) -> str:
        """
        Generate markdown evaluation report.

        Args:
            results: EvaluationResult object
            output_path: Optional path to save report

        Returns:
            Markdown formatted report
        """
        report = ["# RAG System Evaluation Report\n"]
        report.append(f"**Number of queries evaluated:** {results.num_queries}\n")
        report.append(f"**Average latency:** {results.latency_ms:.2f}ms\n")

        report.append("\n## Retrieval Performance (Recall@k)\n")
        report.append("| k | Recall@k |\n")
        report.append("|---|----------|\n")
        for k, recall in sorted(results.recall_at_k.items()):
            report.append(f"| {k} | {recall:.3f} |\n")

        report.append(f"\n## Answer Quality\n")
        report.append(
            f"**Average Answer Similarity:** {results.answer_similarity:.3f}\n"
        )

        report.append("\n## Sample Results\n")
        for i, result in enumerate(results.detailed_results[:3], 1):
            report.append(f"\n### Query {i}\n")
            report.append(f"**Question:** {result['query']}\n\n")
            report.append(f"**Generated Answer:** {result['generated_answer']}\n\n")
            if result.get("ground_truth_answer"):
                report.append(
                    f"**Expected Answer:** {result['ground_truth_answer']}\n\n"
                )
                report.append(
                    f"**Similarity Score:** {result.get('answer_similarity', 0):.3f}\n"
                )
            report.append(f"**Latency:** {result['latency_ms']:.2f}ms\n")

        report_text = "".join(report)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_path}")

        return report_text


def create_sample_eval_dataset() -> List[Dict[str, Any]]:
    """
    Create a sample evaluation dataset for testing.
    """
    return [
        {
            "query": "What is machine learning?",
            "relevant_docs": ["doc_001", "doc_002"],
            "answer": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
        },
        {
            "query": "Explain neural networks",
            "relevant_docs": ["doc_003"],
            "answer": "Neural networks are computing systems inspired by biological neural networks that constitute animal brains, consisting of interconnected nodes or neurons.",
        },
        {
            "query": "What is deep learning?",
            "relevant_docs": ["doc_002", "doc_004"],
            "answer": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to progressively extract higher-level features from raw input.",
        },
    ]
