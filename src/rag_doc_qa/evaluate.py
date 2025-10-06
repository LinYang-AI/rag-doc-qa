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
from datetime import datetime

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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "recall_at_k": self.recall_at_k,
            "answer_similarity": self.answer_similarity,
            "latency_ms": self.latency_ms,
            "num_queries": self.num_queries,
            "summary": {
                "avg_recall@5": self.recall_at_k.get(5, 0),
                "avg_similarity": self.answer_similarity
            }
        }

class RAGEvaluator:
    """
    Evaluates RAG system performance.
    """
    
    def __init__(self, similarity_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.similarity_model = SentenceTransformer(similarity_model)
        self.stats = {
            "evaluations_run": 0,
            "total_time": 0
        }
        
    def evaluate(self, 
                 rag_pipeline,
                 eval_dataset: List[Dict[str, Any]],
                 k_values: List[int] = [1, 3, 5, 10]) -> EvaluationResult:
        """
        Evaluate RAG pipeline on a dataset.
        
        Args:
            rag_pipeline: RAG pipeline instance
            eval_dataset: List of evaluation examples
            k_values: Values of k for Recall@k
            
        Returns:
            EvaluationResult object
        """
        start_time = datetime.now()
        
        recall_scores = {k: [] for k in k_values}
        answer_similarities = []
        latencies = []
        detailed_results = []
        
        for i, example in enumerate(eval_dataset):
            logger.info(f"Evaluating example {i+1}/{len(eval_dataset)}")
            
            query = example["query"]
            ground_truth_docs = example.get("relevant_docs", [])
            ground_truth_answer = example.get("answer", "")
            
            # Measure latency
            query_start = time.time()
            
            # Get retrieval results
            retrieved_chunks = rag_pipeline.retriever.retrieve(query, max(k_values))
            
            # Generate answer
            context = rag_pipeline.retriever.get_context(retrieved_chunks[:5])
            generation_result = rag_pipeline.generator.generate(query, context, retrieved_chunks[:5])
            
            latency = (time.time() - query_start) * 1000  # Convert to ms
            latencies.append(latency)
            
            # Calculate Recall@k
            retrieved_doc_ids = [chunk.doc_id if hasattr(chunk, 'doc_id') else chunk.get("doc_id", "") 
                                for chunk in retrieved_chunks]
            
            for k in k_values:
                if ground_truth_docs:
                    relevant_found = sum(
                        1 for doc_id in retrieved_doc_ids[:k]
                        if doc_id in ground_truth_docs
                    )
                    recall = relevant_found / len(ground_truth_docs)
                    recall_scores[k].append(recall)
            
            # Calculate answer similarity
            similarity = 0.0
            if ground_truth_answer:
                similarity = self._calculate_answer_similarity(
                    generation_result.text, 
                    ground_truth_answer
                )
                answer_similarities.append(similarity)
            
            # Store detailed result
            detailed_results.append({
                "query": query,
                "generated_answer": generation_result.text,
                "ground_truth_answer": ground_truth_answer,
                "retrieved_docs": retrieved_doc_ids[:5],
                "ground_truth_docs": ground_truth_docs,
                "latency_ms": latency,
                "answer_similarity": similarity,
                "recall_scores": {k: recall_scores[k][-1] if recall_scores[k] else 0 
                                 for k in k_values}
            })
        
        # Calculate averages
        avg_recall = {k: np.mean(scores) if scores else 0 
                     for k, scores in recall_scores.items()}
        avg_similarity = np.mean(answer_similarities) if answer_similarities else 0
        avg_latency = np.mean(latencies) if latencies else 0
        
        # Update stats
        elapsed = (datetime.now() - start_time).total_seconds()
        self.stats["evaluations_run"] += 1
        self.stats["total_time"] += elapsed
        
        return EvaluationResult(
            recall_at_k=avg_recall,
            answer_similarity=avg_similarity,
            latency_ms=avg_latency,
            num_queries=len(eval_dataset),
            detailed_results=detailed_results
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
    
    def calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate additional metrics from results.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary of metrics
        """
        if not results:
            return {}
        
        # Extract metrics
        similarities = [r["answer_similarity"] for r in results if r.get("answer_similarity") is not None]
        latencies = [r["latency_ms"] for r in results if r.get("latency_ms") is not None]
        
        metrics = {
            "mean_similarity": np.mean(similarities) if similarities else 0,
            "std_similarity": np.std(similarities) if similarities else 0,
            "min_similarity": np.min(similarities) if similarities else 0,
            "max_similarity": np.max(similarities) if similarities else 0,
            "mean_latency": np.mean(latencies) if latencies else 0,
            "p50_latency": np.percentile(latencies, 50) if latencies else 0,
            "p95_latency": np.percentile(latencies, 95) if latencies else 0,
            "p99_latency": np.percentile(latencies, 99) if latencies else 0
        }
        
        return metrics
    
    def generate_report(self, results: EvaluationResult, output_path: Optional[Path] = None) -> str:
        """
        Generate markdown evaluation report.
        
        Args:
            results: EvaluationResult object
            output_path: Optional path to save report
            
        Returns:
            Markdown formatted report
        """
        report = ["# RAG System Evaluation Report\n"]
        report.append(f"**Generated:** {datetime.now().isoformat()}\n")
        report.append(f"**Number of queries:** {results.num_queries}\n")
        report.append(f"**Average latency:** {results.latency_ms:.2f}ms\n")
        
        report.append("\n## 📊 Retrieval Performance (Recall@k)\n")
        report.append("| k | Recall@k | Performance |\n")
        report.append("|---|----------|-------------|\n")
        for k, recall in sorted(results.recall_at_k.items()):
            perf = "🟢 Excellent" if recall > 0.8 else "🟡 Good" if recall > 0.6 else "🔴 Needs Improvement"
            report.append(f"| {k} | {recall:.3f} | {perf} |\n")
        
        report.append(f"\n## 💬 Answer Quality\n")
        report.append(f"**Average Answer Similarity:** {results.answer_similarity:.3f}\n")
        
        quality = "Excellent" if results.answer_similarity > 0.8 else "Good" if results.answer_similarity > 0.6 else "Fair"
        report.append(f"**Overall Quality:** {quality}\n")
        
        # Calculate additional metrics
        additional_metrics = self.calculate_metrics(results.detailed_results)
        
        report.append("\n## ⚡ Performance Metrics\n")
        report.append("| Metric | Value |\n")
        report.append("|--------|-------|\n")
        report.append(f"| Mean Latency | {additional_metrics.get('mean_latency', 0):.2f}ms |\n")
        report.append(f"| P50 Latency | {additional_metrics.get('p50_latency', 0):.2f}ms |\n")
        report.append(f"| P95 Latency | {additional_metrics.get('p95_latency', 0):.2f}ms |\n")
        report.append(f"| P99 Latency | {additional_metrics.get('p99_latency', 0):.2f}ms |\n")
        
        report.append("\n## 📝 Sample Results\n")
        for i, result in enumerate(results.detailed_results[:3], 1):
            report.append(f"\n### Query {i}\n")
            report.append(f"**Question:** {result['query']}\n\n")
            report.append(f"**Generated Answer:** {result['generated_answer'][:500]}...\n\n")
            if result.get('ground_truth_answer'):
                report.append(f"**Expected Answer:** {result['ground_truth_answer'][:500]}...\n\n")
                report.append(f"**Similarity Score:** {result.get('answer_similarity', 0):.3f}\n")
            report.append(f"**Latency:** {result['latency_ms']:.2f}ms\n")
            report.append(f"**Recall@5:** {result['recall_scores'].get(5, 0):.3f}\n")
        
        report.append("\n## 🎯 Recommendations\n")
        if results.answer_similarity < 0.6:
            report.append("- ⚠️ Consider fine-tuning the generation model\n")
        if results.recall_at_k.get(5, 0) < 0.7:
            report.append("- ⚠️ Improve retrieval by adjusting chunk size or embedding model\n")
        if results.latency_ms > 1000:
            report.append("- ⚠️ Optimize for latency with caching or smaller models\n")
        
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
            "answer": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."
        },
        {
            "query": "Explain neural networks",
            "relevant_docs": ["doc_003"],
            "answer": "Neural networks are computing systems inspired by biological neural networks that constitute animal brains."
        },
        {
            "query": "What is deep learning?",
            "relevant_docs": ["doc_002", "doc_004"],
            "answer": "Deep learning is a subset of machine learning using neural networks with multiple layers."
        }
    ]

def main():
    """CLI for evaluation."""
    import argparse
    from .web_app import RAGPipeline
    
    parser = argparse.ArgumentParser(description="Evaluate RAG system")
    parser.add_argument("--dataset", default="examples/eval_dataset.json", help="Evaluation dataset path")
    parser.add_argument("--output", default="evaluation_report.md", help="Output report path")
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5, 10], help="K values for Recall@k")
    
    args = parser.parse_args()
    
    # Load dataset
    with open(args.dataset, "r") as f:
        eval_dataset = json.load(f)
    
    # Initialize pipeline
    pipeline = RAGPipeline()
    
    # Run evaluation
    evaluator = RAGEvaluator()
    results = evaluator.evaluate(pipeline, eval_dataset, args.k_values)
    
    # Generate report
    report = evaluator.generate_report(results, Path(args.output))
    print(report)
    print(f"\nReport saved to: {args.output}")

if __name__ == "__main__":
    main()