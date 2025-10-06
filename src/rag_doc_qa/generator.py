"""
Advanced LLM generation module with multiple backend support.
Includes streaming, token management, and response optimization.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Generator, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import hashlib

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList
)

from rag_doc_qa.config import GeneratorConfig, QA_PROMPT_TEMPLATE, SYSTEM_PROMPT, CHAT_PROMPT_TEMPLATE
from rag_doc_qa.retriever import RetrievalResult

logger = logging.getLogger(__name__)

@dataclass
class GenerationResult:
    """Enhanced generation result with metadata."""
    text: str
    sources: List[Dict[str, Any]]
    prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    tokens_used: int = 0
    generation_time: float = 0.0
    confidence_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "sources": self.sources,
            "prompt": self.prompt[:200] + "..." if len(self.prompt) > 200 else self.prompt,
            "metadata": self.metadata,
            "tokens_used": self.tokens_used,
            "generation_time": self.generation_time,
            "confidence_score": self.confidence_score
        }

class StopOnTokens(StoppingCriteria):
    """Custom stopping criteria for generation."""
    
    def __init__(self, stop_token_ids: List[int]):
        self.stop_token_ids = stop_token_ids
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

class LLMGenerator:
    """
    Advanced LLM generator with multiple backend support and optimizations.
    """
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        self.openai_client = None
        self.model_type = None  # 'seq2seq' or 'causal'
        
        # Statistics
        self.stats = {
            "total_generations": 0,
            "total_tokens": 0,
            "avg_generation_time": 0.0,
            "errors": 0
        }
        
        # Response cache
        self.response_cache = {}
        
        # Initialize backend
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the appropriate generation backend."""
        if self.config.backend == "hf":
            self._initialize_hf_model()
        elif self.config.backend == "openai":
            self._initialize_openai()
        else:
            logger.warning(f"Unknown backend: {self.config.backend}")
            self._initialize_mock()
    
    def _initialize_hf_model(self):
        """Initialize Hugging Face model with optimizations."""
        try:
            model_name = self.config.hf_model_name
            logger.info(f"Loading HF model: {model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Determine model type and load accordingly
            try:
                # Try loading as Seq2Seq model (T5, BART, etc.)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                    device_map="auto" if self.config.device == "cuda" else None,
                    load_in_8bit=self.config.use_8bit and self.config.device == "cuda"
                )
                self.model_type = "seq2seq"
                logger.info("Loaded Seq2Seq model")
            except Exception:
                # Fall back to Causal LM (GPT, Llama, etc.)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                    device_map="auto" if self.config.device == "cuda" else None,
                    load_in_8bit=self.config.use_8bit and self.config.device == "cuda"
                )
                self.model_type = "causal"
                logger.info("Loaded Causal LM")
            
            # Move to device if needed
            if self.config.device == "cpu":
                self.model = self.model.to("cpu")
            
            # Set up generation config
            self.generation_config = GenerationConfig(
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                repetition_penalty=self.config.repetition_penalty,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Optimize for inference
            self.model.eval()
            if hasattr(torch, 'compile') and self.config.device == "cuda":
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    logger.info("Model compiled for faster inference")
                except Exception as e:
                    logger.warning(f"Could not compile model: {e}")
            
            logger.info(f"Model loaded on {self.config.device}")
            
        except Exception as e:
            logger.error(f"Error loading HF model: {e}")
            self._initialize_mock()
    
    def _initialize_openai(self):
        """Initialize OpenAI client."""
        try:
            import openai
            from openai import OpenAI
            
            if not self.config.openai_api_key:
                raise ValueError("OpenAI API key not provided")
            
            self.openai_client = OpenAI(api_key=self.config.openai_api_key)
            logger.info(f"OpenAI client initialized with model: {self.config.openai_model}")
            
        except ImportError:
            logger.error("OpenAI package not installed. Install with: pip install openai")
            self._initialize_mock()
        except Exception as e:
            logger.error(f"Error initializing OpenAI: {e}")
            self._initialize_mock()
    
    def _initialize_mock(self):
        """Initialize mock generator for testing."""
        logger.info("Using mock generator (no LLM)")
        self.model = None
        self.tokenizer = None
    
    def generate(self,
                query: str,
                context: str,
                sources: List[Union[Dict[str, Any], RetrievalResult]],
                chat_history: Optional[str] = None,
                system_prompt: Optional[str] = None,
                **kwargs) -> GenerationResult:
        """
        Generate answer based on query and context.
        
        Args:
            query: User question
            context: Retrieved context
            sources: Source documents/chunks
            chat_history: Optional conversation history
            system_prompt: Optional custom system prompt
            **kwargs: Additional generation parameters
            
        Returns:
            GenerationResult with answer and metadata
        """
        start_time = time.time()
        
        # Format sources
        formatted_sources = self._format_sources(sources)
        
        # Choose prompt template
        if chat_history:
            prompt = CHAT_PROMPT_TEMPLATE.format(
                chat_history=chat_history,
                context=context,
                question=query
            )
        else:
            prompt = QA_PROMPT_TEMPLATE.format(
                context=context,
                question=query
            )
        
        # Add system prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
        
        # Check cache
        cache_key = hashlib.md5(full_prompt.encode()).hexdigest()
        if cache_key in self.response_cache:
            cached_result = self.response_cache[cache_key]
            logger.info("Using cached response")
            return cached_result
        
        # Truncate if needed
        full_prompt = self._truncate_prompt(full_prompt, query)
        
        # Generate based on backend
        try:
            if self.config.backend == "openai" and self.openai_client:
                answer = self._generate_openai(full_prompt, **kwargs)
                tokens_used = self._estimate_tokens(full_prompt + answer)
            elif self.config.backend == "hf" and self.model:
                answer, tokens_used = self._generate_hf(full_prompt, **kwargs)
            else:
                answer = self._generate_mock(query, context, formatted_sources)
                tokens_used = 0
            
            # Calculate confidence (simplified)
            confidence = self._calculate_confidence(answer, context)
            
            # Create result
            result = GenerationResult(
                text=answer,
                sources=formatted_sources,
                prompt=full_prompt,
                metadata={
                    "backend": self.config.backend,
                    "model": self._get_model_name(),
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "max_tokens": kwargs.get("max_tokens", self.config.max_new_tokens)
                },
                tokens_used=tokens_used,
                generation_time=time.time() - start_time,
                confidence_score=confidence
            )
            
            # Cache result
            self.response_cache[cache_key] = result
            
            # Update statistics
            self.stats["total_generations"] += 1
            self.stats["total_tokens"] += tokens_used
            self.stats["avg_generation_time"] = (
                (self.stats["avg_generation_time"] * (self.stats["total_generations"] - 1) +
                 result.generation_time) / self.stats["total_generations"]
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            self.stats["errors"] += 1
            
            return GenerationResult(
                text=f"Error generating response: {str(e)}",
                sources=formatted_sources,
                prompt=full_prompt,
                metadata={"error": str(e)},
                generation_time=time.time() - start_time
            )
    
    def _generate_hf(self, prompt: str, **kwargs) -> Tuple[str, int]:
        """Generate using Hugging Face model."""
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_context_length
        )
        
        # Move to device
        if self.config.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Update generation config with kwargs
        gen_config = GenerationConfig.from_model_config(self.model.config)
        gen_config.update(**{
            "max_new_tokens": kwargs.get("max_tokens", self.config.max_new_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "do_sample": kwargs.get("do_sample", self.config.do_sample)
        })
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=gen_config,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        if self.model_type == "seq2seq":
            # For seq2seq, output doesn't include input
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            # For causal LM, remove input from output
            input_length = inputs["input_ids"].shape[1]
            response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        # Count tokens
        tokens_used = len(outputs[0])
        
        return response.strip(), tokens_used
    
    def _generate_openai(self, prompt: str, **kwargs) -> str:
        """Generate using OpenAI API."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        response = self.openai_client.chat.completions.create(
            model=kwargs.get("model", self.config.openai_model),
            messages=messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_new_tokens),
            top_p=kwargs.get("top_p", self.config.top_p)
        )
        
        return response.choices[0].message.content
    
    def _generate_mock(self, query: str, context: str, sources: List[Dict]) -> str:
        """Mock generator for testing."""
        if not context:
            return "I couldn't find relevant information to answer your question."
        
        # Extract key sentences from context
        sentences = context.split(". ")[:3]
        answer = f"Based on the provided context: {'. '.join(sentences)}"
        
        if sources:
            source_info = sources[0]
            answer += f"\n\n[Source: {source_info.get('source', 'document')}]"
        
        return answer
    
    def stream_generate(self,
                       query: str,
                       context: str,
                       sources: List[Union[Dict[str, Any], RetrievalResult]],
                       **kwargs) -> Generator[str, None, None]:
        """
        Stream generation for real-time response.
        
        Args:
            query: User question
            context: Retrieved context
            sources: Source documents
            **kwargs: Generation parameters
            
        Yields:
            Generated text chunks
        """
        # For now, simulate streaming by yielding words
        # Full implementation would use model.generate() with streaming
        result = self.generate(query, context, sources, **kwargs)
        
        words = result.text.split()
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            time.sleep(0.02)  # Simulate typing delay
    
    async def generate_async(self,
                            query: str,
                            context: str,
                            sources: List[Union[Dict[str, Any], RetrievalResult]],
                            **kwargs) -> GenerationResult:
        """
        Asynchronous generation for better concurrency.
        
        Args:
            query: User question
            context: Retrieved context
            sources: Source documents
            **kwargs: Generation parameters
            
        Returns:
            GenerationResult
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.generate,
            query,
            context,
            sources,
            None,  # chat_history
            None,  # system_prompt
            kwargs
        )
    
    def _format_sources(self, sources: List[Union[Dict[str, Any], RetrievalResult]]) -> List[Dict]:
        """Format sources for output."""
        formatted = []
        
        for source in sources[:5]:  # Limit to top 5 sources
            if isinstance(source, RetrievalResult):
                formatted.append({
                    "text": source.text[:200] + "..." if len(source.text) > 200 else source.text,
                    "source": source.metadata.get("source", "Unknown"),
                    "page": source.metadata.get("page", "N/A"),
                    "score": source.final_score
                })
            else:
                formatted.append({
                    "text": source.get("text", "")[:200] + "...",
                    "source": source.get("source", "Unknown"),
                    "page": source.get("page", "N/A"),
                    "score": source.get("score", 0.0)
                })
        
        return formatted
    
    def _truncate_prompt(self, prompt: str, query: str) -> str:
        """Truncate prompt to fit context window."""
        if self.tokenizer:
            # Use tokenizer to count tokens accurately
            tokens = self.tokenizer.encode(prompt)
            if len(tokens) > self.config.max_context_length:
                # Keep query intact, truncate context
                query_tokens = self.tokenizer.encode(query)
                available = self.config.max_context_length - len(query_tokens) - 100  # Buffer
                
                # Truncate and decode
                truncated_tokens = tokens[:available]
                prompt = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                prompt += f"\n\nQuestion: {query}\n\nAnswer:"
        else:
            # Simple character-based truncation
            if len(prompt) > self.config.max_context_length * 4:  # Rough estimate
                available = self.config.max_context_length * 4 - len(query) - 200
                prompt = prompt[:available] + f"...\n\nQuestion: {query}\n\nAnswer:"
        
        return prompt
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough estimation: ~4 characters per token
            return len(text) // 4
    
    def _calculate_confidence(self, answer: str, context: str) -> float:
        """Calculate confidence score for generated answer."""
        if not answer or not context:
            return 0.0
        
        # Simple heuristic based on answer length and context overlap
        answer_lower = answer.lower()
        context_lower = context.lower()
        
        # Check for hallucination indicators
        if "i don't know" in answer_lower or "not found" in answer_lower:
            return 0.3
        
        # Calculate word overlap
        answer_words = set(answer_lower.split())
        context_words = set(context_lower.split())
        
        overlap = len(answer_words & context_words)
        total = len(answer_words)
        
        if total == 0:
            return 0.5
        
        overlap_ratio = overlap / total
        
        # Combine with answer length signal
        length_score = min(1.0, len(answer) / 500)  # Prefer substantial answers
        
        confidence = (overlap_ratio * 0.7 + length_score * 0.3)
        
        return min(1.0, confidence)
    
    def _get_model_name(self) -> str:
        """Get the name of the current model."""
        if self.config.backend == "hf":
            return self.config.hf_model_name
        elif self.config.backend == "openai":
            return self.config.openai_model
        else:
            return "mock"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        return {
            **self.stats,
            "backend": self.config.backend,
            "model": self._get_model_name(),
            "device": self.config.device,
            "cache_size": len(self.response_cache),
            "avg_tokens_per_generation": (
                self.stats["total_tokens"] / max(1, self.stats["total_generations"])
            )
        }
    
    def clear_cache(self):
        """Clear response cache."""
        self.response_cache.clear()
        logger.info("Response cache cleared")
    
    def benchmark(self, num_queries: int = 10) -> Dict[str, float]:
        """
        Benchmark generation performance.
        
        Args:
            num_queries: Number of test queries
            
        Returns:
            Benchmark metrics
        """
        import random
        
        test_queries = [
            "What is machine learning?",
            "Explain neural networks",
            "How does deep learning work?",
            "What are transformers in AI?",
            "Describe reinforcement learning"
        ]
        
        test_context = "Machine learning is a method of data analysis. " * 50
        
        total_time = 0
        total_tokens = 0
        
        for _ in range(num_queries):
            query = random.choice(test_queries)
            start = time.time()
            
            result = self.generate(
                query,
                test_context,
                [],
                use_cache=False
            )
            
            total_time += result.generation_time
            total_tokens += result.tokens_used
        
        return {
            "num_queries": num_queries,
            "total_time": total_time,
            "avg_time_per_query": total_time / num_queries,
            "total_tokens": total_tokens,
            "avg_tokens_per_query": total_tokens / num_queries,
            "queries_per_second": num_queries / total_time if total_time > 0 else 0
        }