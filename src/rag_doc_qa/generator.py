"""
LLM generation module for RAG system.
Supports both local Hugging Face models and OpenAI API.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

from .config import GeneratorConfig, QA_PROMPT_TEMPLATE, SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result from LLM generation."""

    text: str
    sources: List[Dict[str, Any]]
    prompt: str
    metadata: Dict[str, Any]


class LLMGenerator:
    """
    Handles LLM-based answer generation with multiple backend support.
    """

    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        self.model = None
        self.tokenizer = None
        self.openai_client = None

        # Initialize based on backend
        if self.config.backend == "hf":
            self._initialize_hf_model()
        elif self.config.backend == "openai":
            self._initialize_openai()
        else:
            logger.warning(
                f"Unknown backend: {self.config.backend}. Using stub implementation."
            )

    def _initialize_hf_model(self):
        """Initialize Hugging Face model."""
        try:
            logger.info(f"Loading HF model: {self.config.model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

            # Try Seq2Seq model first (like T5)
            try:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=(
                        torch.float16 if self.config.device == "cuda" else torch.float32
                    ),
                    device_map="auto" if self.config.device == "cuda" else None,
                )
            except Exception:
                # Fall back to causal LM
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=(
                        torch.float16 if self.config.device == "cuda" else torch.float32
                    ),
                    device_map="auto" if self.config.device == "cuda" else None,
                )

            if self.config.device == "cpu":
                self.model = self.model.to("cpu")

            logger.info(f"Model loaded successfully on {self.config.device}")

        except Exception as e:
            logger.error(f"Error loading HF model: {e}")
            logger.info("Using stub generator. Set OPENAI_API_KEY for OpenAI backend.")
            self.model = None

    def _initialize_openai(self):
        """Initialize OpenAI client."""
        try:
            import openai
            from openai import OpenAI

            if not self.config.openai_api_key:
                raise ValueError("OpenAI API key not provided")

            self.openai_client = OpenAI(api_key=self.config.openai_api_key)
            logger.info("OpenAI client initialized")

        except ImportError:
            logger.error(
                "OpenAI package not installed. Install with: pip install openai"
            )
            self.openai_client = None
        except Exception as e:
            logger.error(f"Error initializing OpenAI: {e}")
            self.openai_client = None

    def generate(
        self, query: str, context: str, sources: List[Dict[str, Any]], **kwargs
    ) -> GenerationResult:
        """
        Generate answer based on query and context.

        Args:
            query: User question
            context: Retrieved context
            sources: Source documents metadata
            **kwargs: Additional generation parameters

        Returns:
            GenerationResult with answer and metadata
        """
        # Format prompt
        prompt = QA_PROMPT_TEMPLATE.format(context=context, question=query)

        # Truncate if needed
        if len(prompt) > self.config.max_context_length:
            # Truncate context, keeping query intact
            available_context_length = self.config.max_context_length - len(query) - 200
            context = context[:available_context_length] + "..."
            prompt = QA_PROMPT_TEMPLATE.format(context=context, question=query)

        # Generate based on backend
        if self.config.backend == "openai" and self.openai_client:
            answer = self._generate_openai(prompt, **kwargs)
        elif self.config.backend == "hf" and self.model:
            answer = self._generate_hf(prompt, **kwargs)
        else:
            answer = self._generate_stub(query, context, sources)

        return GenerationResult(
            text=answer,
            sources=sources[:3],  # Top 3 sources
            prompt=prompt,
            metadata={
                "backend": self.config.backend,
                "model": (
                    self.config.model_name
                    if self.config.backend == "hf"
                    else self.config.openai_model
                ),
                "temperature": kwargs.get("temperature", self.config.temperature),
                "max_tokens": kwargs.get("max_tokens", self.config.max_new_tokens),
            },
        )

    def _generate_hf(self, prompt: str, **kwargs) -> str:
        """Generate using Hugging Face model."""
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_context_length,
            )

            if self.config.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get("max_tokens", self.config.max_new_tokens),
                    temperature=kwargs.get("temperature", self.config.temperature),
                    top_p=kwargs.get("top_p", self.config.top_p),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode, removing input prompt if present
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # For causal models, remove the prompt from response
            if (
                hasattr(self.model, "config")
                and self.model.config.is_encoder_decoder == False
            ):
                response = response[len(prompt) :].strip()

            return response

        except Exception as e:
            logger.error(f"Error generating with HF model: {e}")
            return f"Error generating response: {e}"

    def _generate_openai(self, prompt: str, **kwargs) -> str:
        """Generate using OpenAI API."""
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

            response = self.openai_client.chat.completions.create(
                model=kwargs.get("model", self.config.openai_model),
                messages=messages,
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_new_tokens),
                top_p=kwargs.get("top_p", self.config.top_p),
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return f"Error generating response: {e}"

    def _generate_stub(self, query: str, context: str, sources: List[Dict]) -> str:
        """
        Stub generator for when no LLM is available.
        Returns a formatted response with the retrieved context.
        """
        if not context:
            return "I couldn't find relevant information to answer your question."

        # Create a simple extractive answer
        sentences = context.split(". ")[:3]  # First 3 sentences
        answer = ". ".join(sentences)

        if sources:
            source_info = sources[0].get("metadata", {})
            source_name = source_info.get("source", "document")
            page = source_info.get("page", "N/A")
            answer += f"\n\n[Source: {source_name}, Page: {page}]"

        return answer

    def stream_generate(self, query: str, context: str, sources: List[Dict], **kwargs):
        """
        Stream generation for real-time response.
        Yields tokens as they're generated.
        """
        # For now, just yield the complete response
        # Can be extended for true streaming with HF transformers or OpenAI streaming
        result = self.generate(query, context, sources, **kwargs)

        # Simulate streaming by yielding words
        words = result.text.split()
        for word in words:
            yield word + " "
            time.sleep(0.05)  # Simulate typing delay
