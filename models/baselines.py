"""
LLM-based post-processing for ASR predictions.
Uses locally deployed LLM via vLLM or HuggingFace Transformers.
"""

import os
from typing import Optional
import pandas as pd


class OneBestLLM:
    """
    Post-processes 1-best ASR predictions using a locally deployed LLM.
    
    Supports two backends:
    1. vLLM server (OpenAI-compatible API) - recommended for speed
    2. HuggingFace Transformers (direct loading) - simpler setup
    
    This model takes the top-1 prediction from an ASR model (e.g., OWSM)
    and uses an LLM to correct potential transcription errors.
    """
    
    DEFAULT_SYSTEM_PROMPT = """You are an expert ASR post-processor. Your task is to correct transcription errors in the given ASR output.

Rules:
1. Fix obvious spelling errors and typos
2. Correct grammatical issues that are likely transcription errors
3. Fix common ASR mistakes (homophones, word boundaries, etc.)
4. Preserve the original meaning and intent
5. Do NOT add or remove significant content
6. Do NOT change the style or formality of the text
7. If the transcription looks correct, return it unchanged

Return ONLY the corrected transcription, nothing else."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        backend: str = "transformers",
        vllm_api_base: str = "http://localhost:8000/v1",
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        device: str = "cuda",
    ):
        """
        Initialize the local LLM post-processor.
        
        Args:
            model_name: HuggingFace model name or path.
                Recommended for 4090 (24GB):
                - "Qwen/Qwen2.5-7B-Instruct" (fastest, great quality)
                - "meta-llama/Llama-3.1-8B-Instruct" (requires HF token)
                - "mistralai/Mistral-7B-Instruct-v0.3"
            backend: "transformers" (direct load) or "vllm" (API server).
            vllm_api_base: vLLM server URL (only used if backend="vllm").
            system_prompt: Custom system prompt. If None, uses default.
            max_new_tokens: Maximum tokens to generate.
            device: Device to use ("cuda" or "cpu").
        """
        self.model_name = model_name
        self.backend = backend
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.vllm_api_base = vllm_api_base
        
        if backend == "transformers":
            self._init_transformers()
        elif backend == "vllm":
            self._init_vllm_client()
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'transformers' or 'vllm'.")
    
    def _init_transformers(self):
        """Initialize HuggingFace Transformers model."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
    
    def _init_vllm_client(self):
        """Initialize OpenAI client for vLLM server."""
        from openai import OpenAI
        
        self.client = OpenAI(
            api_key="not-needed",  # vLLM doesn't require API key
            base_url=self.vllm_api_base,
        )
        print(f"Connected to vLLM server at {self.vllm_api_base}")
        
    def __call__(self, asr_text: str) -> str:
        """
        Post-process an ASR transcription using the LLM.
        
        Args:
            asr_text: The ASR transcription to post-process.
            
        Returns:
            The corrected transcription.
        """
        return self.correct(asr_text)
    
    def correct(self, asr_text: str) -> str:
        """
        Post-process an ASR transcription using the LLM.
        
        Args:
            asr_text: The ASR transcription to post-process.
            
        Returns:
            The corrected transcription.
        """
        # Handle NaN/None/empty values
        if asr_text is None or (isinstance(asr_text, float) and pd.isna(asr_text)):
            return ""
        
        asr_text = str(asr_text)
        if not asr_text.strip():
            return asr_text
        
        if self.backend == "transformers":
            return self._correct_transformers(asr_text)
        else:
            return self._correct_vllm(asr_text)
    
    def _correct_transformers(self, asr_text: str) -> str:
        """Correct using HuggingFace Transformers."""
        import torch
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"ASR Output: {asr_text}"},
        ]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # Greedy decoding for consistency
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the generated tokens (exclude prompt)
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        
        return response.strip()
    
    def _correct_vllm(self, asr_text: str) -> str:
        """Correct using vLLM server."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"ASR Output: {asr_text}"},
                ],
                max_tokens=self.max_new_tokens,
                temperature=0,  # Greedy decoding
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Warning: vLLM request failed: {e}")
            return asr_text
