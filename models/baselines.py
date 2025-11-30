"""
LLM-based post-processing for ASR predictions.
Uses locally deployed LLM via vLLM or HuggingFace Transformers.
"""

import os
from typing import Optional, List, Tuple
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
    
    def _build_prompt(self, asr_text: str) -> str:
        """Build the user prompt."""
        return f"ASR Output: {asr_text}"
        
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
        
        prompt = self._build_prompt(asr_text)
        
        if self.backend == "transformers":
            return self._generate_transformers(prompt)
        else:
            return self._generate_vllm(prompt)
    
    def _generate_transformers(self, prompt: str) -> str:
        """Generate response using HuggingFace Transformers."""
        import torch
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
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
    
    def _generate_vllm(self, prompt: str) -> str:
        """Generate response using vLLM server."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_new_tokens,
                temperature=0,  # Greedy decoding
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Warning: vLLM request failed: {e}")
            return prompt.replace("ASR Output: ", "")  # Return original on failure


class NBestLLM(OneBestLLM):
    """
    Post-processes N-best ASR hypotheses using a locally deployed LLM.
    
    Inherits from OneBestLLM with the same interface.
    The only difference is the prompt format - it shows all N-best hypotheses
    with their scores to help the LLM make a better decision.
    
    Input: List of (text, score) tuples
    Output: Corrected transcription string
    """
    
    DEFAULT_SYSTEM_PROMPT = """You are an expert ASR post-processor. You will be given a list of candidate transcriptions from an ASR system, ranked by confidence score.

Your task is to determine the most accurate transcription. Consider:
1. Words appearing in multiple candidates are more likely correct
2. Higher scores indicate higher confidence, but the top candidate isn't always correct
3. Fix common ASR errors: homophones, word boundaries, spelling
4. Ensure grammatical correctness and semantic coherence
5. Preserve the original meaning and intent
6. If a candidate looks correct, you can return it unchanged

Return ONLY the corrected transcription, nothing else."""

    def __call__(self, hypotheses: List[Tuple[str, float]]) -> str:
        """
        Post-process N-best ASR hypotheses using the LLM.
        
        Args:
            hypotheses: List of (text, score) tuples, ranked by score (highest first).
            
        Returns:
            The corrected transcription string.
        """
        return self.correct(hypotheses)
    
    def correct(self, hypotheses: List[Tuple[str, float]]) -> str:
        """
        Post-process N-best ASR hypotheses using the LLM.
        
        Args:
            hypotheses: List of (text, score) tuples.
            
        Returns:
            The corrected transcription string.
        """
        # Handle empty case
        if not hypotheses:
            return ""
        
        # Filter out empty/NaN hypotheses
        valid_hypotheses = []
        for text, score in hypotheses:
            if text is None or (isinstance(text, float) and pd.isna(text)):
                continue
            text = str(text)
            if text.strip():
                valid_hypotheses.append((text, score))
        
        if not valid_hypotheses:
            return ""
        
        # If only one valid hypothesis, just process it like OneBestLLM
        if len(valid_hypotheses) == 1:
            return super().correct(valid_hypotheses[0][0])
        
        prompt = self._build_prompt(valid_hypotheses)
        
        if self.backend == "transformers":
            return self._generate_transformers(prompt)
        else:
            return self._generate_vllm(prompt)
    
    def _build_prompt(self, hypotheses: List[Tuple[str, float]]) -> str:
        """Build the user prompt with all N-best hypotheses."""
        lines = ["ASR Candidates:"]
        
        for i, (text, score) in enumerate(hypotheses, 1):
            if score is None or (isinstance(score, float) and pd.isna(score)):
                score_str = "N/A"
            else:
                score_str = f"{score:.4f}"
            
            lines.append(f"[{i}] (score: {score_str}) {text}")
        
        return "\n".join(lines)
