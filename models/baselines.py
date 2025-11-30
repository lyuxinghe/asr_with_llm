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
    
    DEFAULT_SYSTEM_PROMPT = """You are an ASR error correction system. Your task is to fix word-level errors in speech recognition output.
EXAMPLES:
Input: "yeah yeah we're gonna meet up"
Output: "yeah yeah we're gonna meet up"
(Keep repeated words exactly as-is)

Input: "i mean we're not even their yet"
Output: "i mean we're not even there yet"
(Fix "their" → "there", keep everything else)

Input: "um i don't know um what to do"
Output: "um i don't know um what to do"
(Keep all "um" and disfluencies)

Input: "we need to do do that first"
Output: "we need to do do that first"
(Keep repeated "do do" - speaker stuttered)

Input: "the recei get there"
Output: "the receipt get there"
(Fix misspelling, keep grammar errors)

Input: "참 yeah okay"
Output: "um yeah okay"
(Replace non-English with likely English word)

RULES:
- NEVER remove repeated words like "yeah yeah", "i i", "the the"
- NEVER remove filler words like "um", "uh", "like", "you know"
- NEVER fix grammar - keep "we was" as "we was"
- ONLY fix: misspellings, wrong homophones, non-English characters
- Output must have SAME word count as input

Return ONLY the corrected text."""

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
    
    DEFAULT_SYSTEM_PROMPT = """You are an ASR error correction system. You will be given multiple candidate transcriptions from a speech recognition system with confidence scores.
EXAMPLE:
Candidates:
[1] (score: -1.5) "yeah yeah we gonna meet"
[2] (score: -2.0) "yeah we're gonna meet"
[3] (score: -2.5) "yeah yeah we're going to meet"
Output: "yeah yeah we're gonna meet"
(Keep "yeah yeah" from [1], use "we're gonna" from [2])

EXAMPLE:
Candidates:
[1] (score: -3.0) "참 okay done"
[2] (score: -4.0) "um okay done"
Output: "um okay done"
(Replace non-English "참" with "um" from candidate [2])

EXAMPLE:
Candidates:
[1] (score: -1.0) "i i don't know"
[2] (score: -1.5) "i don't know"
Output: "i i don't know"
(Keep repeated "i i" - the speaker stuttered)

EXAMPLE:
Candidates:
[1] (score: -2.0) "we need to do do that"
[2] (score: -2.5) "we need to do that"
Output: "we need to do do that"
(Keep repeated "do do" from top candidate - speaker stuttered)

RULES:
- NEVER remove repeated words - keep "yeah yeah", "i i", "the the"
- NEVER remove fillers - keep "um", "uh", "like"
- Prefer words appearing in multiple candidates
- Replace non-English characters with English from other candidates
- Higher scores (less negative) = higher confidence
- Output word count should match the best candidate

Return ONLY the transcription."""

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
