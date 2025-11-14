# CMU 18781 Term Project: LLM REASONING FOR SPEECH RECOGNITION ERROR CORRECTION

## Quick Start

### 1. Installation

```bash
# Create conda env
conda create -n asr_llm python=3.10
conda activate asr_llm

# Install espnet and other dependencies
cd espnet
pip install -e .
pip install espnet-model-zoo "datasets<4.0.0" torchaudio
```

### 2. Inference
```
python main.py
```
