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
```bash
# AMI dataset, IHM subset 
python main.py --dataset edinburghcstr/ami --subset ihm
# AMI dataset, SDM subset 
python main.py --dataset edinburghcstr/ami --subset sdm
# Earnings22 dataset 
python main.py --dataset edinburghcstr/ami --subset chunked
# Evaluate test set only
python main.py --split test
# Using different models and enabling topk predictions
python main.py --model owsm_v4 --nbest 10
```
### 3. WER Calculation
```bash
python wer.py --csv ${Path_to_your_csv_file}$
```