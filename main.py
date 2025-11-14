import torch
from tqdm import tqdm
from datasets import load_dataset
from espnet2.bin.s2t_inference import Speech2Text
import pandas as pd

# Load model
PRETRAINED_MODEL="espnet/owsm_v4_small_370M" # you may try out "espnet/owsm_ctc_v4_1B" if you have GPU mem > 15GB
owsm_language="<eng>" # language code in ISO3
model = Speech2Text.from_pretrained(
    PRETRAINED_MODEL,
    device="cuda",
    dtype="bfloat16",
    use_flash_attn=False, # set to True for better efficiency if flash attn is installed and dtype is float16 or bfloat16
    lang_sym=owsm_language,
    task_sym='<asr>',
)

# Load dataset
for split in ['train', 'test']:
  ds = load_dataset("edinburghcstr/ami", "ihm", cache_dir="/data/lyuxing/hf_datasets", split=split)
  audio_list = ds['audio']  # load into cache memory for faster access
  text_list = ds['text']


  '''
  # Inference example (first element)
  pred = model(ds['audio'][0]['array'])[0][3] # [3]: without language, task and timestamps tags
  gt = ds['text'][0]
  '''

  all_gts = []
  all_preds = []

  print(f"Running inference on {split} split...")

  for i in tqdm(range(len(ds))):
      wav = audio_list[i]["array"]       # numpy array shape (T,)
      gt_text = text_list[i]             # ground truth string

      # Single-item inference
      pred_text = model(wav)[0][3]        # predicted hypothesis

      all_gts.append(gt_text)
      all_preds.append(pred_text)

  # Save results
  df = pd.DataFrame({
      "gt": all_gts,
      "pred": all_preds,
  })
  result_csv = f"ami_owsm_{split}_predictions.csv"
  df.to_csv(result_csv, index=False)
  print(f"Saved predictions to {result_csv}")
