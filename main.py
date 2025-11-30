import argparse
import os
from tqdm import tqdm
from datasets import load_dataset
from espnet2.bin.s2t_inference import Speech2Text
import pandas as pd

# Registered models
REGISTERED_MODELS = ["owsm_v4", "1-best_llm"]

# Prediction output directory
PREDICTIONS_DIR = "predictions"


def get_prediction_path(dataset: str, subset: str, split: str, model: str) -> str:
    """
    Generate prediction file path with model name.
    
    Args:
        dataset: Dataset name (e.g., 'edinburghcstr/ami')
        subset: Dataset subset (e.g., 'ihm')
        split: Dataset split (e.g., 'test')
        model: Model name (e.g., 'owsm_v4')
    
    Returns:
        Path to the prediction CSV file.
    """
    subset_tag = subset if subset is not None else "none"
    filename = f"{dataset.replace('/', '_')}_{subset_tag}_{split}_{model}_preds.csv"
    return os.path.join(PREDICTIONS_DIR, filename)


def run_owsm_inference(args):
    """Run OWSM model inference."""
    print(f"Loading ASR model: {args.model}")
    
    PRETRAINED_MODEL = "espnet/owsm_v4_small_370M"
    owsm_language = "<eng>"
    model = Speech2Text.from_pretrained(
        PRETRAINED_MODEL,
        device="cuda",
        dtype="bfloat16",
        use_flash_attn=False,
        lang_sym=owsm_language,
        task_sym="<asr>",
        nbest=args.nbest,
    )
    
    splits = [s.strip() for s in args.splits.split(",")]

    for split in splits:
        print(f"\nLoading dataset: {args.dataset}, subset={args.subset}, split={split}")

        # Load according to whether subset is provided
        if args.subset is None:
            ds = load_dataset(args.dataset, cache_dir=args.cache_dir, split=split)
        else:
            ds = load_dataset(args.dataset, args.subset, cache_dir=args.cache_dir, split=split)

        audio_list = ds["audio"]
        text_list = ds["text"] if args.dataset == 'edinburghcstr/ami' else ds["transcription"]

        # Determine how many samples to process
        total = len(ds) if args.limit is None else min(len(ds), args.limit)
        print(f"Processing {total} / {len(ds)} samples...")

        results = []

        print(f"Running inference on split: {split} ...")

        for i in tqdm(range(total)):
            wav = audio_list[i]["array"]
            gt_text = text_list[i]
            
            # [(text, token, token_int, text_nospecial, hypothesis object)
            # see ./espnet/espnet2/bin/s2t_inference, ./espnet/espnet/nets/beam_search.py
            hyps = model(wav)
            
            # Create a dictionary for the current sample
            row = {"gt": gt_text}
            
            for j, hyp in enumerate(hyps):
                # hyp is a tuple: (text, token, token_int, text_nospecial, hypothesis_object)
                # We want text_nospecial (index 3) and score from hypothesis_object (index 4)
                pred_text = hyp[3]
                # hypothesis object is at index 4
                hyp_obj = hyp[4]
                score = hyp_obj.score
                
                # Convert tensor to float if necessary
                if hasattr(score, "item"):
                    score = score.item()
                
                # Store 1-based index keys
                row[f"pred_text_{j+1}"] = pred_text
                row[f"pred_score_{j+1}"] = score

            results.append(row)

        # Save predictions
        df = pd.DataFrame(results)
        out_csv = get_prediction_path(args.dataset, args.subset, split, args.model)
        df.to_csv(out_csv, index=False)
        print(f"Saved predictions → {out_csv}")


def run_1best_llm_inference(args):
    """Run 1-best LLM post-processing inference."""
    from models import OneBestLLM
    
    # Create the LLM model instance
    print(f"Initializing 1-best LLM model: {args.llm_model} (backend: {args.llm_backend})")
    llm_model = OneBestLLM(
        model_name=args.llm_model,
        backend=args.llm_backend,
        vllm_api_base=args.vllm_url,
    )
    
    splits = [s.strip() for s in args.splits.split(",")]
    
    for split in splits:
        # Look for the source prediction CSV (from owsm_v4)
        source_model = "owsm_v4"
        source_csv = get_prediction_path(args.dataset, args.subset, split, source_model)
        
        if not os.path.exists(source_csv):
            raise FileNotFoundError(
                f"Source prediction file not found: {source_csv}\n"
                f"Please run OWSM inference first with: "
                f"python main.py --model owsm_v4 --dataset {args.dataset} "
                f"--subset {args.subset or 'none'} --splits {split}"
            )
        
        print(f"\nLoading source predictions from: {source_csv}")
        df = pd.read_csv(source_csv)
        
        # Determine how many samples to process
        total = len(df) if args.limit is None else min(len(df), args.limit)
        print(f"Processing {total} / {len(df)} samples...")
        
        results = []
        
        print(f"Running LLM post-processing on split: {split} ...")
        
        for i in tqdm(range(total)):
            row = df.iloc[i]
            gt_text = row["gt"]
            pred_text_1 = row["pred_text_1"]
            
            # Run LLM post-processing on 1-best prediction
            corrected_text = llm_model(pred_text_1)
            
            result_row = {
                "gt": gt_text,
                "pred_text_1": corrected_text,
                "original_pred": pred_text_1,
            }
            results.append(result_row)
        
        # Save predictions
        result_df = pd.DataFrame(results)
        out_csv = get_prediction_path(args.dataset, args.subset, split, args.model)
        result_df.to_csv(out_csv, index=False)
        print(f"Saved predictions → {out_csv}")


def main():
    # ---------------------------
    # Argument Parser
    # ---------------------------
    parser = argparse.ArgumentParser(description="Run ESPnet ASR inference on a dataset.")
    parser.add_argument("--model", type=str, default='owsm_v4',
                        choices=REGISTERED_MODELS,
                        help=f"Model name. Options: {REGISTERED_MODELS}")
    parser.add_argument("--dataset", type=str, default='edinburghcstr/ami',
                        help="Dataset name, e.g. 'edinburghcstr/ami', 'distil-whisper/earnings22'")
    parser.add_argument("--subset", type=str, default=None,
                        help="Dataset subset/config, e.g. 'ihm', 'sdm' for 'ami', and 'chunked' for 'earnings22'.")
    parser.add_argument("--splits", type=str, default="test",
                        help="Comma-separated dataset splits, e.g. 'train,test'")
    parser.add_argument("--nbest", type=int, default=1,
                        help="Number of hypotheses to output from the model.")
    parser.add_argument("--cache_dir", type=str, default="/data/lyuxing/hf_datasets",
                        help="Cache directory for HuggingFace datasets.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples to process (for sanity checking).")
    
    # LLM-specific arguments
    parser.add_argument("--llm_model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="LLM model name for post-processing (HuggingFace model ID)")
    parser.add_argument("--llm_backend", type=str, default="transformers",
                        choices=["transformers", "vllm"],
                        help="LLM backend: 'transformers' (direct load) or 'vllm' (API server)")
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000/v1",
                        help="vLLM server URL (only used with --llm_backend vllm)")

    args = parser.parse_args()

    # ---------------------------
    # Create predictions directory
    # ---------------------------
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    # ---------------------------
    # Run inference based on model type
    # ---------------------------
    if args.model == "owsm_v4":
        run_owsm_inference(args)
    elif args.model == "1-best_llm":
        run_1best_llm_inference(args)
    else:
        raise NotImplementedError(f"Model '{args.model}' is not implemented.")


if __name__ == "__main__":
    main()
