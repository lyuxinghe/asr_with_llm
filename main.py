import argparse
import torch
from tqdm import tqdm
from datasets import load_dataset
from espnet2.bin.s2t_inference import Speech2Text
import pandas as pd


def main():

    # ---------------------------
    # Argument Parser
    # ---------------------------
    parser = argparse.ArgumentParser(description="Run ESPnet ASR inference on a dataset.")
    parser.add_argument("--model", type=str, default='owsm_v4',
                        help="Model name, e.g. 'owsm_v4'")
    parser.add_argument("--dataset", type=str, default='edinburghcstr/ami',
                        help="Dataset name, e.g. 'edinburghcstr/ami', 'distil-whisper/earnings22'")
    parser.add_argument("--subset", type=str, default=None,
                        help="Dataset subset/config, e.g. 'ihm', 'sdm' for 'ami', and 'chunked' for 'earnings22'.")
    parser.add_argument("--splits", type=str, default="train,test",
                        help="Comma-separated dataset splits, e.g. 'train,test'")
    parser.add_argument("--nbest", type=int, default=1,
                        help="Number of hypotheses to output from the model.")
    parser.add_argument("--cache_dir", type=str, default="/data/lyuxing/hf_datasets",
                        help="Cache directory for HuggingFace datasets.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples to process (for sanity checking).")

    args = parser.parse_args()

    # ---------------------------
    # Load Model
    # ---------------------------
    print(f"Loading ASR model: {args.model}")
    if args.model == "owsm_v4":
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
    else:
        raise NotImplementedError()
    
    # ---------------------------
    # Run inference over each split
    # ---------------------------
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

        all_gts = []
        all_preds = []

        print(f"Running inference on split: {split} ...")

        for i in tqdm(range(total)):
            wav = audio_list[i]["array"]
            gt_text = text_list[i]

            hyps = model(wav)
            breakpoint()
            pred_text = hyps[0][3]  # Only saving the best prediction, even if nbest != 1

            all_gts.append(gt_text)
            all_preds.append(pred_text)

        # ---------------------------
        # Save predictions
        # ---------------------------
        df = pd.DataFrame({"gt": all_gts, "pred": all_preds})

        subset_tag = args.subset if args.subset is not None else "none"
        out_csv = f"{args.dataset.replace('/', '_')}_{subset_tag}_{split}_preds.csv"

        df.to_csv(out_csv, index=False)
        print(f"Saved predictions → {out_csv}")


if __name__ == "__main__":
    main()
