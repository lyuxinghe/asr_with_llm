import argparse
import string
import pandas as pd
import math


def preprocess(text) -> str:
    # handle NaN / non-string
    if not isinstance(text, str):
        # pandas NaN is a float that is math.isnan(...)
        if text is None:
            text = ""
        elif isinstance(text, float) and math.isnan(text):
            text = ""
        else:
            text = str(text)

    # remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))
    # convert to lowercase
    text = text.lower()
    return text



def edit_distance(ref, hyp):
    """
    ref, hyp: lists of tokens (words)
    returns: full edit-distance DP table
    """
    len_r, len_h = len(ref), len(hyp)

    # edit distance table
    ed_table = [[0] * (len_h + 1) for _ in range(len_r + 1)]

    # base cases
    for i in range(len_r + 1):
        ed_table[i][0] = i
    for j in range(len_h + 1):
        ed_table[0][j] = j

    # fill the table
    for i in range(1, len_r + 1):
        for j in range(1, len_h + 1):
            if ref[i - 1] == hyp[j - 1]:
                ed_table[i][j] = ed_table[i - 1][j - 1]
            else:
                ed_table[i][j] = 1 + min(
                    ed_table[i - 1][j],     # deletion
                    ed_table[i][j - 1],     # insertion
                    ed_table[i - 1][j - 1]  # substitution
                )

    return ed_table


def wer(hyp: str, ref: str) -> float:
    """
    Compute WER (%) for a single hypothesis/reference pair.
    """
    # 1. Preprocess
    ref = preprocess(ref)
    hyp = preprocess(hyp)
    ref_tokens = ref.split()
    hyp_tokens = hyp.split()

    if len(ref_tokens) == 0:
        # Avoid division by zero; define WER as 0 if reference is empty
        return 0.0

    # 2. Edit distance table
    ed_table = edit_distance(ref_tokens, hyp_tokens)

    # 3. Normalize by number of words in reference
    error_rate = ed_table[-1][-1] / len(ref_tokens) * 100.0
    return round(error_rate, 2)


def corpus_wer(preds, gts) -> float:
    """
    Compute corpus-level WER (%) over lists of predictions and references.
    """
    total_edits = 0
    total_ref_words = 0

    for hyp, ref in zip(preds, gts):
        ref_p = preprocess(ref)
        hyp_p = preprocess(hyp)
        ref_tokens = ref_p.split()
        hyp_tokens = hyp_p.split()

        if len(ref_tokens) == 0:
            continue

        ed_table = edit_distance(ref_tokens, hyp_tokens)
        dist = ed_table[-1][-1]

        total_edits += dist
        total_ref_words += len(ref_tokens)

    if total_ref_words == 0:
        return 0.0

    return round(total_edits / total_ref_words * 100.0, 2)


def main():
    parser = argparse.ArgumentParser(
        description="Compute WER from a CSV with columns 'gt' and 'pred'."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV file (must contain columns 'gt' and 'pred').",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    if "gt" not in df.columns or "pred" not in df.columns:
        raise ValueError("CSV must contain 'gt' and 'pred' columns.")

    gts = df["gt"].tolist()
    preds = df["pred"].tolist()

    # Sentence-level WERs
    sent_wers = [wer(h, r) for h, r in zip(preds, gts)]    
    avg_sent_wer = sum(sent_wers) / len(sent_wers) if sent_wers else 0.0

    # Corpus-level WER
    corp_wer = corpus_wer(preds, gts)

    print(f"Number of utterances: {len(df)}")
    print(f"Average sentence-level WER: {avg_sent_wer:.2f}%")
    print(f"Corpus-level WER:          {corp_wer:.2f}%")


if __name__ == "__main__":
    main()
