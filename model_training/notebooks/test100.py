import os
import sys
import re
import torch
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from tqdm import tqdm

# ── MODIFY THESE PATHS TO MATCH YOUR SETUP ─────────────────────────────────────

# Directory where you keep all your saved models
MODELS_DIR = "combined_corpus_nb/saved_models"

# Path to the folder containing the RoBERTa tokenizer
ROBERTA_TOKENIZER_PATH = os.path.join(MODELS_DIR, "roberta_v3", "roberta_v3_tokenizer")

# Path to the folder where your fine‐tuned RoBERTa model lives
ROBERTA_MODEL_PATH = "TextAttack/roberta_adv_finetuned"

# Maximum token length used at inference (should match training)
MAX_LENGTH = 512

# Device for inference
device_bert = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── PATH TO YOUR TSV ───────────────────────────────────────────────────────────

# Înlocuiește cu calea completă ori relativă către fișierul tău .tsv
TSV_PATH = "../datasets/teste100.tsv"


# ── PREPROCESSING FUNCTION ────────────────────────────────────────────────────

def wordopt_lite(text: str) -> str:
    """
    A minimal “cleaning” step: replace URLs with [URL], remove non-ASCII,
    collapse whitespace, etc. Must match what you used during training.
    """
    text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)
    text = text.replace("\n", " ")
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── LOAD TOKENIZER & MODEL ONCE ────────────────────────────────────────────────

print("🔄 Loading RoBERTa tokenizer…")
roberta_tok = RobertaTokenizer.from_pretrained(ROBERTA_TOKENIZER_PATH)

print("🔄 Loading RoBERTa model…")
roberta_model = RobertaForSequenceClassification.from_pretrained(ROBERTA_MODEL_PATH)
roberta_model.to(device_bert)
roberta_model.eval()


# ── INFERENCE FUNCTION ────────────────────────────────────────────────────────

@torch.no_grad()
def predict_roberta(text: str) -> (str, float):
    """
    1) Clean with wordopt_lite
    2) Tokenize + pad/truncate to MAX_LENGTH
    3) Run through RoBERTa, apply softmax to logits
    4) Return (“REAL” or “FAKE”, probability_of_that_choice)
    """
    cleaned = wordopt_lite(text)
    toks = roberta_tok(
        cleaned,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    toks = {k: v.to(device_bert) for k, v in toks.items()}

    out = roberta_model(**toks)
    logits = out.logits[0]
    probs = torch.softmax(logits, dim=-1)

    pred_idx = probs.argmax().item()
    prob_val = probs[pred_idx].item()

    label_str = "REAL" if pred_idx == 1 else "FAKE"
    return label_str, prob_val


# ── MAIN EVALUATION SCRIPT ────────────────────────────────────────────────────

def main():
    # 1) Read the TSV into a DataFrame
    if not os.path.exists(TSV_PATH):
        print(f"❌ File not found: {TSV_PATH}")
        sys.exit(1)

    print(f"📂 Reading TSV from '{TSV_PATH}' …")
    df = pd.read_csv(TSV_PATH, sep='\t', dtype=str)

    if "text" not in df.columns or "label(1=real,0=fake)" not in df.columns:
        print("❌ The TSV must have columns named exactly 'text' and 'label(1=real,0=fake)'.")
        sys.exit(1)

    df = df.dropna(subset=["text", "label(1=real,0=fake)"]).reset_index(drop=True)
    total = len(df)
    if total == 0:
        print("⚠️ No rows to evaluate.")
        sys.exit(0)

    # Prepare lists to store predictions
    predicted_labels = []
    predicted_probs  = []

    correct = 0
    incorrect = 0

    def normalize_label(lab: str) -> str:
        lab = lab.strip().upper()
        if lab in ("1", "__LABEL__1"):
            return "REAL"
        if lab in ("0", "__LABEL__0"):
            return "FAKE"
        if lab in ("REAL", "FAKE"):
            return lab
        raise ValueError(f"Unrecognized label '{lab}'")

    print("🚀 Running inference on each row …")
    for idx, row in tqdm(df.iterrows(), total=total):
        text = row["text"]
        true_lab = normalize_label(row["label(1=real,0=fake)"])

        try:
            pred_lab, pred_prob = predict_roberta(text)
        except Exception as e:
            print(f"⚠️ Error on row {idx}: {e}")
            incorrect += 1
            # For errors, we still append something so lengths match
            predicted_labels.append("ERROR")
            predicted_probs.append(0.0)
            continue

        predicted_labels.append(pred_lab)
        predicted_probs.append(pred_prob)

        if pred_lab == true_lab:
            correct += 1
        else:
            incorrect += 1

    accuracy = correct / total
    print("────────────────────────────────────────────")
    print(f"Total examples  : {total}")
    print(f"Correct         : {correct}")
    print(f"Incorrect       : {incorrect}")
    print(f"Accuracy        : {accuracy * 100:.2f}%")
    print("────────────────────────────────────────────")

    # 2) Add the new columns to the DataFrame
    df["predicted"]      = predicted_labels
    df["predicted_prob"] = predicted_probs

    # 3) Write out a new TSV (same folder, name appended)
    base_dir, fname = os.path.split(TSV_PATH)
    output_fname = os.path.splitext(fname)[0] + "_with_predictions.tsv"
    output_path = os.path.join(base_dir, output_fname)

    print(f"💾 Saving results to '{output_path}' …")
    df.to_csv(output_path, sep="\t", index=False)

    print("✅ Done. You can open the new TSV with the 'predicted' column.")


if __name__ == "__main__":
    main()
