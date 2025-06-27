import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.nn.functional import softmax
from tqdm import tqdm
import re

# ======================= CONFIG ============================
MODEL_PATH = "../../TextAttack/roberta_adv_finetuned"
TOKENIZER_PATH = "../saved_models/roberta_v3/roberta_v3_tokenizer"
DATA_PATH = "../../../datasets/Combined_Corpus/All_cleaned.csv"  # <- modifică
OUTPUT_CSV = "roberta_misclassified.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 512

# ======================= FUNCȚII ============================
def wordopt(text):
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    text = text.replace('\n', ' ')
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_df(df):
    df["word_count"] = df["text"].apply(lambda x: len(x.split()))
    df = df[df["word_count"] >= 30]
    df = df[df["language"] == "en"]
    df["text"] = df["text"].apply(wordopt)
    return df.drop(columns=["word_count"])

# ======================= ÎNCĂRCARE ============================
df = pd.read_csv(DATA_PATH)
df = preprocess_df(df)

train_val, test = train_test_split(df, test_size=0.15, random_state=42, stratify=df["label"])
train, val = train_test_split(train_val, test_size=0.1765, random_state=42, stratify=train_val["label"])  # 0.1765 * 0.85 ≈ 0.15

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

# ======================= TOKENIZER + MODEL ============================
tokenizer = RobertaTokenizer.from_pretrained(TOKENIZER_PATH)
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

# ======================= INFERENȚĂ ============================
wrong_preds = []

for idx, row in tqdm(test.iterrows(), total=len(test)):
    text = row["text"]
    true_label = row["label"]
    
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        conf = probs[0][pred].item()

    if pred != true_label:
        wrong_preds.append({
            "text": text,
            "predicted_label": pred,
            "correct_label": true_label,
            "confidence": round(conf, 4)
        })

# ======================= SALVARE ============================
out_df = pd.DataFrame(wrong_preds)
out_df.to_csv(OUTPUT_CSV, index=False)
print(f"Salvat {len(out_df)} articole greșit clasificate în {OUTPUT_CSV}")
