#!/usr/bin/env python3
import os
# Disable all GPUs—CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import re
import string
import numpy as np
import pandas as pd
import fasttext
import tensorflow as tf

# Hide GPUs in TF2
tf.config.set_visible_devices([], "GPU")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# ─── CONFIG ───────────────────────────────────────────────────────────────────

# 1) FastText supervised bin
FT_BIN       = "combined_corpus_nb/fasttext_supervised_model.bin"
# 2) Folder where your three Keras models live
MODELS_DIR   = "combined_corpus_nb/saved_models/fasttext_supervised"
# 3) The 100-sample test file
TEST_TSV     = "../datasets/teste100.tsv"

# Must match what you did at training
MAX_NUM_WORDS = 20000
MAX_SEQ_LEN   = 100

# ─── TEXT CLEANING ─────────────────────────────────────────────────────────────

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r'<[^>]+>', '', t)
    t = re.sub(r'https?://\S+|www\.\S+', '', t)
    t = re.sub(r'\d+', '', t)
    t = t.translate(str.maketrans("", "", string.punctuation))
    t = re.sub(r'\s+', ' ', t).strip()
    return " ".join(
        lemmatizer.lemmatize(w) for w in t.split()
        if w not in stop_words
    )

def normalize_label(raw: str) -> str:
    r = raw.strip().upper()
    if r in ("1","__LABEL__1","REAL"): return "REAL"
    if r in ("0","__LABEL__0","FAKE"): return "FAKE"
    raise ValueError(f"Unknown label '{raw}'")

# ─── STEP 1: Load FT bin ───────────────────────────────────────────────────────

print("🔄 Loading FastText supervised model…")
ft = fasttext.load_model(FT_BIN)
dim = ft.get_dimension()
print(f"   FastText embedding dim = {dim}")

# ─── STEP 2: Rebuild Tokenizer (must match training) ────────────────────────────

print("🔄 Loading full cleaned corpus to rebuild tokenizer…")
all_clean = "../datasets/Combined_Corpus/All_cleaned.csv"
df_all = pd.read_csv(all_clean)
df_all = df_all.dropna(subset=["text"])
if "language" in df_all.columns:
    df_all = df_all[df_all["language"]=="en"]
df_all["clean"] = df_all["text"].astype(str).map(clean_text)

tok = Tokenizer(num_words=MAX_NUM_WORDS)
tok.fit_on_texts(df_all["clean"].tolist())
print(f"   Vocab size (capped at {MAX_NUM_WORDS}):", len(tok.word_index))

# ─── STEP 3: Load & preprocess test set ────────────────────────────────────────

print("🔄 Loading and cleaning test set…")
df_test = pd.read_csv(TEST_TSV, sep="\t", dtype=str)
assert "text" in df_test.columns and "label(1=real,0=fake)" in df_test.columns

df_test["clean"] = df_test["text"].map(clean_text)
seqs = tok.texts_to_sequences(df_test["clean"].tolist())
X_test = pad_sequences(seqs, maxlen=MAX_SEQ_LEN)

y_true = [normalize_label(l) for l in df_test["label(1=real,0=fake)"]]

# ─── STEP 4: Inference with each model ────────────────────────────────────────

for model_name in ["LSTM", "CNN", "CNN_LSTM"]:
    h5 = os.path.join(MODELS_DIR, f"{model_name}_fasttext.h5")
    if not os.path.exists(h5):
        print(f"⚠️  Missing model file: {h5}, skipping {model_name}")
        continue

    print(f"\n🔄 Loading Keras model '{model_name}' …")
    model = load_model(h5, compile=False)

    print("🚀 Running inference…")
    probs = model.predict(X_test, batch_size=128, verbose=0).flatten()
    preds = ["REAL" if p>0.5 else "FAKE" for p in probs]

    acc = accuracy_score(y_true, preds)
    print(f"✅ {model_name:<8} — Accuracy: {acc*100:.2f}%")

    out = df_test[["text", "label(1=real,0=fake)"]].copy()
    out["predicted"]      = preds
    out["predicted_prob"] = probs
    fname = f"teste100_{model_name}_predictions.tsv"
    out.to_csv(fname, sep="\t", index=False)
    print(f"💾 Saved predictions → {fname}")

print("\n✅ All done!")
