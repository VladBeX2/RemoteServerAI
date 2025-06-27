import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import re
import string
import pickle
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

from sklearn.metrics import accuracy_score

# ─── CONFIG ────────────────────────────────────────────────────────────────────

TSV_PATH        = "../datasets/teste100.tsv"
MODEL_DIR       = "combined_corpus_nb/saved_models"                  # adjust if needed
TOKENIZER_PATH  = os.path.join(MODEL_DIR, "cnn_lstm_glove", "tokenizer.pkl")
MAX_SEQUENCE_LEN = 100

# mapping model names → .h5 files
MODEL_FILES = {
    "LSTM":      os.path.join(MODEL_DIR, "cnn_lstm_glove","LSTM_GloVe.h5"),
    "CNN":       os.path.join(MODEL_DIR, "cnn_lstm_glove","CNN_GloVe.h5"),
    "CNN_LSTM":  os.path.join(MODEL_DIR, "cnn_lstm_glove","CNN_LSTM_GloVe.h5"),
}

# ─── TEXT CLEANING (must match your training) ─────────────────────────────────

def wordopt(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[“”‘’]', '', text)
    tokens = text.split()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

def normalize_label(raw: str) -> str:
    r = raw.strip().upper()
    if r in ("1", "__LABEL__1", "REAL"): return "REAL"
    if r in ("0", "__LABEL__0", "FAKE"): return "FAKE"
    raise ValueError(f"Unknown label '{raw}'")

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    # 1) load test set
    if not os.path.exists(TSV_PATH):
        print(f"❌ Test file not found: {TSV_PATH}")
        return
    df = pd.read_csv(TSV_PATH, sep="\t", dtype=str)
    if "text" not in df.columns or "label(1=real,0=fake)" not in df.columns:
        print("❌ TSV must have 'text' and 'label(1=real,0=fake)' columns.")
        return

    texts = df["text"].tolist()
    y_true = [normalize_label(l) for l in df["label(1=real,0=fake)"]]
    base, fname = os.path.split(TSV_PATH)
    stem, _ = os.path.splitext(fname)

    # 2) load tokenizer
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    # 3) clean, seq, pad once
    cleaned = [wordopt(t) for t in texts]
    seqs    = tokenizer.texts_to_sequences(cleaned)
    X_test  = pad_sequences(seqs, maxlen=MAX_SEQUENCE_LEN)

    # 4) loop models
    for name, path in MODEL_FILES.items():
        if not os.path.exists(path):
            print(f"⚠️  Model file not found: {path}, skipping {name}")
            continue

        print(f"\n🔄 Loading {name} from {path} …")
        model = load_model(path)

        # inference
        probs = model.predict(X_test, verbose=0).flatten()
        preds = ["REAL" if p > 0.5 else "FAKE" for p in probs]

        # accuracy
        acc = accuracy_score(y_true, preds)
        print(f"✅ {name:<8} — Test Accuracy: {acc*100:5.2f}%")

        # save TSV
        out = df.copy()
        out["predicted"]      = preds
        out["predicted_prob"] = probs
        out_fname = f"{stem}_{name}_GloVe_predictions.tsv"
        out_path  = os.path.join(base or ".", out_fname)
        out.to_csv(out_path, sep="\t", index=False)
        print(f"💾 Saved predictions → {out_path}")

if __name__ == "__main__":
    main()
