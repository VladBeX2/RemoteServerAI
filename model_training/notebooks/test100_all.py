#!/usr/bin/env python3
# evaluate_all_models.py

import os
import sys
import re
import torch
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification
)

# ─── CONFIG ────────────────────────────────────────────────────────────────────

TSV_PATH          = "../datasets/teste100.tsv"
MODELS_DIR        = "combined_corpus_nb/saved_models"
GLOVE_PATH        = "../datasets/GloVe_embeddings/glove.6B.300d.txt"
BERT_SUBFOLDER    = "bert_v2"
ROBERTA_SUBFOLDER = "roberta_v3"
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── UTILITIES ─────────────────────────────────────────────────────────────────

def normalize_label(raw: str) -> str:
    lab = raw.strip().upper()
    if lab in ("1","__LABEL__1","REAL"): return "REAL"
    if lab in ("0","__LABEL__0","FAKE"): return "FAKE"
    raise ValueError(f"Unrecognized label '{raw}'")

def wordopt(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def wordopt_lite(text: str) -> str:
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    text = text.replace('\n', ' ')
    text = text.encode('ascii','ignore').decode('ascii')
    return re.sub(r'\s+', ' ', text).strip()

def load_glove_embeddings(path, dim=300):
    emb = {}
    print("🕐 Loading GloVe…", end="", flush=True)
    with open(path, encoding="utf8") as f:
        for line in f:
            vals = line.split()
            emb[vals[0]] = np.asarray(vals[1:], dtype=np.float32)
    print(f" done: {len(emb)} tokens.")
    return emb

def glove_transform(texts, emb_index, dim=300):
    X = []
    for t in texts:
        toks = t.split()
        vecs = [emb_index[w] for w in toks if w in emb_index]
        X.append(np.mean(vecs,axis=0) if vecs else np.zeros(dim, dtype=np.float32))
    return np.vstack(X)

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    # load test
    if not os.path.exists(TSV_PATH):
        print(f"❌ Missing test file: {TSV_PATH}"); sys.exit(1)
    df = pd.read_csv(TSV_PATH, sep="\t", dtype=str)
    if "text" not in df or "label(1=real,0=fake)" not in df:
        print("❌ TSV needs 'text' & 'label(1=real,0=fake)'"); sys.exit(1)

    texts = df["text"].tolist()
    y_true = [normalize_label(l) for l in df["label(1=real,0=fake)"]]
    base, fname = os.path.split(TSV_PATH)
    stem, _    = os.path.splitext(fname)

    # preload GloVe
    glove_idx = load_glove_embeddings(GLOVE_PATH, dim=300)

    # load transformers
    print("🔄 Loading BERT…")
    bert_tok   = BertTokenizer.from_pretrained(os.path.join(MODELS_DIR,BERT_SUBFOLDER,"bert_v2_tokenizer"))
    bert_mod   = BertForSequenceClassification.from_pretrained(os.path.join(MODELS_DIR,BERT_SUBFOLDER,"bert_v2_torch_model")).to(DEVICE).eval()

    print("🔄 Loading RoBERTa…")
    rob_tok    = RobertaTokenizer.from_pretrained(os.path.join(MODELS_DIR,ROBERTA_SUBFOLDER,"roberta_v3_tokenizer"))
    rob_mod    = RobertaForSequenceClassification.from_pretrained(os.path.join(MODELS_DIR,ROBERTA_SUBFOLDER,"roberta_v3_torch_model")).to(DEVICE).eval()

    # collect configs
    to_eval = [
      {"name":"bert_v2",    "kind":"transformer","tok":bert_tok, "mod":bert_mod, "prep":wordopt_lite},
      {"name":"roberta_v3", "kind":"transformer","tok":rob_tok,  "mod":rob_mod,  "prep":wordopt_lite},
    ]

    # now pick up **all** .joblib pipelines
    for fn in os.listdir(MODELS_DIR):
        if not fn.endswith(".joblib"): continue
        mdl = joblib.load(os.path.join(MODELS_DIR,fn))
        nm  = os.path.splitext(fn)[0]
        kind = "glove" if "GloVe" in nm else "pipeline"
        to_eval.append({"name":nm, "kind":kind, "mod":mdl, "prep":wordopt})

    # run
    for cfg in to_eval:
        preds, probs = [], []
        print(f"\n▶️  Running {cfg['name']} …")

        if cfg["kind"]=="transformer":
            for txt in tqdm(texts, desc=cfg["name"]):
                clean = cfg["prep"](txt)
                toks  = cfg["tok"](clean, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
                toks  = {k:v.to(DEVICE) for k,v in toks.items()}
                with torch.no_grad():
                    out = cfg["mod"](**toks).logits[0]
                    ps  = torch.softmax(out, dim=-1)
                    idx = ps.argmax().item()
                    preds.append("REAL" if idx==1 else "FAKE")
                    probs.append(ps[idx].item())

        elif cfg["kind"]=="glove":
            cleaned = [cfg["prep"](t) for t in texts]
            X = glove_transform(cleaned, glove_idx, dim=300)
            raw = cfg["mod"].predict(X)
            preds = ["REAL" if r==1 else "FAKE" for r in raw]
            if hasattr(cfg["mod"], "predict_proba"):
                pp = cfg["mod"].predict_proba(X)
                probs = [pp[i,r] for i,r in enumerate(raw)]
            else:
                probs = [None]*len(raw)

        else:  # sklearn pipeline (BOW / TFIDF / whatever)
            cleaned = [cfg["prep"](t) for t in texts]
            raw = cfg["mod"].predict(cleaned)
            preds = ["REAL" if r==1 else "FAKE" for r in raw]
            if hasattr(cfg["mod"], "predict_proba"):
                pp = cfg["mod"].predict_proba(cleaned)
                probs = [pp[i,r] for i,r in enumerate(raw)]
            else:
                probs = [None]*len(raw)

        acc = accuracy_score(y_true, preds)
        print(f"✅ {cfg['name']:<30} Accuracy = {acc*100:.2f}%")

        out = df.copy()
        out["predicted"]      = preds
        out["predicted_prob"] = probs
        out_name = f"{stem}_{cfg['name']}_predictions.tsv"
        out_path = os.path.join(base or ".", out_name)
        out.to_csv(out_path, sep="\t", index=False)
        print(f"💾 Saved to {out_path}")

if __name__=="__main__":
    main()
