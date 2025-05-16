from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from transformers import BertForSequenceClassification, BertTokenizer,RobertaTokenizer, RobertaForSequenceClassification
from lime.lime_text import LimeTextExplainer
import tensorflow as tf
from fastapi.responses import JSONResponse
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from newspaper import Article
import os, subprocess,signal
from typing import Optional
from utils import wordopt,wordopt_lite, glove_transform, load_glove_embeddings
import numpy as np
import joblib
import torch

app = FastAPI()

GLOVE_PATH = "../model_training/datasets/GloVe_embeddings/glove.6B.300d.txt"
MODELS_DIR = "../model_training/notebooks/combined_corpus_nb/saved_models"
BERT_TOKENIZER_PATH = os.path.join(MODELS_DIR, "bert_v2","bert_v2_tokenizer")
BERT_MODEL_PATH = os.path.join(MODELS_DIR, "bert_v2","bert_v2_torch_model")
ROBERTA_TOKENIZER_PATH = os.path.join(MODELS_DIR, "roberta_v3","roberta_v3_tokenizer")
ROBERTA_MODEL_PATH = os.path.join(MODELS_DIR, "roberta_v3","roberta_v3_torch_model")
EMBEDDING_DIM = 300

device = torch.device("cpu")
device_bert = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bert_tok = BertTokenizer.from_pretrained(BERT_TOKENIZER_PATH)
bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
bert_model.to(device_bert).eval()

roberta_tok = RobertaTokenizer.from_pretrained(ROBERTA_TOKENIZER_PATH)
roberta_model = RobertaForSequenceClassification.from_pretrained(ROBERTA_MODEL_PATH)
roberta_model.to(device_bert).eval()

try:
    embeddings_index = load_glove_embeddings(GLOVE_PATH)
    print("✅ GloVe embeddings loaded successfully.")
except Exception as e:
    print("❌ Failed to load GloVe embeddings:", e)
    embeddings_index = None

_retrain_proc: Optional[subprocess.Popen] = None

class InputData(BaseModel):
    text: str
    model_name: str

@app.post("/predict")
def predict(data: InputData):
    if data.model_name == "BERT":
        text = wordopt_lite(data.text)
        toks = bert_tok(
            text,   
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        toks = {k: v.to(device_bert) for k, v in toks.items()}
        with torch.no_grad():
            out   = bert_model(**toks)
            logits = out.logits[0]
            probs  = torch.softmax(logits, dim=-1)

        pred  = torch.argmax(probs).item()
        prob  = probs[pred].item()
        label = "REAL" if pred == 1 else "FAKE"

        return {"label": label, "probability": prob}
    
    elif data.model_name == "ROBERTA":
        text = wordopt_lite(data.text)
        toks = roberta_tok(
            text,   
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        toks = {k: v.to(device_bert) for k, v in toks.items()}
        with torch.no_grad():
            out   = roberta_model(**toks)
            logits = out.logits[0]
            probs  = torch.softmax(logits, dim=-1)

        pred  = torch.argmax(probs).item()
        prob  = probs[pred].item()
        label = "REAL" if pred == 1 else "FAKE"

        return {"label": label, "probability": prob}

    else:
        try:
            text = wordopt(data.text)
            fname = f"{data.model_name}_GloVe_300d.joblib"
            model_path = os.path.join(MODELS_DIR, fname)
            model = joblib.load(model_path)
        except:
            raise HTTPException(status_code=500, detail="Failed to load model.")
        
        X_vec  = glove_transform([text],embeddings_index,EMBEDDING_DIM)
        pred = model.predict(X_vec)[0]
        prob = None
        if (hasattr(model, "predict_proba")):
            prob = float(model.predict_proba(X_vec)[0][pred])

        label = "REAL" if pred == 1 else "FAKE"
        return {
            "label": label,
            "probability": prob
        }



@app.post("/predict/retrain")
def start_retrain():
    global _retrain_proc
    # if already running, just return status
    if _retrain_proc and _retrain_proc.poll() is None:
        return {"status": "running"}

    # otherwise kick off the job
    try:
        _retrain_proc = subprocess.Popen(
            ["python", "train.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start retrain: {e}")

    return {"status": "started"}
    
@app.get("/predict/retrain/status")
def retrain_status():
    global _retrain_proc
    if not _retrain_proc:
        return {"status": "none"}

    code = _retrain_proc.poll()
    if code is None:
        # still running
        return {"status": "running"}

    # process finished (code ≥ 0)
    out, err = _retrain_proc.communicate()
    status = "success" if code == 0 else "failed"
    # clear handle so next /retrain can restart
    _retrain_proc = None
    return {"status": status}


@app.post("/predict/retrain/stop")
def stop_retrain():
    global _retrain_proc
    if not _retrain_proc or _retrain_proc.poll() is not None:
        return {"status": "none"}

    try:
        os.kill(_retrain_proc.pid, signal.SIGTERM)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not stop process: {e}")
    _retrain_proc = None
    return {"status": "stopped"}

@app.post("/predict/explain")
def explain_with_lime(data: InputData):
    print("REceived explanaition request")
    text = data.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text received.")

    if data.model_name == "BERT":
        def classifier_fn(texts:list[str]) -> np.ndarray:
            cleaned = [wordopt_lite(t) for t in texts]
            toks = bert_tok(
                cleaned,   
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            toks = {k: v.to(device_bert) for k, v in toks.items()}
            with torch.no_grad():
                out = bert_model(**toks)
                probs = torch.softmax(out.logits, dim=-1)
            return probs.cpu().numpy()
    elif data.model_name == "ROBERTA":
        def classifier_fn(texts:list[str]) -> np.ndarray:
            cleaned = [wordopt_lite(t) for t in texts]
            toks = roberta_tok(
                cleaned,   
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            toks = {k: v.to(device_bert) for k, v in toks.items()}
            with torch.no_grad():
                out = roberta_model(**toks)
                probs = torch.softmax(out.logits, dim=-1)
            return probs.cpu().numpy()
    else:
        cleaned= wordopt(data.text)
        fname = f"{data.model_name}_GloVe_300d.joblib"
        model_path = os.path.join(MODELS_DIR, fname)
        model = joblib.load(model_path)
        def classifier_fn(texts:list[str]) -> np.ndarray:
            cleaned = [wordopt(t) for t in texts]
            X = glove_transform(cleaned,embeddings_index,EMBEDDING_DIM)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)
            
            df = model.decision_function(X)
            scores = np.vstack([-df,df]).T
            exp = np.exp(scores)
            return exp / np.sum(exp, axis=1, keepdims=True)

    explainer = LimeTextExplainer(class_names=["Fake", "Real"])
    explanation = explainer.explain_instance(
        text,
        classifier_fn,
        num_features=10,
        num_samples=500
    )

    # Generăm HTML și îl stilizăm
    html_data = explanation.as_html()
    html_data = html_data.replace(
        "<body>",
        '<body style="background-color:white; color:black;">'
    )
    html_base64 = base64.b64encode(html_data.encode("utf-8")).decode("utf-8")

    return JSONResponse(content={"explanation_html": html_base64})
