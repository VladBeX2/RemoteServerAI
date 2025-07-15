from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from transformers import BertForSequenceClassification, BertTokenizer,RobertaTokenizer, RobertaForSequenceClassification
from lime.lime_text import LimeTextExplainer
from fastapi.responses import JSONResponse
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from newspaper import Article
import os, subprocess,signal
from typing import Optional, List, Dict
from utils import wordopt,wordopt_lite, glove_transform, load_glove_embeddings
import numpy as np
import joblib
import torch
import sys
import threading
import signal
import json
import gc

app = FastAPI()

GLOVE_PATH = "../model_training/datasets/GloVe_embeddings/glove.6B.300d.txt"
MODELS_DIR = "../model_training/notebooks/combined_corpus_nb/saved_models"
ADDITIONAL_NEWS_PATH= "additional_news.json"
BERT_TOKENIZER_PATH = os.path.join(MODELS_DIR, "bert_v2","bert_v2_tokenizer")
BERT_MODEL_PATH = os.path.join(MODELS_DIR, "bert_v2","bert_v2_torch_model")
ROBERTA_TOKENIZER_PATH = os.path.join(MODELS_DIR, "roberta_v3","roberta_v3_tokenizer")
ROBERTA_MODEL_PATH = "../model_training/notebooks/TextAttack/roberta_adv_finetuned"
EMBEDDING_DIM = 300
LOG_DIR = "retrain_logs"
os.makedirs(LOG_DIR, exist_ok=True)

device = torch.device("cpu")
device_bert = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bert_tok = BertTokenizer.from_pretrained(BERT_TOKENIZER_PATH)
bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
bert_model.to(device_bert).eval()

roberta_tok = RobertaTokenizer.from_pretrained(ROBERTA_TOKENIZER_PATH)
roberta_model = RobertaForSequenceClassification.from_pretrained(ROBERTA_MODEL_PATH)
roberta_model.to(device_bert).eval()



SCRIPTS = [
    "scripts/Basic_Alg_GloVe.py",
    "scripts/Basic_algorithms_v2.py",
    "scripts/bert_v2.py",
    "scripts/roberta_v3.py",
]

_retrain_thread: Optional[threading.Thread]  = None
_current_proc: Optional[subprocess.Popen]= None
_error_flag: bool = False
_done_flag: bool = False
_embeddings_index: Optional[Dict[str, List[float]]] = None

def get_glove_index() -> Dict[str, List[float]]:
    global _embeddings_index
    if _embeddings_index is None:
        print("Loading GloVe embeddings into memory…")
        _embeddings_index = load_glove_embeddings(GLOVE_PATH)
        print(f"Loaded {_embeddings_index and len(_embeddings_index)} tokens.")
    return _embeddings_index 

def unload_glove_index() -> None:
    global _embeddings_index
    _embeddings_index = None
    gc.collect()
    print("Freed GloVe embeddings from memory.")


def _run_queue():
    global _current_proc,_error_flag,_done_flag

    for script in SCRIPTS:
        if _error_flag:
            break

        if not os.path.exists(script):
            continue

        log_out = open(os.path.join(LOG_DIR, f"{os.path.basename(script)}.out"), "w")
        log_err = open(os.path.join(LOG_DIR, f"{os.path.basename(script)}.err"), "w")
        print(f"Starting script: {script}")
        _current_proc = subprocess.Popen(
            ["python", script],
            stdout=log_out,
            stderr=log_err,
            text=True
        )

        ret = _current_proc.wait()
        if ret != 0:
            _error_flag = True
            print(f"Error in script {script}")

            break

    _current_proc = None
    _done_flag = True

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
        text = wordopt(data.text)
        fname = f"{data.model_name}_GloVe_300d.joblib"
        model_path = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model not found.")
        model = joblib.load(model_path)
        embeddings_index = get_glove_index()
        X_vec = glove_transform([text], embeddings_index, EMBEDDING_DIM)
        pred = model.predict(X_vec)[0]
        prob = None
        if (hasattr(model, "predict_proba")):
            prob = float(model.predict_proba(X_vec)[0][pred])
        elif hasattr(model, "decision_function"):
            df = model.decision_function(X_vec)
            if df.ndim == 1:  
                scores = np.vstack([-df, df]).T
            else:
                scores = df
            exp = np.exp(scores)
            probs = exp / np.sum(exp, axis=1, keepdims=True)
            prob = float(probs[0][pred])
        unload_glove_index()
        label = "REAL" if pred == 1 else "FAKE"
        return {
            "label": label,
            "probability": prob
        }
class Example(BaseModel):
    text: str
    label: int
class RetrainData(BaseModel):
    examples: List[Example]

@app.post("/predict/retrain")
def start_retrain(payload: RetrainData):
    examples = payload.dict().get("examples")
    if not examples:
        raise HTTPException(status_code=400, detail="No examples provided.")

    global _retrain_thread, _error_flag,_done_flag

    with open(ADDITIONAL_NEWS_PATH, "w") as f:
        json.dump(examples,f)

    if _retrain_thread and _retrain_thread.is_alive():
        return {"status": "running"}
    
    _error_flag = False
    _done_flag = False
    _retrain_thread = threading.Thread(target=_run_queue,daemon=True)
    _retrain_thread.start()

    return {"status": "started"}
    
@app.get("/predict/retrain/status")
def retrain_status():
    global _retrain_thread, _error_flag,_done_flag

    if _retrain_thread is None:
        return {"status": "none"}
    
    if _retrain_thread.is_alive():
        return {"status": "running"}
    
    if _error_flag:
        _retrain_thread = None
        _error_flag = False
        return {"status": "error"}
    
    if _done_flag:
        _retrain_thread = None
        _done_flag = False
        return {"status": "success"}

    _retrain_thread = None
    return {"status": "none"}


@app.post("/predict/retrain/stop")
def stop_retrain():
    global _current_proc, _retrain_thread, _error_flag

    if not _retrain_thread or not _retrain_thread.is_alive():
        return {"status": "none"}
    
    try:
        if _current_proc and _current_proc.poll() is None:
            _current_proc.terminate()
    except:
        pass
    
    _current_proc = None
    _retrain_thread = None
    _error_flag = False
    _done_flag = False

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
            embeddings_index = get_glove_index()
            X = glove_transform(cleaned,embeddings_index,EMBEDDING_DIM)
            if hasattr(model, "predict_proba"):
                return model.predict_proba(X)
            
            if hasattr(model, "decision_function"):
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

    html_data = explanation.as_html()
    html_data = html_data.replace(
        "<body>",
        '<body style="background-color:white; color:black;">'
    )
    html_base64 = base64.b64encode(html_data.encode("utf-8")).decode("utf-8")

    return JSONResponse(content={"explanation_html": html_base64})
