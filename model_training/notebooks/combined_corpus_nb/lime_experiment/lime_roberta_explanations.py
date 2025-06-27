import os
import re
import torch
import pandas as pd
import numpy as np
import base64
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from lime.lime_text import LimeTextExplainer
from torch.nn.functional import softmax

# ====================== CONFIG ======================
MODEL_PATH = "../../TextAttack/roberta_adv_finetuned"
TOKENIZER_PATH = "../saved_models/roberta_v3/roberta_v3_tokenizer"
MISCLASSIFIED_PATH = "roberta_misclassified.csv"
OUTPUT_PATH = "lime_explanations.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 512

# ====================== FUNCȚII ======================
def wordopt(text):
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    text = text.replace('\n', ' ')
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def classifier_fn(texts):
    cleaned = [wordopt(t) for t in texts]
    toks = tokenizer(
        cleaned,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    toks = {k: v.to(DEVICE) for k, v in toks.items()}
    with torch.no_grad():
        out = model(**toks)
        probs = softmax(out.logits, dim=-1)
    return probs.cpu().numpy()

# ====================== ÎNCĂRCARE MODEL ======================
tokenizer = RobertaTokenizer.from_pretrained(TOKENIZER_PATH)
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE).eval()

# ====================== ÎNCĂRCARE ARTICOLE GREȘITE ======================
df = pd.read_csv(MISCLASSIFIED_PATH)
texts = df["text"].tolist()
true_labels = df["correct_label"].tolist()
predicted_labels = df["predicted_label"].tolist()

# ====================== EXPLICAȚII LIME ======================
explainer = LimeTextExplainer(class_names=["Fake", "Real"])
lime_results = []

for i, text in enumerate(tqdm(texts, desc="Procesare articole greșite")):
    explanation = explainer.explain_instance(
        wordopt(text),
        classifier_fn,
        num_features=10,
        num_samples=1000
    )
    contributions = explanation.as_list()
    lime_results.append({
        "text": text,
        "true_label": true_labels[i],
        "predicted_label": predicted_labels[i],
        "contributions": contributions
    })

# ====================== SALVARE CSV ======================
lime_df = pd.DataFrame(lime_results)
lime_df.to_csv(OUTPUT_PATH, index=False)
print(f"Rezultatele LIME au fost salvate în '{OUTPUT_PATH}'")
