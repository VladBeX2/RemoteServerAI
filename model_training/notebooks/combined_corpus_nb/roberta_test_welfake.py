import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import pandas as pd
import numpy as np
import json
import csv
import re
from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def wordopt(text):
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    text = text.replace('\n', ' ')
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load model and tokenizer
model = RobertaForSequenceClassification.from_pretrained("../TextAttack/roberta_adv_finetuned").to(device)
tokenizer = RobertaTokenizer.from_pretrained("saved_models/roberta_v3/roberta_v3_tokenizer")
model.eval()

# Load and preprocess dataset
df = pd.read_csv("../../datasets/WELFake_cleaned.csv")
df["text"] = df["text"].apply(wordopt)
texts = df["text"].tolist()
true_labels = 1 - df["label"].values  # invertim etichetele
indices = df["Unnamed: 0"].tolist()

# Batch processing
batch_size = 64
predictions = []
confidences = []
wrong_data = []

for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    batch_indices = indices[i:i+batch_size]
    batch_labels = true_labels[i:i+batch_size]

    encodings = tokenizer(batch_texts, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()
        probs = probs.cpu().numpy()

    predictions.extend(preds)
    confidences.extend(probs)

    for j, (p, t) in enumerate(zip(preds, batch_labels)):
        if p != t:
            wrong_data.append({
                "Unnamed: 0": batch_indices[j],
                "confidence": float(probs[j][p])
            })

# Metrics
report = classification_report(true_labels, predictions, output_dict=True)

# Save report
with open("inference_report_adv.json", "w") as f:
    json.dump(report, f, indent=4)

# Save misclassified
with open("wrong_predictions_adv.csv", "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["Unnamed: 0", "confidence"])
    writer.writeheader()
    writer.writerows(wrong_data)
