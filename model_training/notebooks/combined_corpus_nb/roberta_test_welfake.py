import os
import re
import unicodedata
import pandas as pd
import numpy as np
import torch
from langdetect import detect
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def is_english(text):
    try:
        return detect(text) == 'en'
    except Exception:
        return False

def wordopt(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    
    text = text.replace('\n', ' ')
    
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

df = pd.read_csv("../../datasets/WELFake_Dataset.csv")
#df = df_all.sample(frac=0.1,random_state=42)
print("Dimensiunea inițială a setului de date:", df.shape)

df.dropna(subset=["text", "label"], inplace=True)
print("După eliminarea valorilor nule:", df.shape)

mask = df["text"].apply(is_english)
df = df[mask]
print("După filtrarea limbii engleze:", df.shape)

df["text"] = df["text"].fillna("").astype(str)
df["text"] = df["text"].apply(wordopt)


df["word_count"] = df["text"].apply(lambda x: len(x.split()))
df = df[df["word_count"] >= 30]
print("După eliminarea textelor cu <30 cuvinte:", df.shape)
df['label'] = df['label'].apply(lambda x: 1 - x)

texts = df["text"].tolist()
labels = df["label"].tolist()

model_path = "saved_models/roberta_torch_model"
tokenizer_path = "saved_models/roberta_tokenizer"

model = RobertaForSequenceClassification.from_pretrained(model_path)
#model.to("cpu")
tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)

max_length = 512

class RobertaDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.labels = labels
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

test_dataset = RobertaDataset(texts, labels, tokenizer, max_length)

inference_args = TrainingArguments(
    output_dir="./roberta_teston_WELFake_results",
    per_device_eval_batch_size=16,
    do_predict=True,
    #no_cuda=True
)
trainer = Trainer(
    model=model,
    args=inference_args
)

predictions_output = trainer.predict(test_dataset)
logits = predictions_output.predictions
predicted_labels = np.argmax(logits, axis=-1)

acc = accuracy_score(labels, predicted_labels)
prec = precision_score(labels, predicted_labels, average="weighted", zero_division=0)
rec = recall_score(labels, predicted_labels, average="weighted", zero_division=0)
f1 = f1_score(labels, predicted_labels, average="weighted", zero_division=0)


if trainer.is_world_process_zero():
    with open("rezultate_roberta_welfake.txt", "w") as f:
        f.write("=== Rezultate pe noul set de date ===\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n")
    results_df = pd.DataFrame({
        "text": texts,
        "true_label": labels,
        "predicted_label": predicted_labels
    })
    results_df.to_csv("roberta_teston_WELFake_predictions.csv", index=False)
    print("Am salvat predicțiile în roberta_teston_WELFake_predictions.csv")
