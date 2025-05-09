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

df = pd.read_csv("../../datasets/Combined_Corpus/All.csv")
#df = df_all.sample(frac=0.1,random_state=42)
print("Dimensiunea inițială a setului de date:", df.shape)

df.dropna(subset=["Statement", "Label"], inplace=True)
print("După eliminarea valorilor nule:", df.shape)

#mask = df["Statement"].apply(is_english)
#df = df[mask]
#print("După filtrarea limbii engleze:", df.shape)

df["Statement"] = df["Statement"].fillna("").astype(str)
df["Statement"] = df["Statement"].apply(wordopt)


df = df[df["word_count"] >= 30]
print("După eliminarea Statementelor cu <30 cuvinte:", df.shape)
df['Label'] = df['Label'].apply(lambda x: 1 - x)

texts = df["Statement"].tolist()
labels = df["Label"].tolist()

model_path = "saved_models/roberta2_torch_model"
tokenizer_path = "saved_models/roberta2_tokenizer"

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
    output_dir="./roberta2_testOn_CombinedCorpus_results",
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

print("=== Rezultate pe noul set de date ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}")

if trainer.is_world_process_zero():
    results_df = pd.DataFrame({
        "text": texts,
        "true_label": labels,
        "predicted_label": predicted_labels
    })
    results_df.to_csv("roberta2_testOn_CombinedCorpus_predictions.csv", index=False)
    print("Am salvat predicțiile în roberta2_testOn_CombinedCorpus_predictions.csv")
