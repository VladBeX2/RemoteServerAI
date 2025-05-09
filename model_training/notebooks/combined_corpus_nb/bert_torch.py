import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import re
import string
import os
import csv
import unicodedata

def wordopt(text):
    text = text.lower()
    
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    
    text = text.replace('\n', ' ')
    
    text = unicodedata.normalize('NFKD', text)

    text = text.encode('ascii', 'ignore').decode('ascii')
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

df = pd.read_csv("../../datasets/Combined_Corpus/All.csv")
df = df[df['word_count'] >= 30]
print("Incepem curatarea textului...")
df['Statement'] = df['Statement'].apply(wordopt)

texts = df["Statement"].tolist()
labels = df["Label"].tolist()

train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
max_length = 512
print("definitie Tokenizare...")
def tokenize_texts(texts, labels, tokenizer, max_length):
    """Tokenizează o listă de texte și întoarce un Dataset PyTorch."""
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    return NewsDataset(encodings, labels)

print("definitie Dataset...")

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings  
        self.labels = labels        
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

print("Tokenizare...")
train_dataset = tokenize_texts(train_texts, train_labels, tokenizer, max_length)
test_dataset = tokenize_texts(test_texts, test_labels, tokenizer, max_length)

print("definitie Model...")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)

print("definitie Metrici...")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()  
    preds = (probs > 0.5).astype(int)
    labels = labels.astype(int)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

print("definitie Trainer_args ...")
training_args = TrainingArguments(
    output_dir="./results",            # Directorul unde se salvează checkpoint-urile
    num_train_epochs=10,               # Mărim numărul de epoci pentru a oferi modelului mai mult timp de antrenare
    per_device_train_batch_size=16,    # Batch size pe GPU; dacă memoria permite, se poate mări
    gradient_accumulation_steps=2,       # Efectiv, batch size-ul total devine 16*2=32 pe GPU
    per_device_eval_batch_size=16,     # Batch size pentru evaluare
    evaluation_strategy="epoch",       # Evaluează la finalul fiecărei epoci; poți încerca și "steps" dacă datasetul este mare
    save_strategy="epoch",             # Salvează modelul la finalul fiecărei epoci
    logging_dir="./logs",              # Directorul pentru loguri
    logging_steps=50,                  # Log la fiecare 50 de pași
    learning_rate=1e-5,                # Rata de învățare; poți experimenta și cu valori mai mici
    warmup_steps=1000,                 # Mai mulți pași de încălzire pentru a stabiliza rata de învățare inițială
    weight_decay=0.01,                 # Un mic weight decay poate ajuta la regularizare
    load_best_model_at_end=True,       # Încarcă cel mai bun model pe setul de validare la final
    fp16=True,                         # Dacă GPU-ul suportă, folosește precizie mixtă pentru antrenare mai rapidă
    save_total_limit=3                 # Limitează numărul de checkpoint-uri salvate pentru a economisi spațiu
)


print("definitie Trainer 2 ...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

print("Antrenare...")
trainer.train()

print("Evaluare...")
eval_results = trainer.evaluate()
print("Eval Results:", eval_results)

model_path="saved_models/bert_perf_torch_model"
trainer.save_model(model_path)
tokenizer.save_pretrained("saved_models/bert_perf_torch_tokenizer")

results_file = "performance_results.csv"

file_exists = os.path.exists(results_file)

row = {
    "model_path": model_path,
    "eval_loss": eval_results.get("eval_loss"),
    "eval_accuracy": eval_results.get("eval_accuracy"),
    "eval_precision": eval_results.get("eval_precision"),
    "eval_recall": eval_results.get("eval_recall"),
    "eval_f1": eval_results.get("eval_f1"),
    "epoch": eval_results.get("epoch"),
}

if trainer.is_world_process_zero():
    with open(results_file, mode="a", newline="") as f:
        fieldnames = ["model_path", "eval_loss", "eval_accuracy", "eval_precision", "eval_recall", "eval_f1", "epoch"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader() 
        writer.writerow(row)