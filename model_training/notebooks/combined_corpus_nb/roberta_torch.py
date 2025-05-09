import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import csv
import unicodedata
import os

def wordopt(text):
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    
    text = text.replace('\n', ' ')
    
    text = unicodedata.normalize('NFKD', text)

    text = text.encode('ascii', 'ignore').decode('ascii')
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

df = pd.read_csv("../../datasets/Combined_Corpus/All.csv")
df = df[df['word_count'] >= 30]
print("Incepem curatarea textului...")
df["Statement"] = df["Statement"].apply(wordopt)

texts = df["Statement"].tolist()
labels = df["Label"].tolist()

train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
max_length = 512  

def tokenize_texts(texts, labels, tokenizer, max_length):
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    return RobertaDataset(encodings, labels)

class RobertaDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings  
        self.labels = labels        
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

print("Tokenizare...")
train_dataset = tokenize_texts(train_texts, train_labels, tokenizer, max_length)
test_dataset  = tokenize_texts(test_texts, test_labels, tokenizer, max_length)

model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), axis=-1).numpy()
    labels = labels
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="weighted", zero_division=0)
    rec = recall_score(labels, preds, average="weighted", zero_division=0)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

training_args = TrainingArguments(
    output_dir="./results_roberta",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,       
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs_roberta",
    logging_steps=50,
    learning_rate=1e-5,
    warmup_steps=500,
    load_best_model_at_end=True,
    save_total_limit=3
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)
if trainer.is_world_process_zero():
    model_path="saved_models/roberta_torch_model"
    trainer.save_model(model_path)
    tokenizer.save_pretrained("saved_models/roberta_tokenizer")

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

    with open(results_file, mode="a", newline="") as f:
        fieldnames = ["model_path", "eval_loss", "eval_accuracy", "eval_precision", "eval_recall", "eval_f1", "epoch"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()  
        writer.writerow(row)