import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import re
import unicodedata
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import csv
import os
import time

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
print("Dimensiunea setului de date inițial:", df.shape)
df.dropna(subset=["text", "label"], inplace=True)
print("După eliminarea valorilor nule:", df.shape)

df["text"] = df["text"].fillna("").astype(str)
df["word_count"] = df["text"].apply(lambda x: len(x.split()))
df = df[df["word_count"] >= 30]
print("După filtrarea pe numărul de cuvinte:", df.shape)

print("Aplicăm preprocesarea textului...")
df["text"] = df["text"].apply(wordopt)

texts = df["text"].tolist()
labels = df["label"].tolist()

train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
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

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), axis=-1).numpy()
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="weighted", zero_division=0)
    rec = recall_score(labels, preds, average="weighted", zero_division=0)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def get_training_args(fold):
    return TrainingArguments(
        output_dir=f"./roberta_kfold/results_roberta_fold{fold}",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"./roberta_kfold/logs_roberta_fold{fold}",
        logging_steps=50,
        learning_rate=1e-5,
        warmup_steps=500,
        load_best_model_at_end=True,
        save_total_limit=3,
    )

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_metrics = []
fold_times = []
fold_number = 1

for train_index, val_index in kf.split(train_val_texts):
    print(f"\n=== Fold {fold_number} ===")
    train_texts_fold = [train_val_texts[i] for i in train_index]
    train_labels_fold = [train_val_labels[i] for i in train_index]
    val_texts_fold = [train_val_texts[i] for i in val_index]
    val_labels_fold = [train_val_labels[i] for i in val_index]
    
    train_dataset_fold = tokenize_texts(train_texts_fold, train_labels_fold, tokenizer, max_length)
    val_dataset_fold   = tokenize_texts(val_texts_fold, val_labels_fold, tokenizer, max_length)
    
    model_fold = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
    
    training_args = get_training_args(fold_number)
    
    trainer_fold = Trainer(
        model=model_fold,
        args=training_args,
        train_dataset=train_dataset_fold,
        eval_dataset=val_dataset_fold,
        compute_metrics=compute_metrics
    )
    
    start_time = time.time()
    trainer_fold.train()
    fold_time = time.time() - start_time
    fold_times.append(fold_time)
    
    eval_results_fold = trainer_fold.evaluate()
    print(f"Fold {fold_number} evaluation: {eval_results_fold}")
    
    fold_metrics.append({
        "fold": fold_number,
        "eval_loss": eval_results_fold.get("eval_loss"),
        "eval_accuracy": eval_results_fold.get("eval_accuracy"),
        "eval_precision": eval_results_fold.get("eval_precision"),
        "eval_recall": eval_results_fold.get("eval_recall"),
        "eval_f1": eval_results_fold.get("eval_f1"),
        "train_time": fold_time
    })
    
    fold_number += 1

if Trainer.is_world_process_zero(trainer_fold):
    results_csv = "roberta_kfold/cv_performance_results.csv"
    results_df = pd.DataFrame(fold_metrics)
    results_df.to_csv(results_csv, index=False)
    print(f"\nRezultatele validării încrucișate au fost salvate în {results_csv}")

print("\nTimp mediu de antrenare pe fold:", np.mean(fold_times), "secunde")

print("\n=== Reantrenare pe întregul set train+validation și evaluare pe test ===")
final_train_dataset = tokenize_texts(train_val_texts, train_val_labels, tokenizer, max_length)
final_test_dataset  = tokenize_texts(test_texts, test_labels, tokenizer, max_length)
final_training_args = TrainingArguments(
    output_dir="./roberta_kfold/results_roberta_final",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./roberta_kfold/logs_roberta_final",
    logging_steps=50,
    learning_rate=1e-5,
    warmup_steps=500,
    load_best_model_at_end=True,
    save_total_limit=3,
)
final_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
final_trainer = Trainer(
    model=final_model,
    args=final_training_args,
    train_dataset=final_train_dataset,
    eval_dataset=final_test_dataset,
    compute_metrics=compute_metrics
)

final_trainer.train()
final_eval_results = final_trainer.evaluate()
print("Final test evaluation:", final_eval_results)

if final_trainer.is_world_process_zero():
    final_trainer.save_model("roberta_kfold/final_model")
    final_results_csv = "roberta_kfold/final_test_results.csv"
    final_results_df = pd.DataFrame([final_eval_results])
    final_results_df.to_csv(final_results_csv, index=False)
    print(f"Rezultatele finale au fost salvate în {final_results_csv}")
