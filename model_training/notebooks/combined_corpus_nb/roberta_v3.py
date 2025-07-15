import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import csv
import unicodedata
import os
import matplotlib.pyplot as plt  
import shutil

def wordopt(text):
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    text = text.replace('\n', ' ')
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

 
df = pd.read_csv("../../datasets/Combined_Corpus/All_cleaned.csv")
print(df.shape)
df["word_count"] = df["text"].apply(lambda x: len(x.split()))
before_word_count_filter = len(df)
df = df[df["word_count"] >= 30]
eliminated_word_count = before_word_count_filter - len(df)
print(f"Eliminate {eliminated_word_count} înregistrări (sub 30 cuvinte). Dimensiune nouă: {df.shape}")
df.drop(columns=["word_count"], inplace=True)
df = df[df["language"] == 'en' ]
print("După eliminarea non-EN, dimensiune:", df.shape)

print("Aplic curățarea textului...")
df["text"] = df["text"].apply(wordopt)

texts = df["text"].tolist()
labels = df["label"].tolist()

 
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.3, random_state=42
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
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="weighted", zero_division=0)
    rec = recall_score(labels, preds, average="weighted", zero_division=0)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

 
class TrainingMetricsCallback(TrainerCallback):
    def __init__(self):
        self.trainer = None
    def on_train_begin(self, args, state, control, **kwargs):
        if "trainer" in kwargs:
            self.trainer = kwargs["trainer"]
    def on_epoch_end(self, args, state, control, **kwargs):
        if self.trainer is None:
            print("Trainer is None in callback.")
            return control
         
        train_pred = self.trainer.predict(self.trainer.train_dataset)
        train_metrics = self.trainer.compute_metrics((train_pred.predictions, train_pred.label_ids))

         
        epoch = int(round(state.epoch))

         
        if "f1" in train_metrics:
            print(f"Epoch {epoch} - Train F1: {train_metrics['f1']:.4f}")
            state.log_history.append({"epoch": epoch, "train_f1": train_metrics["f1"]})
        return control

 
training_args = TrainingArguments(
    output_dir="./saved_models/roberta_v3/results_roberta_v3",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,       
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",     
    save_strategy="epoch",
    logging_dir="./saved_models/roberta_v3/logs_roberta_v3",
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

 
callback = TrainingMetricsCallback()
callback.trainer = trainer
trainer.add_callback(callback)

 
trainer.train()

 
training_history = trainer.state.log_history

 
eval_logs = [log for log in training_history if "f1" in log and "epoch" in log and "eval_accuracy" in log.keys()]
 
train_logs = [log for log in training_history if "train_f1" in log and "epoch" in log]

 
 

epochs_eval = [lg["epoch"] for lg in eval_logs]
eval_f1_scores = [lg["f1"] for lg in eval_logs]

epochs_train = [lg["epoch"] for lg in train_logs]
train_f1_scores = [lg["train_f1"] for lg in train_logs]

 
if len(epochs_train) != len(epochs_eval):
    min_len = min(len(epochs_train), len(epochs_eval))
    epochs = epochs_train[:min_len]
    train_f1_scores = train_f1_scores[:min_len]
    eval_f1_scores = eval_f1_scores[:min_len]
else:
    epochs = epochs_train   

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))

if len(epochs) == len(train_f1_scores) == len(eval_f1_scores) and len(epochs) > 0:
    plt.plot(epochs, train_f1_scores, label="Train F1", marker='o')
    plt.plot(epochs, eval_f1_scores, label="Eval F1", marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("Learning Curve (F1 Score vs. Epoch)")
    plt.legend()
    plt.grid(True)
    plt.savefig("learning_curve_f1.png")
    plt.close()
    print("Learning curve saved to 'learning_curve_f1.png'.")
else:
    print("Nu s-au putut asocia train_f1 și eval_f1 pe epoci comune sau logurile sunt goale.")

 
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

 
if trainer.is_world_process_zero():
    model_path = "saved_models/roberta_v3/roberta_v3_torch_model"
    trainer.save_model(model_path)
    tokenizer.save_pretrained("saved_models/roberta_v3/roberta_v3_tokenizer")

    results_file = "performance_results.csv"
    file_exists = os.path.exists(results_file)

    row = {
        "model_path": model_path,
        "eval_loss": eval_results.get("eval_loss"),
        "eval_accuracy": eval_results.get("eval_accuracy"),
        "eval_precision": eval_results.get("eval_precision"),
        "eval_recall": eval_results.get("eval_recall"),
        "eval_f1": eval_results.get("f1"),    
        "epoch": eval_results.get("epoch"),
    }
    with open(results_file, mode="a", newline="") as f:
        fieldnames = ["model_path", "eval_loss", "eval_accuracy", "eval_precision", "eval_recall", "eval_f1", "epoch"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()  
        writer.writerow(row)

    source_file = "roberta_v3.py"
    model_dir = "saved_models/roberta_v3"
    shutil.copy(source_file, model_dir)
    print(f"Scriptul sursă a fost copiat în {model_dir}.")
