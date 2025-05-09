import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
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

df = pd.read_csv("../../datasets/CC_WELF_merged_cleaned.csv")
print(df.shape)
df["word_count"] = df["text"].apply(lambda x: len(x.split()))
before_word_count_filter = len(df)
df = df[df["word_count"] >= 30]
eliminated_word_count = before_word_count_filter - len(df)
print(f"Eliminate {eliminated_word_count} înregistrări deoarece aveau mai puțin de 30 de cuvinte. Noua dimensiune: {df.shape}")
df.drop(columns=["word_count"], inplace=True)
df=df[df["language"] == 'en' ]
print("Dupa eliminarea Statements care nu sunt in limba engleza", df.shape)

print("Incepem curatarea textului...")
df["text"] = df["text"].apply(wordopt)

texts = df["text"].tolist()
labels = df["label"].tolist()

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
    output_dir="./saved_models/roberta/results_roberta",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,       
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./saved_models/roberta/logs_roberta",
    logging_steps=50,
    learning_rate=1e-5,
    warmup_steps=500,
    load_best_model_at_end=True,
    save_total_limit=3,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

# Dupa antrenament, extragem istoricul de log si generam curba de invatare.
# training_history = trainer.state.log_history

# train_losses = []
# eval_losses = []
# epochs_train = []
# epochs_eval = []

# Parcurgem log-urile salvate de Trainer.
# for log in training_history:
#     # Cand apare "loss" inseamna ca e vorba de training loss la sfarsitul unui batch/step/epoch
#     if "loss" in log and "epoch" in log:
#         train_losses.append(log["loss"])
#         epochs_train.append(log["epoch"])
#     # Cand apare "eval_loss" inseamna ca e vorba de loss pe setul de validare
#     if "eval_loss" in log and "epoch" in log:
#         eval_losses.append(log["eval_loss"])
#         epochs_eval.append(log["epoch"])

# Facem un grafic cu training loss si eval loss.
# plt.figure(figsize=(8, 6))
# plt.plot(epochs_train, train_losses, label="Training Loss", marker='o')
# plt.plot(epochs_eval, eval_losses, label="Validation Loss", marker='x')
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Learning Curve (Loss vs. Epoch)")
# plt.legend()
# plt.grid(True)
# plt.savefig("roberta_v2_learning_curve.png")  # Salveaza graficul local
# plt.close()  # Inchide figura, util daca rulezi in medii tip notebook

eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)
if trainer.is_world_process_zero():
    model_path="saved_models/roberta/roberta_torch_model"
    trainer.save_model(model_path)
    tokenizer.save_pretrained("saved_models/roberta/roberta_tokenizer")

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

    source_file = "roberta.py"
    model_dir = "saved_models/roberta"
    shutil.copy(source_file, model_dir)