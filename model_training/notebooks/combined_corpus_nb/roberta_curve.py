import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import re
import unicodedata
import os
from langdetect import detect
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Subset
import matplotlib.pyplot as plt


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

# Citirea și preprocesarea datasetului
df = pd.read_csv("../../datasets/Combined_Corpus/All.csv")
print("Dimensiunea inițială a setului de date:", df.shape)

initial_count = len(df)
df.dropna(subset=["Statement", "Label"], inplace=True)
after_dropna_count = len(df)
print(f"Eliminate {initial_count - after_dropna_count} înregistrări din cauza Statementului sau label-ului nul. Noua dimensiune: {df.shape}")

non_en_count = (df["language"] != "en").sum()
df = df[df["language"] == "en"]
print(f"Eliminate {non_en_count} înregistrări deoarece Statementul nu e în engleză. Noua dimensiune: {df.shape}")


changed_wordopt = (df["Statement"] != df["Statement"].apply(wordopt)).sum()
df["Statement"] = df["Statement"].apply(wordopt)
print(f"Modificate {changed_wordopt} înregistrări prin aplicarea funcției wordopt.")

before_word_count_filter = len(df)
df = df[df["word_count"] >= 30]
eliminated_word_count = before_word_count_filter - len(df)
print(f"Eliminate {eliminated_word_count} înregistrări deoarece aveau mai puțin de 30 de cuvinte. Noua dimensiune: {df.shape}")
df.drop(columns=["word_count"], inplace=True)

# Crearea listelor cu texte și etichete
texts = df["Statement"].tolist()
labels = df["Label"].tolist()

# Împărțirea în seturi de antrenament și testare
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Inițializarea tokenizerului
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
max_length = 512 

# Funcția de tokenizare a textelor, creând un obiect de tip Dataset
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

# Definirea funcției de calcul a metricilor
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), axis=-1).numpy()
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="weighted", zero_division=0)
    rec = recall_score(labels, preds, average="weighted", zero_division=0)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

# EXPERIMENT: CURBA DE ÎNVĂȚARE
# Vom antrena modelul pe fracții din setul de antrenament și vom înregistra loss-ul de antrenare și evaluare

fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
subset_sizes = []
train_losses = []
eval_losses = []

for frac in fractions:
    subset_size = int(len(train_dataset) * frac)
    subset_indices = list(range(subset_size))
    train_subset = Subset(train_dataset, subset_indices)
    subset_sizes.append(subset_size)
    
    # Setările de antrenare pentru experiment
    training_args_subset = TrainingArguments(
        output_dir=f"./results_subset_{int(frac*100)}",
        num_train_epochs=3,                    # Folosim 3 epoci pentru rapiditate
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,       
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="no",                    # Nu salvăm modele în timpul experimentului
        logging_dir=f"./logs_subset_{int(frac*100)}",
        logging_steps=50,
        learning_rate=1e-5,
        warmup_steps=500,
        disable_tqdm=True,                     # Pentru output mai curat
        no_cuda=False                          # Setează True dacă dorești să rulezi pe CPU
    )
    
    # Reinițializăm modelul de la greutățile pre-antrenate
    model_subset = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
    
    trainer_subset = Trainer(
        model=model_subset,
        args=training_args_subset,
        train_dataset=train_subset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    
    print(f"\nAntrenare pe {subset_size} mostre (fracție {frac})...")
    train_output = trainer_subset.train()
    training_loss = train_output.training_loss
    eval_output = trainer_subset.evaluate()
    evaluation_loss = eval_output.get("eval_loss", None)
    
    train_losses.append(training_loss)
    eval_losses.append(evaluation_loss)
    
    print(f"Subset size: {subset_size} | Training Loss: {training_loss:.4f} | Eval Loss: {evaluation_loss:.4f}")

# Plotăm curba de învățare
if trainer_subset.is_world_process_zero():
    plt.figure(figsize=(8, 6))
    plt.plot(subset_sizes, train_losses, label="Training Loss", marker="o")
    plt.plot(subset_sizes, eval_losses, label="Evaluation Loss", marker="o")
    plt.xlabel("Numărul de mostre de antrenament")
    plt.ylabel("Loss")
    plt.title("Curba de învățare pentru modelul RoBERTa")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("learning_curve_roberta.png")
