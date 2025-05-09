import os
import re
import unicodedata
import pandas as pd
import torch
import numpy as np
import csv
import matplotlib.pyplot as plt
import shutil
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
from torch.nn import CrossEntropyLoss

# ---------------------------
# 1. Funcție de preprocesare
# ---------------------------
def wordopt(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    text = text.replace('\n', ' ')
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ------------------------------------
# 2. Încarcă datasetul combinat și filtrează
# ------------------------------------
df = pd.read_csv("../../datasets/CC_MC_WELF_merged.csv")
print("Dimensiunea inițială a datasetului:", df.shape)

df.dropna(subset=["text", "label"], inplace=True)
print("După dropna:", df.shape)

df = df[df["language"] == "en"]
print("După filtrarea limbii engleze:", df.shape)

print("Aplic curățarea textului...")
df["text"] = df["text"].apply(wordopt)

# ------------------------------
# 3. Împărțirea datasetului pe surse
# ------------------------------
sources = df["source"].unique()
print("Sursele disponibile:", sources)

train_dfs = []
test_dfs = {}  # dict pentru seturile de test separate

for src in sources:
    df_src = df[df["source"] == src]
    train_src, test_src = train_test_split(
        df_src, test_size=0.2, random_state=42, stratify=df_src["label"]
    )
    train_dfs.append(train_src)
    test_dfs[src] = test_src
    print(f"Sursa {src}: Train: {train_src.shape}, Test: {test_src.shape}")

# Combinăm toate seturile de train
train_df_combined = pd.concat(train_dfs, ignore_index=True)
print("Dimensiunea setului de train combinat:", train_df_combined.shape)

# Combinăm și toate seturile de test
combined_test_df = pd.concat(list(test_dfs.values()), ignore_index=True)
print("Dimensiunea setului de test combinat:", combined_test_df.shape)

train_texts = train_df_combined["text"].tolist()
train_labels = train_df_combined["label"].tolist()
test_texts = combined_test_df["text"].tolist()
test_labels = combined_test_df["label"].tolist()

# ------------------------------
# 4. Setare tokenizer și Dataset
# ------------------------------
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
max_length = 512

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

def tokenize_texts(texts, labels, tokenizer, max_length):
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    return RobertaDataset(encodings, labels)

print("Tokenizare pentru setul de train combinat...")
train_dataset = tokenize_texts(train_texts, train_labels, tokenizer, max_length)
print("Tokenizare pentru setul de test combinat...")
test_dataset = tokenize_texts(test_texts, test_labels, tokenizer, max_length)

# ------------------------------
# 5. Calcularea ponderilor claselor
# ------------------------------
train_labels_np = np.array(train_labels)
classes = np.unique(train_labels_np)
class_weights = compute_class_weight('balanced', classes=classes, y=train_labels_np)
class_weights = torch.tensor(class_weights, dtype=torch.float)
print("Ponderile claselor:", class_weights)

# ------------------------------
# 6. Definirea Trainer-ului personalizat pentru weighted loss
# ------------------------------
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device)
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ------------------------------
# 7. Callback pentru calcularea metricalor pe setul de train la finalul fiecărei epoci
# ------------------------------
class TrainingMetricsCallback(TrainerCallback):
    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        # Utilizează self.trainer pentru a face predict pe setul de train
        train_pred = self.trainer.predict(self.trainer.train_dataset)
        train_metrics = self.trainer.compute_metrics((train_pred.predictions, train_pred.label_ids))
        print(f"Epoch {state.epoch} - Training metrics: {train_metrics}")
        # Adaugă metricile pe setul de train în log_history
        state.log_history.append({**{"epoch": state.epoch}, **{f"train_{k}": v for k, v in train_metrics.items()}})
        return control

# ------------------------------
# 8. Setări de antrenare
# ------------------------------
training_args = TrainingArguments(
    output_dir="./saved_models/roberta/results_combined",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,       
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",  # Salvează un checkpoint după fiecare epocă
    logging_dir="./saved_models/roberta/logs_combined",
    logging_steps=50,
    learning_rate=1e-5,
    warmup_steps=500,
    load_best_model_at_end=True,
    # Am eliminat save_total_limit pentru a păstra toate checkpoint-urile
    fp16=True
)

# ------------------------------
# 9. Inițializarea modelului și Trainer-ului
# ------------------------------
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), axis=-1).numpy()
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="weighted", zero_division=0)
    rec = recall_score(labels, preds, average="weighted", zero_division=0)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

trainer = WeightedTrainer(
    class_weights=class_weights,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Adăugăm callback-ul și setăm referința la trainer
callback = TrainingMetricsCallback()
trainer.add_callback(callback)
callback.set_trainer(trainer)

# ------------------------------
# 10. Antrenare și colectare learning curve (F1 score pe train și eval vs. epoch)
# ------------------------------
print("Încep antrenamentul...")
start_train = time.time()
trainer.train()
train_duration = time.time() - start_train
print(f"Antrenamentul a durat {train_duration:.2f} secunde.")

training_history = trainer.state.log_history

epochs_train = []
train_f1_scores = []
epochs_eval = []
eval_f1_scores = []

for log in training_history:
    if "train_f1" in log and "epoch" in log:
        epochs_train.append(log["epoch"])
        train_f1_scores.append(log["train_f1"])
    if "eval_f1" in log and "epoch" in log:
        epochs_eval.append(log["epoch"])
        eval_f1_scores.append(log["eval_f1"])

plt.figure(figsize=(8, 6))
plt.plot(epochs_train, train_f1_scores, label="Train F1 Score", marker='o')
plt.plot(epochs_eval, eval_f1_scores, label="Eval F1 Score", marker='x')
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("Learning Curve (F1 Score vs. Epoch)")
plt.legend()
plt.grid(True)
plt.savefig("learning_curve_f1.png")
plt.close()
print("Learning curve salvată ca 'learning_curve_f1.png'.")

# ------------------------------
# 11. Evaluare pe fiecare sursă separat
# ------------------------------
for src, df_test in test_dfs.items():
    print(f"\n=== Evaluare pentru sursa {src} ===")
    test_texts_src = df_test["text"].tolist()
    test_labels_src = df_test["label"].tolist()
    test_dataset_src = tokenize_texts(test_texts_src, test_labels_src, tokenizer, max_length)
    
    eval_results_src = trainer.evaluate(eval_dataset=test_dataset_src)
    print(f"Rezultate pentru sursa {src}: {eval_results_src}")
    
    results_file = f"performance_results_{src}.csv"
    file_exists = os.path.exists(results_file)
    row = {
        "source": src,
        "eval_loss": eval_results_src.get("eval_loss"),
        "eval_accuracy": eval_results_src.get("eval_accuracy"),
        "eval_precision": eval_results_src.get("eval_precision"),
        "eval_recall": eval_results_src.get("eval_recall"),
        "eval_f1": eval_results_src.get("eval_f1"),
        "epoch": eval_results_src.get("epoch"),
    }
    with open(results_file, mode="a", newline="") as f:
        fieldnames = ["source", "eval_loss", "eval_accuracy", "eval_precision", "eval_recall", "eval_f1", "epoch"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# ------------------------------
# 12. Salvare model și tokenizer
# ------------------------------
if trainer.is_world_process_zero():
    final_model_path = "saved_models/roberta/roberta_torch_model"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained("saved_models/roberta/roberta_tokenizer")
    print(f"Modelul a fost salvat în {final_model_path}.")

    source_file = os.path.basename(__file__) if '__file__' in globals() else "script.py"
    model_dir = "saved_models/roberta"
    shutil.copy(source_file, model_dir)
    print(f"Scriptul sursă a fost copiat în {model_dir}.")
