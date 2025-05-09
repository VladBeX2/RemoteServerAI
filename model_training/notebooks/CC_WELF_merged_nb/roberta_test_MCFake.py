import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import re
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# Funcție simplă de preprocesare, identică cu ce ai folosit la antrenare
def wordopt(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    text = text.replace('\n', ' ')
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 1. Încarcă noul dataset (fără a filtra limba engleză sau nr. de cuvinte)
df = pd.read_csv("../../datasets/MC_Fake_dataset.csv")  # Pune calea ta
print("Dimensiunea inițială:", df.shape)

# 2. Elimină rândurile cu text/label lipsă
df.dropna(subset=["text", "labels"], inplace=True)
print("Dimensiunea după dropna:", df.shape)

# 3. Aplică funcția de preprocesare
df["text"] = df["text"].astype(str)
df["text"] = df["text"].apply(wordopt)

# Extragem textele și etichetele
texts = df["text"].tolist()
df["labels"] = 1 - df["labels"]  # Inversăm etichetele pentru a fi compatibile cu modelul
labels_true = df["labels"].tolist()

# 4. Încarcă modelul și tokenizer-ul salvate anterior
model_path = "saved_models/roberta/roberta_torch_model"
tokenizer_path = "saved_models/roberta/roberta_tokenizer"

tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)

max_length = 512

# 5. Definim un Dataset PyTorch pentru inferență
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

test_dataset = RobertaDataset(texts, labels_true, tokenizer, max_length)

# 6. Configurăm Trainer pentru inferență (nu antrenăm nimic)
inference_args = TrainingArguments(
    output_dir="./inference_results",
    per_device_eval_batch_size=16,
    do_predict=True
)

trainer = Trainer(
    model=model,
    args=inference_args
)

# 7. Facem predicții pe noul set de date
predictions_output = trainer.predict(test_dataset)
logits = predictions_output.predictions

# Obținem etichetele prezise prin argmax
labels_pred = np.argmax(logits, axis=-1)

# 8. Calculăm metricile
acc = accuracy_score(labels_true, labels_pred)
prec = precision_score(labels_true, labels_pred, average="weighted", zero_division=0)
rec = recall_score(labels_true, labels_pred, average="weighted", zero_division=0)
f1 = f1_score(labels_true, labels_pred, average="weighted", zero_division=0)
if trainer.is_world_process_zero():
    print("=== Rezultate pe noul set de date ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    # (Opțional) Vedem câte exemple sunt prezise corect
    correct_count = (labels_true == labels_pred).sum()
    print(f"Număr de înregistrări prezise corect: {correct_count} din {len(df)}")

    # (Opțional) Salvăm predicțiile într-un fișier CSV
    results_df = pd.DataFrame({
        "text": texts,
        "true_label": labels_true,
        "predicted_label": labels_pred
    })
    results_df.to_csv("MC_Fake_dataset_predictions.csv", index=False)
    print("Am salvat predicțiile în MC_Fake_dataset_predictions.csv")
