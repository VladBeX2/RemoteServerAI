import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import re
import string
import unicodedata
from sklearn.model_selection import train_test_split
import numpy as np
import csv
import os
import torch.nn.functional as F

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

texts = df["Statement"].tolist()
labels = df["Label"].tolist()

train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

tokenizer_path = "saved_models/roberta_tokenizer"
tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
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

test_dataset = tokenize_texts(test_texts, test_labels, tokenizer, max_length)

model_path = "saved_models/roberta_torch_model"
model = RobertaForSequenceClassification.from_pretrained(model_path)

inference_args = TrainingArguments(
    output_dir="./results_inference",
    per_device_eval_batch_size=16,
    do_predict=True,
)

trainer = Trainer(
    model=model,
    args=inference_args
)

predictions_output = trainer.predict(test_dataset)
logits = predictions_output.predictions

probabilities = F.softmax(torch.tensor(logits), dim=1).numpy()
predicted_labels = np.argmax(logits, axis=-1)

confidence_scores = [probabilities[i][predicted_labels[i]] for i in range(len(predicted_labels))]


results_df = pd.DataFrame({
    "text": test_texts,
    "true_label": test_labels,
    "predicted_label": predicted_labels,
    "confidence_score": confidence_scores
})

if trainer.is_world_process_zero():
    results_csv = "roberta_test_predictions.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"Predicțiile au fost salvate în {results_csv}")
