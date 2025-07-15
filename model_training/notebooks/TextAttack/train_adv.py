import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

 
 
 
BASELINE_MODEL_PATH = "../combined_corpus_nb/saved_models/roberta_v3/roberta_v3_torch_model"
TOKENIZER_PATH      = "../combined_corpus_nb/saved_models/roberta_v3/roberta_v3_tokenizer"
TRAIN_AUG_PATH      = "train_augmented.csv"
VAL_AUG_PATH        = "val_augmented.csv"
OUTPUT_DIR          = "roberta_adv_finetuned_2"
RANDOM_STATE        = 42
MAX_LENGTH          = 512

 
 
 
train_df = pd.read_csv(TRAIN_AUG_PATH)
val_df   = pd.read_csv(VAL_AUG_PATH)

 
train_df = train_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
val_df   = val_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

 
 
 
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = AutoModelForSequenceClassification.from_pretrained(
    BASELINE_MODEL_PATH,
    num_labels=2
)

 
 
 
class AdvDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.texts = dataframe["text"].tolist()
        self.labels = dataframe["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item

train_dataset = AdvDataset(train_df, tokenizer, MAX_LENGTH)
val_dataset   = AdvDataset(val_df, tokenizer, MAX_LENGTH)

 
 
 
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="./logs",
    seed=RANDOM_STATE,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

 
 
 
trainer.train()

 
 
 
trainer.save_model(OUTPUT_DIR)
print(f"Adversarial fine-tuning complet. Model salvat în {OUTPUT_DIR}")
