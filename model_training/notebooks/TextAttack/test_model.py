import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =====================================
# Configurații și căi
# =====================================
ADV_MODEL_DIR       = "roberta_adv_finetuned"  # directorul modelului adversarial-trained
TOKENIZER_PATH      = "../combined_corpus_nb/saved_models/roberta_v3/roberta_v3_tokenizer"
TRAIN_SPLIT_SCRIPT  = "split_dataset.py"   # asigură-te că e modulabil import-ul
DATA_PATH           = "../../datasets/Combined_Corpus/All_cleaned.csv"
TEST_AUG_PATH       = "test_augmented.csv"

LABEL_COL           = "label"
TEXT_COL            = "text"
LANG_COL            = "language"
MIN_WORDS           = 30
MAX_WORDS           = 500
RANDOM_STATE        = 42
MAX_LENGTH          = 512
BATCH_SIZE          = 32

# =====================================
# Definește dataset-ul PyTorch
# =====================================
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = int(self.labels[idx])
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

# =====================================
# Funcție de evaluare
# =====================================
def evaluate_dataset(model, tokenizer, texts, labels):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i : i + BATCH_SIZE]
            batch_labels = labels[i : i + BATCH_SIZE]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(model.device)
            outputs = model(**enc)
            batch_preds = outputs.logits.argmax(dim=-1).cpu().numpy().tolist()
            preds.extend(batch_preds)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return acc, f1

# =====================================
# Încarcă tokenizer și model adv
# =====================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
adv_model = AutoModelForSequenceClassification.from_pretrained(
    ADV_MODEL_DIR
).to(device)

# =====================================
# 1) Test pe test_augmented.csv
# =====================================
df_aug = pd.read_csv(TEST_AUG_PATH)
texts_aug = df_aug[TEXT_COL].tolist()
labels_aug = df_aug[LABEL_COL].tolist()

acc_aug, f1_aug = evaluate_dataset(adv_model, tokenizer, texts_aug, labels_aug)
print(f"Adversarial-trained model on test_augmented.csv -> Accuracy: {acc_aug:.3%}, F1: {f1_aug:.3%}")

# =====================================
# 2) Test pe clean test_df (split_dataset)
# =====================================
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(TRAIN_SPLIT_SCRIPT)))
from split_dataset import load_and_split

_, _, test_df = load_and_split(
    csv_path=DATA_PATH,
    text_col=TEXT_COL,
    label_col=LABEL_COL,
    lang_col=LANG_COL,
    min_word_count=MIN_WORDS,
    test_size=0.15,
    val_size=0.15,
    random_state=RANDOM_STATE
)

texts_clean = test_df[TEXT_COL].tolist()
labels_clean = test_df[LABEL_COL].tolist()

acc_clean, f1_clean = evaluate_dataset(adv_model, tokenizer, texts_clean, labels_clean)
print(f"Adversarial-trained model on clean test_df -> Accuracy: {acc_clean:.3%}, F1: {f1_clean:.3%}")
