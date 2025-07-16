#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import re
import string
import numpy as np
import pandas as pd
import fasttext
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Conv1D, GlobalMaxPooling1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, Callback

DATA_PATH       = "../../datasets/Combined_Corpus/All_cleaned.csv"
SAVE_ROOT       = "saved_models/fasttext_supervised"
FT_SUP_MODEL    = "fasttext_supervised_model.bin"
EMBEDDING_DIM   = 300
MAX_NUM_WORDS   = 20000
MAX_SEQ_LEN     = 100
BATCH_SIZE      = 128
EPOCHS          = 20
RND_SEED        = 42
VAL_RATIO       = 0.15
TEST_RATIO      = 0.15

os.makedirs(SAVE_ROOT, exist_ok=True)
np.random.seed(RND_SEED)
tf.random.set_seed(RND_SEED)

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["text"])
if "language" in df.columns:
    df = df[df["language"] == "en"]
if "word_count" in df.columns:
    df = df[df["word_count"] >= 30]

stop_words = set(stopwords.words("english"))
lemmatizer  = WordNetLemmatizer()
def clean_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r'<[^>]+>', '', t)
    t = re.sub(r'http\S+|www\S+', '', t)
    t = re.sub(r'\S+@\S+', '', t)
    t = re.sub(r'\d+', '', t)
    t = t.translate(str.maketrans("", "", string.punctuation))
    t = re.sub(r'\s+', ' ', t).strip()
    toks = [lemmatizer.lemmatize(w) for w in t.split() if w not in stop_words]
    return " ".join(toks)

print("Cleaning text…")
df["clean"] = df["text"].astype(str).map(clean_text)
y = df["label"].astype(int).values
texts = df["clean"].tolist()

print("Loading supervised FastText model…")
ft_sup = fasttext.load_model(FT_SUP_MODEL)

EMBEDDING_DIM = ft_sup.get_dimension()
print("FastText embedding dim =", EMBEDDING_DIM)

tok = Tokenizer(num_words=MAX_NUM_WORDS)
tok.fit_on_texts(texts)
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

print("💾 Saving tokenizer as JSON...")
tokenizer_json = tok.to_json()
tokenizer_path = os.path.join(SAVE_ROOT, "tokenizer.json")
with open(tokenizer_path, "w", encoding="utf-8") as f:
    f.write(tokenizer_json)
print(f"✅ Tokenizer saved to {tokenizer_path}")
sequences = tok.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=MAX_SEQ_LEN)

word_index = tok.word_index
num_words  = min(MAX_NUM_WORDS, len(word_index) + 1)
emb_matrix = np.zeros((num_words, EMBEDDING_DIM), dtype="float32")

for w, i in word_index.items():
    if i >= num_words: 
        continue
    emb_matrix[i] = ft_sup.get_word_vector(w)

test_size = TEST_RATIO
X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y, test_size=test_size, random_state=RND_SEED, stratify=y)

val_size = VAL_RATIO / (1 - TEST_RATIO)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=val_size, random_state=RND_SEED, stratify=y_tmp)

print(f"Splits → train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

class TestEvalCallback(Callback):
    def __init__(self, test_data):
        super().__init__()
        self.test_x, self.test_y = test_data
        self.test_losses = []
        self.test_accs   = []

    def on_epoch_end(self, epoch, logs=None):
        loss, acc = self.model.evaluate(self.test_x, self.test_y, verbose=0)
        self.test_losses.append(loss)
        self.test_accs.append(acc)

def build_lstm():
    m = Sequential([
        Embedding(num_words, EMBEDDING_DIM, weights=[emb_matrix],
                  input_length=MAX_SEQ_LEN, trainable=False),
        LSTM(128, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation="sigmoid")
    ])
    m.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return m

def build_cnn():
    m = Sequential([
        Embedding(num_words, EMBEDDING_DIM, weights=[emb_matrix],
                  input_length=MAX_SEQ_LEN, trainable=False),
        Conv1D(128, 5, activation="relu"),
        GlobalMaxPooling1D(),
        Dense(1, activation="sigmoid")
    ])
    m.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return m

def build_cnn_lstm():
    m = Sequential([
        Embedding(num_words, EMBEDDING_DIM, weights=[emb_matrix],
                  input_length=MAX_SEQ_LEN, trainable=False),
        Conv1D(64, 3, activation="relu"),
        MaxPooling1D(2),
        LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation="sigmoid")
    ])
    m.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return m

for name, factory in [("LSTM", build_lstm),
                      ("CNN", build_cnn),
                      ("CNN_LSTM", build_cnn_lstm)]:
    print(f"\n=== Training {name} ===")
    model    = factory()
    test_cb  = TestEvalCallback((X_test, y_test))
    early    = EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)

    history  = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early, test_cb],
        verbose=2
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"{name} — Final Test Accuracy: {test_acc*100:.2f}%")

    model_path = os.path.join(SAVE_ROOT, f"{name}_fasttext.h5")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    epochs_ran = len(history.history["loss"])
    plt.figure(figsize=(10,4))

    # plt.subplot(1,2,1)
    # plt.plot(range(1, epochs_ran+1), history.history["loss"],   label="train")
    # plt.plot(range(1, epochs_ran+1), history.history["val_loss"],label="val")
    # plt.plot(range(1, epochs_ran+1), test_cb.test_losses,       label="test")
    # plt.title(f"{name} Loss")
    # plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

    # plt.subplot(1,2,2)
    # plt.plot(range(1, epochs_ran+1), history.history["accuracy"],      label="train")
    # plt.plot(range(1, epochs_ran+1), history.history["val_accuracy"],  label="val")
    # plt.plot(range(1, epochs_ran+1), test_cb.test_accs,               label="test")
    # plt.title(f"{name} Accuracy")
    # plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()

    # plt.tight_layout()
    # curve_path = os.path.join(SAVE_ROOT, f"{name}_learning_curve.png")
    # plt.savefig(curve_path)
    # plt.close()
    # print(f"Learning curves saved to {curve_path}")

print("\nDone training all models.")
