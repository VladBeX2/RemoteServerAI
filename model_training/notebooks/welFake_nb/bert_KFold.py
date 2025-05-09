import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

gpus = tf.config.list_physical_devices('GPU')
print(f"Available GPUs: {gpus}")

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

strategy = tf.distribute.MirroredStrategy()

df = pd.read_csv("../datasets/WELFake_Dataset.csv")
print("Număr valori nule în coloana 'text':", df["text"].isnull().sum())
df = df.dropna(subset=["text"])
texts = df["text"].tolist()
labels = df["label"].tolist()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
max_length = 128

def tokenize_texts(texts, tokenizer, max_length):
    return tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="tf"
    )

k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

fold_no = 1
for train_index, val_index in kf.split(texts):
    print(f"\n===== Antrenare pentru fold {fold_no} =====")
    
    train_texts = [texts[i] for i in train_index]
    val_texts = [texts[i] for i in val_index]
    train_labels = [labels[i] for i in train_index]
    val_labels = [labels[i] for i in val_index]
    
    train_tokens = tokenize_texts(train_texts, tokenizer, max_length)
    val_tokens = tokenize_texts(val_texts, tokenizer, max_length)
    
    with strategy.scope():
        model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )
    
    history = model.fit(
        x={
            "input_ids": train_tokens["input_ids"],
            "attention_mask": train_tokens["attention_mask"]
        },
        y=tf.convert_to_tensor(train_labels, dtype=tf.float32),
        validation_data=(
            {
                "input_ids": val_tokens["input_ids"],
                "attention_mask": val_tokens["attention_mask"]
            },
            tf.convert_to_tensor(val_labels, dtype=tf.float32)
        ),
        epochs=3,
        batch_size=16 * strategy.num_replicas_in_sync  
    )
    
    results = model.evaluate(
        x={
            "input_ids": val_tokens["input_ids"],
            "attention_mask": val_tokens["attention_mask"]
        },
        y=tf.convert_to_tensor(val_labels, dtype=tf.float32),
        batch_size=16
    )
    print(f"Fold {fold_no} - Validation Accuracy: {results[1] * 100:.2f}%")
    
    save_path = f"../saved_models/saved_bert_model_fold_{fold_no}"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Modelul pentru fold {fold_no} a fost salvat în: {save_path}")
    
    fold_no += 1
