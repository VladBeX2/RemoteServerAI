import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ✅ 1️⃣ CONFIGURAȚIE MODEL (ROBERTA FAKE NEWS)
MODEL_NAME = "hamzab/roberta-fake-news-classification"
GPU_ID = 0  # Alege GPU-ul corect
BATCH_SIZE = 16  # Ajustează în funcție de VRAM

# ✅ 2️⃣ SELECTEAZĂ GPU-UL SAU CPU
device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")

# ✅ 3️⃣ ÎNCARCĂ MODELUL ȘI TOKENIZER-UL
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ✅ 4️⃣ ÎNCARCĂ DATASET-UL
DATASET_PATH = "../datasets/WELFake_Dataset.csv"
df = pd.read_csv(DATASET_PATH)

# ✅ 5️⃣ VERIFICĂ LABEL-URILE ȘI CONVERTIM DACĂ ESTE NECESAR
# Modelul folosește "FAKE" și "REAL", dar WELFake folosește 0 și 1.
label_mapping = {0: "FAKE", 1: "REAL"}  # Mapăm label-urile dataset-ului nostru
df["label_text"] = df["label"].map(label_mapping)

# ✅ 6️⃣ CURĂȚĂ DATELE
df = df.dropna(subset=["text"])  # Elimină NaN
df["text"] = df["text"].astype(str)  # Convertim textele la string
texts = df["text"].tolist()

# ✅ 7️⃣ FUNCȚIE PENTRU INFERENȚĂ
def predict_batch(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)  # Alegem clasa cu probabilitatea cea mai mare
    return predictions.cpu().numpy()

# ✅ 8️⃣ EXECUTĂ INFERENȚA PE TOATE TEXTELE
predictions = []
for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Processing Batches"):
    batch_texts = texts[i : i + BATCH_SIZE]
    batch_preds = predict_batch(batch_texts)
    predictions.extend(batch_preds)

# ✅ 9️⃣ MAPĂM PREDICȚIILE ÎNAPOI LA LABEL-URI (FAKE/REAL → 0/1)
prediction_mapping = {"FAKE": 1, "REAL": 0}
df["predicted_label"] = [prediction_mapping["FAKE"] if pred == 0 else prediction_mapping["REAL"] for pred in predictions]

# ✅ 🔟 EXTRAGEM LABELURILE ȘI PREDICȚIILE
y_true = df["label"]
y_pred = df["predicted_label"]

# ✅ 1️⃣1️⃣ CALCULĂM METRICILE
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="binary")
recall = recall_score(y_true, y_pred, average="binary")
f1 = f1_score(y_true, y_pred, average="binary")

# ✅ 1️⃣2️⃣ RAPORT CLASIFICARE
report = classification_report(y_true, y_pred, target_names=["Fake News", "Real News"])

# ✅ 1️⃣3️⃣ AFIȘEAZĂ METRICILE
print("\n===== METRICILE DE EVALUARE =====")
print(f"Acuratețe: {accuracy:.4f}")
print(f"Precizie: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}\n")

print("\n===== RAPORT CLASIFICARE =====")
print(report)

print(f"\n✅ Evaluarea s-a terminat, metricile au fost afișate.")
