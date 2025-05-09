import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ✅ 1️⃣ CONFIGURAȚIE MODEL
MODEL_NAME = "jy46604790/Fake-News-Bert-Detect"  # Model RoBERTa pentru fake news detection
OUTPUT_FILE = "predictions_fake_news_corrected.csv"
GPU_ID = 0  # Asigură-te că GPU-ul este valid
BATCH_SIZE = 16

# ✅ 2️⃣ SELECTEAZĂ GPU-UL SAU CPU
device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")

# ✅ 3️⃣ ÎNCARCĂ MODELUL ȘI TOKENIZER-UL
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ✅ 4️⃣ ÎNCARCĂ DATASET-UL
DATASET_PATH = "../datasets/WELFake_Dataset.csv"  # Verifică locația dataset-ului
df = pd.read_csv(DATASET_PATH)

# ✅ 5️⃣ INVERSEAZĂ LABEL-URILE (PENTRU A SE POTRIVI CU MODELUL)
# Modelul folosește 0 = Fake News, 1 = Real News, dar dataset-ul este invers.
df["label_corrected"] = df["label"].apply(lambda x: 1 - x)  # 0 <-> 1

# ✅ 6️⃣ VERIFICĂ ȘI CURĂȚĂ DATELE
df = df.dropna(subset=["text"])  # Elimină NaN
df["text"] = df["text"].astype(str)  # Conversie la string
texts = df["text"].tolist()

# ✅ 7️⃣ FUNCȚIE PENTRU INFERENȚĂ BATCH
def predict_batch(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.softmax(outputs.logits, dim=1)[:, 1].tolist()  # Probabilitatea ca știrea să fie Real News

# ✅ 8️⃣ EXECUTĂ INFERENȚA
predictions = []
for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Processing Batches"):
    batch_texts = texts[i : i + BATCH_SIZE]
    batch_preds = predict_batch(batch_texts)
    predictions.extend(batch_preds)

# ✅ 9️⃣ ADĂUGĂ PREDICȚIILE ÎN DATASET
df["prediction"] = predictions

# ✅ 🔟 CONVERTIM PREDICȚIILE PROBABILISTICE ÎN LABELS 0/1
df["predicted_label"] = (df["prediction"] > 0.5).astype(int)

# ✅ 1️⃣1️⃣ EXTRAGEM VALORILE REALE ȘI PREVĂZUTE
y_true = df["label_corrected"]  # Folosim labelurile corectate
y_pred = df["predicted_label"]

# ✅ 1️⃣2️⃣ CALCULĂM METRICILE CORECTE
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="binary")
recall = recall_score(y_true, y_pred, average="binary")
f1 = f1_score(y_true, y_pred, average="binary")

# ✅ 1️⃣3️⃣ AFIȘEAZĂ RAPORTUL COMPLET
report = classification_report(y_true, y_pred, target_names=["Fake News", "Real News"])


# ✅ 1️⃣5️⃣ AFIȘEAZĂ REZULTATELE CORECTATE
print("\n===== METRICILE DE EVALUARE (CORECTATE) =====")
print(f"Acuratețe: {accuracy:.4f}")
print(f"Precizie: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}\n")

print("\n===== RAPORT CLASIFICARE =====")
print(report)

