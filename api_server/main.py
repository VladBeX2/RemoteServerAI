from fastapi import FastAPI
from pydantic import BaseModel
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf

# Configurare model
model_path = "../model_training/saved_models/saved_bert_model"
tokenizer_path = "../model_training/saved_models/saved_bert_tokenizer"

model = TFBertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

app = FastAPI()

# Structura pentru cererea POST
class InputData(BaseModel):
    text: str

@app.post("/predict")
def predict(data: InputData):
    # Tokenizare
    tokens = tokenizer(
        data.text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="tf"
    )
    
    # Predicție
    prediction = model.predict(
        {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"]
        }
    )
    
    # Extrage probabilitatea (sigmoid pentru binar)
    prob_fake = float(tf.nn.sigmoid(prediction.logits[0]).numpy()[0])
    prob_real = 1 - prob_fake  # Complementul probabilității pentru clasa REAL

    # Determină eticheta bazată pe probabilitate
    label = "FAKE" if prob_fake >= 0.5 else "REAL"

    # Returnează ambele probabilități
    return {
        "label": label,
        "probability": [prob_fake, prob_real]
    }