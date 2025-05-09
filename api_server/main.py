from fastapi import FastAPI
from pydantic import BaseModel
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf

model_path = "../model_training/notebooks/welFake_nb/saved_models/saved_bert_model"
tokenizer_path = "../model_training/notebooks/welFake_nb/saved_models/saved_bert_tokenizer"

model = TFBertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

app = FastAPI()

class InputData(BaseModel):
    text: str

@app.post("/predict")
def predict(data: InputData):
    tokens = tokenizer(
        data.text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="tf"
    )
    
    prediction = model.predict(
        {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"]
        }
    )
    
    prob_fake = float(tf.nn.sigmoid(prediction.logits[0]).numpy()[0])
    prob_real = 1 - prob_fake  

    label = "FAKE" if prob_fake >= 0.5 else "REAL"

    return {
        "label": label,
        "probability": max(prob_fake, prob_real)
    }