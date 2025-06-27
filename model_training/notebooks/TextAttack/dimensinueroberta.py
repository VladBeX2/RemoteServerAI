from transformers import RobertaForSequenceClassification

# Calea către folderul în care ai salvat modelul
MODEL_PATH = "roberta_adv_finetuned"

# Încarcă modelul
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)

# Afișează arhitectura
print(model)

# Calculează dimensiunea (număr de parametri)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
