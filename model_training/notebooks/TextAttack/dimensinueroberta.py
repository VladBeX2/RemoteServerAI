from transformers import RobertaForSequenceClassification

 
MODEL_PATH = "roberta_adv_finetuned"

 
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)

 
print(model)

 
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
