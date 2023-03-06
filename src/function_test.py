import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
model.load_adapter("sentiment/sst-2@ukp")
model.set_active_adapters("sst-2")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize("AdapterHub is awesome!")
input_tensor = torch.tensor([
    tokenizer.convert_tokens_to_ids(tokens)
])
outputs = model(input_tensor)
print("End")
