from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import numpy as np
import torch

tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")
model = AutoModelForCausalLM.from_pretrained("dbmdz/german-gpt2")

pipe = pipeline(
    'text-generation',
    model="dbmdz/german-gpt2",
    tokenizer="dbmdz/german-gpt2"
)

text = pipe("Der Sinn des Lebens ist es", max_length=100)[0]["generated_text"]

print(text)
