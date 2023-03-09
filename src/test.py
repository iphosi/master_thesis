from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch
import pandas as pd
import os


path = "../adapters/Adapter_Bottleneck/model"
model = AutoModelForCausalLM.from_pretrained("dbmdz/german-gpt2")

config = model.config

eos_token_id = model.config.eos_token_id

model.add_adapter("adapter_1")
model.add_adapter("adapter_2")

adapter_dicts = model.adapter_summary(as_dict=True)
adapter_names = [adapter_dict["name"] for adapter_dict in adapter_dicts if adapter_dict["name"] != "Full model"]

print("End")
