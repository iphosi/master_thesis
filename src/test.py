import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AdapterConfig
)

adapter_name = "Adapter_Bottleneck_LayerNorm"
adapter_config = AdapterConfig(
    mh_adapter=True,
    output_adapter=True,
    reduction_factor=16,
    non_linearity="gelu",
    ln_before=True,
    ln_after=True
)

tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2", device_map="auto")
model = AutoModelForCausalLM.from_pretrained("dbmdz/german-gpt2", device_map="auto")

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

#model.add_adapter(adapter_name=adapter_name, config=adapter_config)
summary = model.adapter_summary(as_dict=True)
param = summary[0]["%param"]

print("End")
