import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from preprocess import specify_config


device = "cuda" if torch.cuda.is_available() else "cpu"
prompt = "Das ist"

# If the model is loaded from remote, add special tokens and resize the vocab.
# model_path = "malteos/bloom-350m-german"
# model_name = model_path.split("/")[-1]
# model, tokenizer = specify_config(model_path=model_path)

# Load preprocessed model from local.
model_path = "../adapters/bloom-350m-german/Orig"
model_name = model_path.split("/")[-2]

model = AutoModelForCausalLM.from_pretrained(model_path)
model_orig = AutoModelForCausalLM.from_pretrained(model_path)
model.eval()
model_orig.eval()

tokenizer = AutoTokenizer.from_pretrained(model_path)

input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

adapter_path = "../adapters/bloom-350m-german/Compacter++/model_r8_n64"
# adapter_path = "../adapters/bloom-350m-german/Adapter_Bottleneck_Sequential/model_r16"
layer_range = 2
leave_out = [layer for layer in range(layer_range)]
model.load_adapter(adapter_path, leave_out=leave_out)
adapter_dicts = model.adapter_summary(as_dict=True)

adapter_names = [
    adapter_dict["name"] for adapter_dict in adapter_dicts
    if adapter_dict["name"] != "Full model"
]
for name in adapter_names:
    model.set_active_adapters(name)

model.to(device)
model_orig.to(device)

max_new_tokens = 100
temperature = 1.0
repetition_penalty = 1.4
penalty_alpha = 1.6
top_k = 4
top_p = 0.7
num_beams = 3
no_repeat_ngram_size = 2
early_stopping = True

# Contrastive Search
output_ids_orig = model_orig.generate(
    input_ids=input_ids,
    repetition_penalty=repetition_penalty,
    temperature=temperature,
    max_new_tokens=max_new_tokens,
    penalty_alpha=penalty_alpha,
    top_k=top_k,
    no_repeat_ngram_size=no_repeat_ngram_size
)

output_ids = model.generate(
    input_ids=input_ids,
    repetition_penalty=repetition_penalty,
    temperature=temperature,
    max_new_tokens=max_new_tokens,
    penalty_alpha=penalty_alpha,
    top_k=top_k,
    no_repeat_ngram_size=no_repeat_ngram_size
)

output_text_orig = tokenizer.decode(
    token_ids=output_ids_orig[0],
    skip_special_tokens=True
)

output_text = tokenizer.decode(
    token_ids=output_ids[0],
    skip_special_tokens=True
)

print("End")

