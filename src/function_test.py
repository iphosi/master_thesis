import torch
import glob
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AdapterTrainer,
    TrainingArguments
)
from datasets import Dataset


def get_rsa_score(
    src_model,
    src_tokenizer,
    tgt_model,
    tgt_tokenizer,
    texts
):
    src_encodings = [src_tokenizer(text, return_tensors="pt") for text in texts]
    tgt_encodings = [tgt_tokenizer(text, return_tensors="pt") for text in texts]


def get_sim_matrix(
    model,
    encodings
):
    pass


device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2", device_map="auto")
model = AutoModelForCausalLM.from_pretrained("dbmdz/german-gpt2", device_map="auto")
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

config = model.config
num_layers = config.n_layer
spec_token_ids = [config.bos_token_id, config.eos_token_id, config.pad_token_id]
vocab_size = config.vocab_size
vocab_df = pd.Series(tokenizer.get_vocab()).to_frame(name="id")
vocab_df = vocab_df[~vocab_df["id"].isin(spec_token_ids)]

assert vocab_size == tokenizer.vocab_size

batch_token_id = [
    torch.as_tensor(value).to(device)
    for value in vocab_df.sample(n=5, random_state=40).values
]

# Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
# hidden_states: Tuple of torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)
batch_hidden_states = list(map(
    lambda token_id: model(token_id, output_hidden_states=True).hidden_states,
    batch_token_id
))

# (batch_size, sequence_length, hidden_size)
hidden_state_size = batch_hidden_states[0][0].size()

print("End")
