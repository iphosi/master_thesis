import torch
from torchmetrics.functional import pairwise_cosine_similarity

import glob
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AdapterTrainer,
    TrainingArguments
)
from datasets import Dataset

from rsa import *


src_tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")
src_model = AutoModelForCausalLM.from_pretrained("dbmdz/german-gpt2")
src_tokenizer.pad_token = src_tokenizer.eos_token
src_model.config.pad_token_id = src_model.config.eos_token_id

"""
tgt_tokenizer = AutoTokenizer.from_pretrained("MiriUll/german-gpt2_easy")
tgt_model = AutoModelForCausalLM.from_pretrained("MiriUll/german-gpt2_easy")
tgt_tokenizer.pad_token = tgt_tokenizer.eos_token
tgt_model.config.pad_token_id = tgt_model.config.eos_token_id
"""

tgt_tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")
tgt_model = AutoModelForCausalLM.from_pretrained("dbmdz/german-gpt2")
tgt_tokenizer.pad_token = tgt_tokenizer.eos_token
tgt_model.config.pad_token_id = tgt_model.config.eos_token_id

tgt_model.load_adapter("../adapters/german-gpt2/Prefix_Tuning/model")
adapter_dicts = tgt_model.adapter_summary(as_dict=True)
adapter_names = [
    adapter_dict["name"] for adapter_dict in adapter_dicts
    if adapter_dict["name"] != "Full model"
]
for name in adapter_names:
    tgt_model.set_active_adapters(name)

input_path = "../datasets/aligned German simplification/mdr_aligned_news.csv"

dataset_df = pd.read_csv(input_path)

normal_texts = dataset_df.dropna(subset=["normal_phrase"])["normal_phrase"].values.tolist()
simple_texts = dataset_df.dropna(subset=["simple_phrase"])["simple_phrase"].values.tolist()

sample_texts = simple_texts

src_rep_spaces = get_rep_spaces(src_model, src_tokenizer, sample_texts)
tgt_rep_spaces = get_rep_spaces(tgt_model, tgt_tokenizer, sample_texts)

scores = get_pearson_scores(src_rep_spaces, tgt_rep_spaces)

print("End")
