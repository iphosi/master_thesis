from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from datasets import Features, Value

from perplexity import *

tokenizer_orig = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")
model_orig = AutoModelForCausalLM.from_pretrained("dbmdz/german-gpt2")
tokenizer_ft = AutoTokenizer.from_pretrained("MiriUll/german-gpt2_easy")
model_ft = AutoModelForCausalLM.from_pretrained("MiriUll/german-gpt2_easy")

# MDR Dataset for Perplexity Calculation
mdr_dataset = load_dataset(
    path="csv",
    data_dir="../datasets/aligned German simplification",
    data_files=[
        "mdr_aligned_dictionary.csv",
        "mdr_aligned_news.csv"
    ],
    features=Features({
        "normal_phrase": Value(dtype="string", id=None),
        "simple_phrase": Value(dtype="string", id=None),
    })
)["train"]

normal_texts = mdr_dataset["normal_phrase"][:3]
simple_texts = mdr_dataset["simple_phrase"][:3]

# Perplexity Calculation
normal_ppl_orig = get_modified_perplexity(
    model_orig,
    tokenizer_orig,
    normal_texts
)
normal_ppl_ft = get_modified_perplexity(
    model_ft,
    tokenizer_orig,
    normal_texts
)
simple_ppl_orig = get_modified_perplexity(
    model_orig,
    tokenizer_ft,
    simple_texts
)
simple_ppl_ft = get_modified_perplexity(
    model_ft,
    tokenizer_ft,
    simple_texts
)
print("End")