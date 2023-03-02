from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from datasets import Features, Value

from get_ppl import *

tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")
model_orig = AutoModelForCausalLM.from_pretrained("dbmdz/german-gpt2")
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

num_samples = 3
normal_texts = mdr_dataset["normal_phrase"][:num_samples]
simple_texts = mdr_dataset["simple_phrase"][:num_samples]
normal_encodings = tokenizer("\n\n".join(normal_texts), return_tensors="pt")
simple_encodings = tokenizer("\n\n".join(simple_texts), return_tensors="pt")

# Perplexity Calculation
ppl_orig = get_perplexity(
    model_orig,
    simple_encodings,
)
ppl_ft = get_perplexity(
    model_ft,
    simple_encodings
)
print("End")