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


def concat_datasets(
    path="../datasets/monolingual Leichte Sprache",
    columns=None
):
    columns = ["phrase"] if columns is None else columns
    files = glob.glob(f"{path}/*.csv")
    return pd.concat((pd.read_csv(file)[columns] for file in files)).dropna()


def group_texts(examples, block_size=50):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [
            t[i:i + block_size] for i in range(0, total_length, block_size)
        ]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2", device_map="auto")
model = AutoModelForCausalLM.from_pretrained("dbmdz/german-gpt2", device_map="auto")
tokenizer.pad_token = tokenizer.eos_token

# Monolingual Dataset
dataset_df = concat_datasets()

dataset = Dataset.from_pandas(dataset_df, preserve_index=False)
column_names = dataset.column_names

dataset = dataset.train_test_split(
    test_size=0.1,
    shuffle=True,
    seed=40
)

dataset = dataset.map(
    lambda batch: tokenizer(batch["phrase"]),
    remove_columns=column_names,
    batched=True
)

dataset = dataset.map(group_texts, batched=True)

dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

model.add_adapter(adapter_name="test")
model.train_adapter(adapter_setup="test")

training_args = TrainingArguments(
    output_dir="../adapters/adapter_test/checkpoints",
    do_train=True,
    remove_unused_columns=False,
    learning_rate=5e-4,
    num_train_epochs=3,
    save_steps=5000
)

trainer = AdapterTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

trainer.train()
model.save_adapter(
    save_directory="../adapters/adapter_test/model",
    adapter_name="test"
)

print("End")
