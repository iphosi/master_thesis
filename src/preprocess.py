import glob
import pandas as pd


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
