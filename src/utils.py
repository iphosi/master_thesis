from datasets import load_dataset
from datasets import Features, Value

# Monolingual Dataset
dataset = load_dataset(
    path="csv",
    data_dir="../datasets/monolingual Leichte Sprache",
    features=Features({'phrase': Value(dtype='string', id=None)})
)["train"]

split_dataset = dataset.train_test_split(
    test_size=0.1,
    shuffle=True,
    seed=40
)
print(split_dataset["test"][0])

ppl_dataset = load_dataset(
    path="csv",
    data_dir="../datasets/aligned German simplification",
    data_files=["mdr_aligned_news.csv"],
    features=Features({
        "normal_phrase": Value(dtype="string", id=None),
        "simple_phrase": Value(dtype="string", id=None),
    })
)["train"]