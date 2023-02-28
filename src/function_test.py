import torch
from datasets import load_dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = load_dataset(path="../datasets/test")
print(dataset["train"]["phrase"][0])

