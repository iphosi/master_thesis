import pandas as pd
from statistics import mean


word_freq_path = "../datasets/dewiki.txt"
word_freq_df = pd.read_csv(
    word_freq_path,
    sep=" ",
    header=None,
    names=["lemma", "frequency"]
)

freq = word_freq_df["frequency"].min()

print("End")
