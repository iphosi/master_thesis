from spacy.tokens import Doc
import pandas as pd
import numpy as np
from statistics import mean


word_freq_path = "../datasets/dewiki.txt"
size = 100000
word_freq_df = pd.read_csv(
    word_freq_path,
    sep=" ",
    header=None,
    names=["lemma", "frequency"]
).iloc[:size]
min_freq = word_freq_df["frequency"].min()


def count_newlines(text):
    return text.count("\n")


def count_sentences(doc):
    return sum(1 for _ in doc.sents)


def clean(lemma_freq):
    if lemma_freq.empty:
        return min_freq
    else:
        return lemma_freq.values[0]


def get_log_freq(doc):
    stop_mask = list(map(lambda t: not t.is_stop, doc))
    punct_mask = list(map(lambda t: not t.is_punct, doc))

    freq = [
        word_freq_df.loc[
            word_freq_df["lemma"] == token.lemma_.lower()
        ]["frequency"] for token in doc
    ]
    log_freq = np.log10(list(map(clean, freq)))

    log_freq = map(lambda f, s, p: f * s * p, log_freq, stop_mask, punct_mask)

    return mean(filter(lambda f: f > 0, log_freq))
