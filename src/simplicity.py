from spacy.tokens import Doc
import pandas as pd
import numpy as np
from statistics import mean


def count_newlines(text: str):
    return text.count("\n")


def clean_none(count: int):
    return count if count else 0


def get_fre(doc: Doc):
    num_sents = sum([1 for _ in doc.sents])
    num_tokens = sum([1 for token in doc if not token.is_punct])
    num_syllables = sum([clean_none(token._.syllables_count) for token in doc])
    return 180 - 1.015 * num_tokens / num_sents - 84.6 * num_syllables / num_tokens


def get_word_log_freq(doc: Doc):
    word_freq_path = "../datasets/dewiki.txt"
    word_freq_df = pd.read_csv(
        word_freq_path,
        sep=" ",
        header=None,
        names=["lemma", "frequency"]
    )

    stop_mask = list(map(lambda t: not t.is_stop, doc))
    punct_mask = list(map(lambda t: not t.is_punct, doc))
    lemmas = list(map(lambda t: t.lemma_.lower(), doc))

    word_log_freq = []
    for lemma in lemmas:
        lemma_freq = word_freq_df.loc[word_freq_df["lemma"] == lemma]["frequency"]
        if lemma_freq.empty:
            freq = word_freq_df["frequency"].min()
        else:
            freq = lemma_freq.values[0]
        word_log_freq.append(np.log10(freq))

    word_log_freq = map(lambda f, s, p: f * s * p, word_log_freq, stop_mask, punct_mask)

    return mean(filter(lambda f: f > 0, word_log_freq))
