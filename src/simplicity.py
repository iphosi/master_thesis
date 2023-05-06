from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from statistics import mean
from nltk.tokenize import word_tokenize, sent_tokenize

word_freq_path = "../datasets/dewiki.txt"
size = 100000
word_freq_df = pd.read_csv(
    word_freq_path,
    sep=" ",
    header=None,
    names=["lemma", "frequency"]
).iloc[:size]
min_freq = word_freq_df["frequency"].min()
max_freq = word_freq_df["frequency"].max()


def count_newlines(text):
    return text.count("\n")


def count_sentences_spacy(doc):
    return sum(1 for _ in doc.sents)


def count_sentences_nltk(text, language="german"):
    language = "german" if language == "de" else language
    return len(sent_tokenize(text, language=language))


def clean(lemma_freq):
    if lemma_freq.empty:
        return min_freq
    else:
        _ = lemma_freq.values
        freq = lemma_freq.values[0]
        return freq


def get_log_freq_spacy(doc):
    stop_mask = list(map(lambda t: not t.is_stop, doc))
    punct_mask = list(map(lambda t: not t.is_punct, doc))

    freq = [
        word_freq_df.loc[
            word_freq_df["lemma"] == token.lemma_.lower()
        ]["frequency"] for token in doc
    ]
    log_freq = np.log10(list(map(clean, freq)))

    log_freq = map(lambda f, s, p: f * s * p, log_freq, stop_mask, punct_mask)
    log_freq = list(filter(lambda f: f > 0, log_freq))

    return mean(log_freq) if log_freq else np.log10(min_freq)


def get_log_freq_nltk(text, language="german", use_lemma=False):
    language = "german" if language == "de" else language

    if use_lemma:
        raise NotImplementedError
    else:
        tokens = word_tokenize(text, language=language)

    tokens = filter(lambda t: t.isalpha() and t.lower() not in set(stopwords.words(language)), tokens)
    freq = [
        word_freq_df.loc[
            word_freq_df["lemma"] == token.lower()
        ]["frequency"] for token in tokens
    ]
    log_freq = np.log10(list(map(clean, freq)))

    return np.mean(log_freq) if log_freq.size else np.log10(min_freq)
