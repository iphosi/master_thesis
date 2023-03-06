from textstat import textstat
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import pandas as pd
import spacy
from simplicity import *

text_path = "../evaluation/generated_texts.json"
df = pd.read_json(text_path)
texts = df.loc[(df["Model Name"] == "german-gpt2") & (df["Tuning Method"] == "ORIG")]["Generated Texts"].values[0]

nlp = spacy.load("de_dep_news_trf")
docs = nlp.pipe(
    texts,
    disable=['tok2vec', 'morphologizer', 'attribute_ruler', 'ner']
)
word_freq_path = "../datasets/dewiki.txt"
word_freq_df = pd.read_csv(
    word_freq_path,
    sep=" ",
    header=None,
    names=["lemma", "frequency"]
)
print("End")