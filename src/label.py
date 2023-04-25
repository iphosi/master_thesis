import pandas as pd
import numpy as np
import os
import glob

from textstat import textstat
from simplicity import get_log_freq_nltk


def join_texts(series_list):
    return list(map(lambda s: "\n".join(s.values), series_list))


def add_labels(input_path, output_path=None, text_column_name="Sentence", num_subtexts=3, use_lemma=False):
    text_dataframe = pd.read_csv(input_path).dropna()

    if text_column_name == "Sentence":
        output_path = f"../datasets/TextComplexity/human_feedback" if not output_path else output_path

    elif text_column_name == "phrase":
        output_path = f"../datasets/TextComplexity/monolingual" if not output_path else output_path

        text_dataframe = text_dataframe.sort_values(["phrase_number"]).groupby(["topic"])["phrase"]
        text_dataframe = text_dataframe.apply(np.array_split, indices_or_sections=num_subtexts).apply(join_texts)
        text_dataframe = text_dataframe.reset_index().explode("phrase").dropna()

    else:
        raise ValueError("Invalid text column name.")

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    output_path = os.path.join(output_path, os.path.basename(input_path))

    text_dataframe["FRE"] = text_dataframe[text_column_name].apply(textstat.flesch_reading_ease)
    text_dataframe["WLF"] = text_dataframe[text_column_name].apply(get_log_freq_nltk, use_lemma=use_lemma)
    text_dataframe.to_csv(output_path, index=False)


if __name__ == "__main__":
    textstat.set_lang("de")

    data_path = "../datasets/monolingual Leichte Sprache"
    dataset_path_list = glob.glob(f"{data_path}/*.csv")

    human_feedback_dataset = "../datasets/TextComplexity/orig/text_complexity.csv"

    add_labels(input_path=dataset_path_list[8], text_column_name="phrase")

    print("End")
