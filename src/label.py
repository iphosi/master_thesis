import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

from textstat import textstat
from simplicity import get_log_freq_nltk


def dynamic_split(topic_group, num_phrases_per_sample=16):
    num_sections = max(1, len(topic_group) // num_phrases_per_sample)
    return np.array_split(topic_group, indices_or_sections=num_sections)


def join_texts(series_list):
    return list(map(lambda s: "\n".join(s.values), series_list))


def add_labels(input_path, output_path=None, text_column_name="Sentence", num_phrases_per_sample=16, use_lemma=False):
    text_dataframe = pd.read_csv(input_path).dropna()
    print(f"Dataset size before labeling: {len(text_dataframe)}")

    if text_column_name == "Sentence":
        output_path = f"../datasets/TextComplexity/human_feedback" if not output_path else output_path
        text_dataframe.rename(columns={"sentence_id": "phrase_id", "Sentence": "phrase"}, inplace=True)
        text_column_name = "phrase"

    elif text_column_name == "phrase":
        output_path = f"../datasets/TextComplexity/monolingual" if not output_path else output_path

        text_dataframe = text_dataframe.sort_values(["phrase_number"]).groupby(["topic"])["phrase"]
        text_dataframe = text_dataframe.apply(dynamic_split, num_phrases_per_sample=num_phrases_per_sample)
        text_dataframe = text_dataframe.apply(join_texts).reset_index().explode("phrase")
        text_dataframe = text_dataframe.replace(["^\s*$"], np.NaN, regex=True).dropna()

    else:
        raise ValueError("Invalid text column name.")

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    output_path = os.path.join(output_path, os.path.basename(input_path))

    tqdm.pandas()

    text_dataframe["FRE"] = text_dataframe[text_column_name].progress_apply(textstat.flesch_reading_ease)
    text_dataframe["WLF"] = text_dataframe[text_column_name].progress_apply(get_log_freq_nltk, use_lemma=use_lemma)

    print(f"Dataset size after labeling: {len(text_dataframe)}")
    text_dataframe.to_csv(output_path, index=False)


if __name__ == "__main__":
    textstat.set_lang("de")

    data_path = "../datasets/monolingual_Leichte_Sprache"
    dataset_path_list = glob.glob(f"{data_path}/*.csv")

    human_feedback_dataset = "../datasets/TextComplexity/orig/text_complexity.csv"

    dataset = dataset_path_list[8]

    add_labels(input_path=dataset, text_column_name="phrase", num_phrases_per_sample=16)

    print("End")
