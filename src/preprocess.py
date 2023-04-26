import glob
import pandas as pd
import os

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, random_split
import unicodedata
from abc import ABC, abstractmethod
from typing import Iterable

from transformers import (
    GPT2TokenizerFast,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification
)
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing


class AbstractDataset(Dataset, ABC):
    def __init__(self, text_dataframe, tokenizer, stride_length, max_length):
        """
        text_dataframe: pandas dataframe with columns topic, phrase
        """
        assert (text_dataframe.columns.values == ["topic", "phrase"]).all()
        self.texts = text_dataframe

        text_list = [unicodedata.normalize("NFC", s) for s in list(self.texts["phrase"].values)]

        self.stride_length = stride_length

        self.encodings = tokenizer(
            text_list,
            truncation=True,
            max_length=max_length,
            stride=stride_length,
            return_special_tokens_mask=False,
            return_overflowing_tokens=False,
            add_special_tokens=True
        )

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self) -> int:
        """
        Returns number of samples in data set

        :return: int - number of samples in data set
        """
        return len(self.encodings["input_ids"])

    def get_source(self, idx) -> str:
        """
        Returns the source/topic of the requested item
        idx: index of a dataset item

        :return: str - the items original source
        """
        idx = self.encodings["overflow_to_sample_mapping"][idx]
        return self.get_name() + " -> " + self.texts.iloc[idx]["topic"]

    @abstractmethod
    def get_name(self) -> str:
        """
        Returns the name of the data set

        :return: str - name of the data set
        """
        pass

    @abstractmethod
    def get_columns(self) -> Iterable[str]:
        """
        Returns the names of all columns that the data set contains

        :return: list - names of the columns that are available
        """
        pass


class MonolingualDataset(AbstractDataset):
    def __init__(self, name, csv_file, tokenizer, stride_length, max_length, num_subtexts):
        text_dataframe = pd.read_csv(csv_file).dropna()
        text_dataframe = text_dataframe.sort_values(["phrase_number"]).groupby(["topic"])["phrase"]
        text_dataframe = text_dataframe.apply(np.array_split, indices_or_sections=num_subtexts).apply(self.join_texts)
        text_dataframe = text_dataframe.reset_index().explode("phrase")

        self.name = name
        super().__init__(text_dataframe, tokenizer, stride_length, max_length)

    def get_name(self) -> str:
        return self.name

    def get_columns(self) -> Iterable[str]:
        return self.texts.columns

    @staticmethod
    def join_texts(series_list):
        return list(map(lambda s: "\n".join(s.values), series_list))


class ConcatMonolingualDataset(ConcatDataset):

    def __init__(self, datasets: Iterable[MonolingualDataset]):
        super().__init__(datasets)

    def get_names(self) -> Iterable[str]:
        """
        Returns a list with the names of all data set that are contained in this combined data set

        :return: list - names of data sets in the data set collection
        """

        return [dataset.get_name() for dataset in self.datasets]

    def get_summary(self) -> str:
        total_items = 0
        individual_items = {}
        for dataset in self.datasets:
            individual_items[dataset.get_name()] = len(dataset)
            total_items += len(dataset)

        for key in individual_items.keys():
            individual_items[key] = "{:.2f}%".format((individual_items[key] / total_items) * 100)

        return f"Dataset contains {total_items} items {individual_items}"


class TextComplexityDataset(Dataset):
    def __init__(self, csv_file, text_column_name, target_label, tokenizer, max_length, stride_length):
        text_dataframe = pd.read_csv(csv_file).dropna()[[text_column_name, target_label]]
        texts = [unicodedata.normalize("NFC", s) for s in list(text_dataframe[text_column_name].values)]
        labels = list(text_dataframe[target_label].values)

        self.texts = texts
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            stride=stride_length,
            add_special_tokens=True
        )
        self.labels = labels
        self.max = max(labels)
        self.min = min(labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = float(self.labels[idx])
        return item

    def __len__(self):
        return len(self.texts)

    def __get_texts__(self):
        return self.texts

    def __get_labels__(self):
        return self.labels


class ConcatTextComplexityDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[TextComplexityDataset], do_scaling=True):
        self.max = max(dataset.max for dataset in datasets)
        self.min = min(dataset.min for dataset in datasets)

        if do_scaling:
            for dataset in datasets:
                dataset.labels = list(
                    map(lambda l: (l - self.min) / (self.max - self.min), dataset.labels)
                )

        super().__init__(datasets)


def get_monolingual_dataset(
    tokenizer,
    max_length,
    stride_length=64,
    num_subtexts=4,
    input_path="../datasets/monolingual Leichte Sprache"
):
    dataset_path_list = glob.glob(f"{input_path}/*.csv")
    dataset_name_list = [
        os.path.splitext(
            os.path.basename(path)
        )[0]
        for path in dataset_path_list
    ]

    dataset_list = [
        MonolingualDataset(name, path, tokenizer, stride_length, max_length, num_subtexts)
        for name, path in zip(dataset_name_list, dataset_path_list)
    ]

    return ConcatMonolingualDataset(dataset_list)


def get_text_complexity_dataset(
    tokenizer,
    max_length,
    stride_length=64,
    text_column_name="Sentence",
    target_label="MOS",
    do_scaling=True,
    input_path="../datasets/TextComplexity/monolingual"
):
    dataset_path_list = glob.glob(f"{input_path}/*.csv")

    dataset_list = [
        TextComplexityDataset(path, text_column_name, target_label, tokenizer, max_length, stride_length)
        for path in dataset_path_list
    ]

    return ConcatTextComplexityDataset(dataset_list, do_scaling)


def split_dataset(
    dataset,
    val_split=0.1,
    test_split=0.0,
    seed=40
):
    val_size = int(val_split * len(dataset))
    test_size = int(test_split * len(dataset))
    train_size = len(dataset) - val_size - test_size
    generator = torch.Generator().manual_seed(seed)

    if val_split > 0 and test_split > 0:
        train_set, val_set, test_set = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=generator
        )

        return train_set, val_set, test_set

    elif val_split > 0:
        train_set, val_set = random_split(
            dataset,
            [train_size, val_size],
            generator=generator
        )

        return train_set, val_set

    elif test_split > 0:
        train_set, test_set = random_split(
            dataset,
            [train_size, test_size],
            generator=generator
        )

        return train_set, test_set


def specify_config(
    model_path=None,
    special_tokens_dict=None,
    head_type="causal",
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    resid_pdrop=0.1,
    output_path="../adapters",
    save_config=False
):
    if model_path is None:
        raise ValueError("Model path is not specified.")

    model_name = model_path.split("/")[1]

    special_tokens_dict = {
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "pad_token": "<|pad|>",
        "unk_token": "<|unk|>"
    } if special_tokens_dict is None else special_tokens_dict

    head_type_list = ["causal", "regression"]
    if head_type not in head_type_list:
        raise ValueError("Unknown head type.")

    bos = special_tokens_dict["bos_token"]
    eos = special_tokens_dict["eos_token"]

    tokenizer_orig = AutoTokenizer.from_pretrained(model_path)
    tokenizer_orig.add_special_tokens(special_tokens_dict)

    tokenizer = Tokenizer.from_pretrained(model_path)
    tokenizer.post_processor = TemplateProcessing(
        single=bos + " $A " + eos,
        special_tokens=[(eos, tokenizer_orig.eos_token_id), (bos, tokenizer_orig.bos_token_id)],
    )
    tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)
    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)

    if head_type == "causal":
        model_config = AutoConfig.from_pretrained(
            model_path,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    elif head_type == "regression":
        model_config = AutoConfig.from_pretrained(
            model_path,
            num_labels=1,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    else:
        model_config = None

    model_config.embd_pdrop = embd_pdrop
    model_config.attn_pdrop = attn_pdrop
    model_config.resid_pdrop = resid_pdrop

    if head_type == "causal":
        model = AutoModelForCausalLM.from_pretrained(model_path, config=model_config)
    elif head_type == "regression":
        model = AutoModelForSequenceClassification.from_pretrained(model_path, config=model_config)
    else:
        model = None

    if tokenizer and model:
        model.resize_token_embeddings(len(tokenizer))

        if save_config:
            tokenizer.save_pretrained(
                os.path.join(output_path, f"{model_name}/Orig/{head_type}")
            )
            model.save_pretrained(
                os.path.join(output_path, f"{model_name}/Orig/{head_type}")
            )

    return model, tokenizer
