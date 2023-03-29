import glob
import pandas as pd
import os

import torch
from torch.utils.data import Dataset, ConcatDataset, random_split
import unicodedata
from abc import ABC, abstractmethod
from typing import Iterable


class AbstractDataset(Dataset, ABC):
    def __init__(self, text_dataframe, stride_length, tokenizer, max_len):
        """
        text_dataframe: pandas dataframe with columns topic, phrase
        """
        assert ((text_dataframe.columns.values == ["topic", "phrase"]).all())
        self.texts = text_dataframe

        text_list = [unicodedata.normalize("NFC", s) for s in list(self.texts["phrase"].values)]

        self.stride_length = stride_length

        self.encodings = tokenizer(
            text_list,
            truncation=True,
            max_length=max_len,
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
    def __init__(self, name, csv_file, stride_length, tokenizer, max_len):
        phrases = pd.read_csv(csv_file).fillna('text')
        texts = phrases.sort_values(["phrase_number"]).groupby(["topic"])["phrase"].apply('\n'.join).reset_index()
        self.name = name
        super().__init__(texts, stride_length, tokenizer, max_len)

    def get_name(self) -> str:
        return self.name

    def get_columns(self) -> Iterable[str]:
        return self.texts.columns


class CombinedDataset(ConcatDataset):

    def __init__(self, datasets: Iterable[AbstractDataset]):
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


def get_dataset(
    tokenizer,
    max_length,
    stride_length=64,
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
        MonolingualDataset(name, path, stride_length, tokenizer, max_length)
        for name, path in zip(dataset_name_list, dataset_path_list)
    ]

    return CombinedDataset(dataset_list)


def split_dataset(
    dataset,
    val_split=0.1,
    test_split=0.0,
    seed=40
):
    train_split = 1 - val_split - test_split
    generator = torch.Generator().manual_seed(seed)

    assert 0 < train_split < 1 and 0 <= val_split < 1 and 0 <= test_split < 1

    if val_split > 0 and test_split > 0:
        train_set, val_set, test_set = random_split(
            dataset,
            [train_split, val_split, test_split],
            generator=generator
        )

        return train_set, val_set, test_set

    elif val_split > 0:
        train_set, val_set = random_split(
            dataset,
            [train_split, val_split],
            generator=generator
        )

        return train_set, val_set

    elif test_split > 0:
        train_set, test_set = random_split(
            dataset,
            [train_split, test_split],
            generator=generator
        )

        return train_set, test_set
