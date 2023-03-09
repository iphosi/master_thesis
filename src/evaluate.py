import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    LogitsProcessorList
)

from datasets import load_dataset
from datasets import Features, Value

import spacy
from spacy_syllables import SpacySyllables

from statistics import mean
from tqdm import tqdm
import csv
import os

from perplexity import *
from simplicity import *
from textstat import textstat


class Evaluate:
    def __init__(
        self,
        model_dict=None,
    ):
        self.model_dict = {
            "german-gpt2": {
                "ORIG": "dbmdz/german-gpt2",
                "FT": "MiriUll/german-gpt2_easy",
                "ADP_BN": "../adapters/Adapter_Bottleneck/model"
            },
            "gerpt2": {
                "ORIG": "benjamin/gerpt2",
                "FT": "MiriUll/gerpt2_easy",
                "ADP": None
            }
        } if model_dict is None else model_dict
        self.lang = "de"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        textstat.set_lang(self.lang)

    def perplexity_eval(
        self,
        model_names,
        ppl_dataset="../datasets/aligned German simplification/mdr_aligned_news.csv",
        output_path="../evaluation/perplexity.csv"
    ):
        # Dataset for Perplexity Evaluation
        ppl_dataset = pd.read_csv(ppl_dataset)

        normal_texts = ppl_dataset.dropna(subset=["normal_phrase"])["normal_phrase"].values.tolist()
        simple_texts = ppl_dataset.dropna(subset=["simple_phrase"])["simple_phrase"].values.tolist()

        columns = [
            "Model Name",
            "Tuning Method",
            "PPL Simple Text",
            "PPL Normal Text",
        ]

        data = []
        for model_name in model_names:
            if model_name not in self.model_dict.keys():
                raise ValueError("Invalid model name.")
            else:
                for tuning_method, model_path in self.model_dict[model_name].items():
                    print(f"{model_name}: {tuning_method}")

                    if model_path:
                        if tuning_method.startswith("ADP"):
                            tokenizer = AutoTokenizer.from_pretrained(self.model_dict[model_name]["ORIG"])
                            model = AutoModelForCausalLM.from_pretrained(self.model_dict[model_name]["ORIG"])
                            model.load_adapter(model_path)

                            adapter_dicts = model.adapter_summary(as_dict=True)
                            adapter_names = [
                                adapter_dict["name"] for adapter_dict in adapter_dicts
                                if adapter_dict["name"] != "Full model"
                            ]
                            for name in adapter_names:
                                model.set_active_adapters(name)

                            model.to(self.device)
                        else:
                            tokenizer = AutoTokenizer.from_pretrained(model_path)
                            model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)

                        tokenizer.pad_token = tokenizer.eos_token
                        model.eval()

                        simple_ppl = get_mod_ppl(
                            model,
                            tokenizer,
                            simple_texts,
                            self.device
                        )
                        normal_ppl = get_mod_ppl(
                            model,
                            tokenizer,
                            normal_texts,
                            self.device
                        )

                        data.append([model_name, tuning_method, round(simple_ppl, 2), round(normal_ppl, 2)])

                    else:
                        print("Model path is not defined.")
                        data.append([model_name, tuning_method, None, None])

        df = pd.DataFrame(data=data, columns=columns)
        df.to_csv(output_path, index=False)

    def simplicity_eval(
        self,
        model_names,
        spacy_model="de_dep_news_trf",
        input_path="../evaluation/generated_texts.json",
        output_path="../evaluation/simplicity.csv"
    ):
        # SpaCy Pipeline
        nlp = spacy.load(spacy_model)

        text_df = pd.read_json(input_path)

        columns = [
            "Model Name",
            "Tuning Method",
            "Average FRE",
            "Average Word Log Frequency",
            "Number of Newlines"
        ]

        data = []

        for model_name in model_names:
            if model_name not in self.model_dict.keys():
                raise ValueError("Invalid model name.")
            else:
                for tuning_method, model_path in self.model_dict[model_name].items():
                    print(f"{model_name}: {tuning_method}")
                    if model_path:
                        texts = text_df.loc[
                            (text_df["Model Name"] == model_name) & (text_df["Tuning Method"] == tuning_method)
                        ]["Generated Texts"].values[0]
                        docs = nlp.pipe(
                            texts,
                            disable=['tok2vec', 'morphologizer', 'attribute_ruler', 'ner']
                        )

                        avg_fre = mean(map(textstat.flesch_reading_ease, texts))
                        avg_word_log_freq = mean(map(get_log_freq, docs))
                        num_newlines = sum(map(count_newlines, texts))

                        data.append(
                            [model_name, tuning_method, round(avg_fre, 2), round(avg_word_log_freq, 2), num_newlines]
                        )

                    else:
                        data.append(
                            [model_name, tuning_method, None, None, None]
                        )

        df = pd.DataFrame(data=data, columns=columns)
        df.to_csv(output_path, index=False)

    def generate_text(
        self,
        model_names,
        input_prompts=None,
        logits_processor=None,
        repetition_penalty=3.0,
        temperature=1.0,
        max_new_tokens=100,
        penalty_alpha=0.6,
        top_k=3,
        num_beams=3,
        output_path="../evaluation/generated_texts.json"
    ):
        # Input Prompts for Readability Evaluation
        input_prompts = [
            "Das", "Heute", "Wir", "Die Türkei", "Dieses Haus", "Mein Vater"
        ] if input_prompts is None else input_prompts

        # Logits Processor
        '''
        Example:
        logits_processor = LogitsProcessorList(
            [
                RepetitionPenaltyLogitsProcessor(penalty=3.0),
                TemperatureLogitsWarper(temperature=1.0)
            ]
        ) if logits_processor is None else logits_processor
        '''

        columns = [
            "Model Name",
            "Tuning Method",
            "Generated Texts"
        ]

        data = []

        for model_name in model_names:
            if model_name not in self.model_dict.keys():
                raise ValueError("Invalid model name.")
            else:
                for tuning_method, model_path in self.model_dict[model_name].items():
                    print(f"{model_name}: {tuning_method}")

                    if model_path:
                        if tuning_method.startswith("ADP"):
                            tokenizer = AutoTokenizer.from_pretrained(self.model_dict[model_name]["ORIG"])
                            model = AutoModelForCausalLM.from_pretrained(self.model_dict[model_name]["ORIG"])
                            model.load_adapter(model_path)

                            adapter_dicts = model.adapter_summary(as_dict=True)
                            adapter_names = [
                                adapter_dict["name"] for adapter_dict in adapter_dicts
                                if adapter_dict["name"] != "Full model"
                            ]
                            for name in adapter_names:
                                model.set_active_adapters(name)

                            model.to(self.device)
                        else:
                            tokenizer = AutoTokenizer.from_pretrained(model_path)
                            model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)

                        tokenizer.pad_token = tokenizer.eos_token
                        model.eval()

                        batch_input_ids = [
                            tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
                            for prompt in input_prompts
                        ]

                        batch_output_ids = []

                        for input_ids in tqdm(batch_input_ids):
                            # Contrastive Search
                            batch_output_ids.append(
                                model.generate(
                                    input_ids=input_ids,
                                    logits_processor=logits_processor,
                                    repetition_penalty=repetition_penalty,
                                    temperature=temperature,
                                    max_new_tokens=max_new_tokens,
                                    penalty_alpha=penalty_alpha,
                                    top_k=top_k,
                                    pad_token_id=model.config.eos_token_id,
                                )
                            )

                            # Multinomial Sampling
                            batch_output_ids.append(
                                model.generate(
                                    input_ids=input_ids,
                                    logits_processor=logits_processor,
                                    repetition_penalty=repetition_penalty,
                                    temperature=temperature,
                                    max_new_tokens=max_new_tokens,
                                    do_sample=True,
                                    pad_token_id=model.config.eos_token_id
                                )
                            )

                            # Beam Search
                            batch_output_ids.append(
                                model.generate(
                                    input_ids=input_ids,
                                    logits_processor=logits_processor,
                                    repetition_penalty=repetition_penalty,
                                    temperature=temperature,
                                    max_new_tokens=max_new_tokens,
                                    num_beams=num_beams,
                                    do_sample=False,
                                    pad_token_id=model.config.eos_token_id
                                )
                            )

                        batch_output_texts = [
                            tokenizer.decode(
                                token_ids=output_ids[0],
                                skip_special_tokens=True
                            ) for output_ids in batch_output_ids
                        ]

                        data.append([model_name, tuning_method, batch_output_texts])

                    else:
                        print("Model path is not defined.")
                        data.append([model_name, tuning_method, None])

        df = pd.DataFrame(data=data, columns=columns)
        df.to_json(output_path)


if __name__ == "__main__":
    model_list = ["german-gpt2"]
    evaluate = Evaluate()
    #evaluate.generate_text(model_names=model_list)
    evaluate.perplexity_eval(model_names=model_list)
    #evaluate.simplicity_eval(model_names=model_list)
    print("End")
