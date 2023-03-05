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


class Evaluate:
    def __init__(
        self,
        model_dict=None,
    ):
        self.model_dict = {
            "german-gpt2": {
                "ORIG": "dbmdz/german-gpt2",
                "FT": "MiriUll/german-gpt2_easy",
                "ADP": None
            },
            "gerpt2": {
                "ORIG": "benjamin/gerpt2",
                "FT": "MiriUll/gerpt2_easy",
                "ADP": None
            }
        } if model_dict is None else model_dict

    def perplexity_eval(
        self,
        model_names,
        ppl_dataset=None,
        num_samples=None,
        output_path="../evaluation"
    ):
        # Dataset for Perplexity Evaluation
        ppl_dataset = load_dataset(
            path="csv",
            data_dir="../datasets/aligned German simplification",
            data_files=[
                "mdr_aligned_dictionary.csv",
                "mdr_aligned_news.csv"
            ],
            features=Features({
                "normal_phrase": Value(dtype="string", id=None),
                "simple_phrase": Value(dtype="string", id=None),
            })
        )["train"] if ppl_dataset is None else ppl_dataset

        if num_samples:
            normal_texts = ppl_dataset["normal_phrase"][:num_samples]
            simple_texts = ppl_dataset["simple_phrase"][:num_samples]
        else:
            normal_texts = ppl_dataset["normal_phrase"]
            simple_texts = ppl_dataset["simple_phrase"]

        header = [
            "Model Name",
            "ORIG: PPL Simple Text",
            "ORIG: PPL Normal Text",
            "FT: PPL Simple Text",
            "FT: PPL Normal Text",
            "ADP: PPL Simple Text",
            "ADP: PPL Normal Text"
        ]

        with open(os.path.join(output_path, "perplexity.csv"), "w", encoding="UTF8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for model_name in model_names:
                if model_name not in self.model_dict.keys():
                    raise ValueError("Invalid model name.")
                else:
                    data = [model_name]
                    for value in self.model_dict[model_name].values():
                        if value:
                            tokenizer = AutoTokenizer.from_pretrained(value)
                            model = AutoModelForCausalLM.from_pretrained(value)

                            simple_ppl = get_mod_ppl(
                                model,
                                tokenizer,
                                simple_texts
                            )
                            normal_ppl = get_mod_ppl(
                                model,
                                tokenizer,
                                normal_texts
                            )
                            data.extend(
                                [
                                    round(simple_ppl, 2),
                                    round(normal_ppl, 2)
                                ]
                            )
                        else:
                            data.extend([None, None])

                    writer.writerow(data)

    def simplicity_eval(
        self,
        model_names,
        input_prompts=None,
        logits_processor=None,
        max_new_tokens=100,
        penalty_alpha=0.6,
        top_k=4,
        num_beams=3,
        spacy_model="de_dep_news_trf",
        output_path="../evaluation"
    ):
        # Input Prompts for Readability Evaluation
        input_prompts = [
            "Das", "Heute", "Wir", "Die Türkei", "Dieses Haus", "Mein Vater"
        ] if input_prompts is None else input_prompts

        # Repetition Penalty
        logits_processor = LogitsProcessorList(
            [
                RepetitionPenaltyLogitsProcessor(penalty=3.0),
                TemperatureLogitsWarper(temperature=1.0)
            ]
        ) if logits_processor is None else logits_processor

        # SpaCy Pipeline
        nlp = spacy.load(spacy_model)
        nlp.add_pipe("syllables", after="tagger")

        header = [
            "Model Name",
            "ORIG: FRE",
            "ORIG: Average Word Log Frequency",
            "ORIG: Number of Newlines",
            "FT: FRE",
            "FT: Average Word Log Frequency",
            "FT: Number of Newlines",
            "ADP: FRE",
            "ADP: Average Word Log Frequency",
            "ADP: Number of Newlines"
        ]

        with open(os.path.join(output_path, "simplicity.csv"), "w", encoding="UTF8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for model_name in model_names:
                if model_name not in self.model_dict.keys():
                    raise ValueError("Invalid model name.")
                else:
                    data = [model_name]
                    for value in self.model_dict[model_name].values():
                        if value:
                            tokenizer = AutoTokenizer.from_pretrained(value)
                            model = AutoModelForCausalLM.from_pretrained(value)
                            batch_input_ids = [
                                tokenizer(prompt, return_tensors="pt").input_ids for prompt in input_prompts
                            ]

                            batch_output_ids = []

                            # Contrastive search
                            batch_output_ids.extend(
                                [
                                    model.generate(
                                        input_ids=input_ids,
                                        logits_processor=logits_processor,
                                        max_new_tokens=max_new_tokens,
                                        penalty_alpha=penalty_alpha,
                                        top_k=top_k,
                                        pad_token_id=model.config.eos_token_id
                                    ) for input_ids in batch_input_ids
                                ]
                            )

                            # Multinomial sampling
                            batch_output_ids.extend(
                                [
                                    model.generate(
                                        input_ids=input_ids,
                                        logits_processor=logits_processor,
                                        max_new_tokens=max_new_tokens,
                                        do_sample=True,
                                        pad_token_id=model.config.eos_token_id
                                    ) for input_ids in batch_input_ids
                                ]
                            )

                            # Beam search
                            batch_output_ids.extend(
                                [
                                    model.generate(
                                        input_ids=input_ids,
                                        logits_processor=logits_processor,
                                        max_new_tokens=max_new_tokens,
                                        num_beams=num_beams,
                                        do_sample=False,
                                        pad_token_id=model.config.eos_token_id
                                    ) for input_ids in batch_input_ids
                                ]
                            )

                            batch_output_texts = [
                                tokenizer.batch_decode(
                                    sequences=output_ids,
                                    skip_special_tokens=True
                                )[0] for output_ids in batch_output_ids
                            ]

                            docs = nlp.pipe(
                                batch_output_texts,
                                disable=['tok2vec', 'morphologizer', 'attribute_ruler', 'ner']
                            )

                            docs = list(docs)

                            avg_fre = mean(map(get_fre, docs))
                            avg_word_log_freq = mean(map(get_word_log_freq, docs))
                            num_newlines = sum(map(count_newlines, batch_output_texts))
                            data.extend(
                                [
                                    round(avg_fre, 2),
                                    round(avg_word_log_freq, 2),
                                    round(num_newlines, 2)
                                ]
                            )
                        else:
                            data.extend([None, None, None])

                    writer.writerow(data)


if __name__ == "__main__":
    model_list = ["german-gpt2", "gerpt2"]
    # model_list = ["german-gpt2"]
    evaluate = Evaluate()
    # evaluate.simplicity_eval(model_names=model_list)
    evaluate.perplexity_eval(model_names=model_list)
    print("End")
