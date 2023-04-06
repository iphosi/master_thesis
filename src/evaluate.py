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
from validity import *
from rsa import *
from textstat import textstat


class Evaluate:
    def __init__(
        self,
        model_dict=None,
        target_error_types=None,
        language="de",
        region="DE"
    ):
        self.model_dict = {
            "german-gpt2": {
                "ORIG": "dbmdz/german-gpt2",
                "FT": "MiriUll/german-gpt2_easy",
                # "ADP_BN_S_r0_8": "../adapters/german-gpt2/Adapter_Bottleneck_Sequential/model_r0_8",
                # "ADP_BN_S_r4": "../adapters/german-gpt2/Adapter_Bottleneck_Sequential/model_r4",
                # "ADP_BN_S_r8": "../adapters/german-gpt2/Adapter_Bottleneck_Sequential/model_r8",
                # "ADP_BN_S_r16": "../adapters/german-gpt2/Adapter_Bottleneck_Sequential/model_r16",
                # "ADP_BN_S_r32": "../adapters/german-gpt2/Adapter_Bottleneck_Sequential/model_r32",
                # "ADP_BN_P_r4_s2": "../adapters/german-gpt2/Adapter_Bottleneck_Parallel/model_r4_s2",
                # "ADP_BN_P_r4_s4": "../adapters/german-gpt2/Adapter_Bottleneck_Parallel/model_r4_s4",
                # "ADP_BN_P_r8_s4": "../adapters/german-gpt2/Adapter_Bottleneck_Parallel/model_r8_s4",
                # "ADP_BN_P_r8_s8": "../adapters/german-gpt2/Adapter_Bottleneck_Parallel/model_r8_s8",
                # "ADP_PFX_b96_p30": "../adapters/german-gpt2/Prefix_Tuning/model_b96_p30",
                # "ADP_PFX_b192_p30": "../adapters/german-gpt2/Prefix_Tuning/model_b192_p30",
                # "ADP_PFX_b192_p60": "../adapters/german-gpt2/Prefix_Tuning/model_b192_p60",
                # "ADP_COMP": "../adapters/german-gpt2/Compacter/model",
                # "ADP_LoRA": "../adapters/german-gpt2/Adapter_LoRA/model",
            },
            "bloom-350m-german": {
                "ORIG": "../adapters/bloom-350m-german/Orig",
                "ADP_BN_S_r16": "../adapters/bloom-350m-german/Adapter_Bottleneck_Sequential/model_r16"
            }
        } if model_dict is None else model_dict
        self.target_error_types = target_error_types
        self.language = language
        self.region = region
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        textstat.set_lang(self.language)

    def ppl_eval(
        self,
        model_name,
        leave_out=None,
        max_length=None,
        stride=512,
        input_path="../datasets/aligned German simplification/mdr_aligned_news.csv",
        output_path="../evaluation"
    ):
        print("-" * 50)
        print("Evaluating perplexity:")

        if leave_out:
            first = leave_out[0] + 1
            last = leave_out[-1] + 1
            output_path = os.path.join(
                output_path,
                f"{model_name}/leave_out/L{first}-{last}/perplexity.csv"
            )
        else:
            output_path = os.path.join(
                output_path,
                f"{model_name}/leave_out/Full/perplexity.csv"
            )

        # Dataset for Perplexity Evaluation
        dataset_df = pd.read_csv(input_path)

        normal_texts = dataset_df.dropna(subset=["normal_phrase"])["normal_phrase"].values.tolist()
        simple_texts = dataset_df.dropna(subset=["simple_phrase"])["simple_phrase"].values.tolist()

        columns = [
            "Model Name",
            "Tuning Method",
            "%Parameter",
            "PPL Simple Text",
            "PPL Normal Text",
        ]

        data = []

        if model_name not in self.model_dict.keys():
            raise ValueError("Invalid model name.")
        else:
            for tuning_method, model_path in self.model_dict[model_name].items():
                print(f"{model_name} | {tuning_method}")

                if model_path:
                    if tuning_method.startswith("ADP"):
                        tokenizer = AutoTokenizer.from_pretrained(self.model_dict[model_name]["ORIG"])
                        model = AutoModelForCausalLM.from_pretrained(self.model_dict[model_name]["ORIG"])
                        model.load_adapter(model_path, leave_out=leave_out)

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

                    model.eval()

                    simple_ppl = get_mod_ppl(
                        model=model,
                        tokenizer=tokenizer,
                        texts=simple_texts,
                        max_length=max_length,
                        stride=stride,
                        device=self.device
                    )
                    normal_ppl = get_mod_ppl(
                        model=model,
                        tokenizer=tokenizer,
                        texts=normal_texts,
                        max_length=max_length,
                        stride=stride,
                        device=self.device
                    )

                    param = sum(
                        summary["%param"] for summary in model.adapter_summary(as_dict=True)
                        if summary["name"] != "Full model"
                    )

                    if tuning_method.startswith("ADP"):
                        data.append(
                            [model_name, tuning_method, round(param, 2), round(simple_ppl, 2), round(normal_ppl, 2)]
                        )
                    else:
                        data.append(
                            [model_name, tuning_method, 100.00, round(simple_ppl, 2), round(normal_ppl, 2)]
                        )

                else:
                    print("Model path is not defined.")

        df = pd.DataFrame(data=data, columns=columns)
        df.to_csv(output_path, index=False)

    def simp_val_eval(
        self,
        model_name,
        leave_out,
        spacy_model="de_dep_news_trf",
        input_path="../evaluation",
        output_path="../evaluation"
    ):
        print("-" * 50)
        print("Evaluating simplicity and validity:")

        if leave_out:
            first = leave_out[0] + 1
            last = leave_out[-1] + 1
            input_path = os.path.join(
                input_path,
                f"{model_name}/leave_out/L{first}-{last}/generated_texts.json"
            )
            output_path = os.path.join(
                output_path,
                f"{model_name}/leave_out/L{first}-{last}/simplicity_validity.csv"
            )
        else:
            input_path = os.path.join(
                input_path,
                f"{model_name}/leave_out/Full/generated_texts.json"
            )
            output_path = os.path.join(
                output_path,
                f"{model_name}/leave_out/Full/simplicity_validity.csv"
            )

        # SpaCy Pipeline
        nlp = spacy.load(spacy_model)

        text_df = pd.read_json(input_path)

        columns = [
            "Model Name",
            "Tuning Method",
            "Average FRE",
            "Average Word Log Frequency",
            "Number of Sentences",
            "Number of Newlines",
            "Number of Errors"
        ]

        data = []

        if model_name not in self.model_dict.keys():
            raise ValueError("Invalid model name.")
        else:
            for tuning_method, model_path in self.model_dict[model_name].items():
                print(f"{model_name} | {tuning_method}")
                if model_path:
                    texts = text_df.loc[
                        (text_df["Model Name"] == model_name) & (text_df["Tuning Method"] == tuning_method)
                    ]["Generated Texts"].values[0]
                    docs = nlp.pipe(
                        texts,
                        disable=['tok2vec', 'morphologizer', 'attribute_ruler', 'ner']
                    )

                    docs = list(docs)

                    avg_fre = mean(map(textstat.flesch_reading_ease, texts))
                    avg_word_log_freq = mean(map(get_log_freq, docs))
                    num_sentences = sum(map(count_sentences, docs))
                    num_newlines = sum(map(count_newlines, texts))
                    num_errors = count_errors(
                        texts=texts,
                        language=f"{self.language}_{self.region}",
                        target_error_types=self.target_error_types,
                        return_errors=False
                    )

                    data.append(
                        [
                            model_name,
                            tuning_method,
                            round(avg_fre, 2),
                            round(avg_word_log_freq, 2),
                            num_sentences,
                            num_newlines,
                            num_errors
                        ]
                    )

                else:
                    print("Model path is not defined.")

        df = pd.DataFrame(data=data, columns=columns)
        df.to_csv(output_path, index=False)

    def generate_text(
        self,
        model_name,
        leave_out=None,
        input_prompts=None,
        logits_processor=None,
        max_new_tokens=100,
        temperature=1.0,
        repetition_penalty=1.4,
        penalty_alpha=1.6,
        top_k=4,
        top_p=0.7,
        num_beams=3,
        no_repeat_ngram_size=2,
        early_stopping=True,
        output_path="../evaluation"
    ):
        print("-"*50)
        print("Generating texts for simplicity and validity evaluation:")

        if leave_out:
            first = leave_out[0] + 1
            last = leave_out[-1] + 1
            output_path = os.path.join(
                output_path,
                f"{model_name}/leave_out/L{first}-{last}/generated_texts.json"
            )
        else:
            output_path = os.path.join(
                output_path,
                f"{model_name}/leave_out/Full/generated_texts.json"
            )

        # Input Prompts for Readability Evaluation
        input_prompts = [
            "Das", "Heute", "Wir", "Die Türkei", "Dieses Haus", "Mein Vater"
        ] if input_prompts is None else input_prompts

        # Logits Processor
        """
        Example:
        logits_processor = LogitsProcessorList(
            [
                RepetitionPenaltyLogitsProcessor(penalty=3.0),
                TemperatureLogitsWarper(temperature=1.0)
            ]
        ) if logits_processor is None else logits_processor
        """

        columns = [
            "Model Name",
            "Tuning Method",
            "Generated Texts"
        ]

        data = []

        if model_name not in self.model_dict.keys():
            raise ValueError("Invalid model name.")
        else:
            for tuning_method, model_path in self.model_dict[model_name].items():
                print(f"{model_name} | {tuning_method}")

                if model_path:
                    if tuning_method.startswith("ADP"):
                        tokenizer = AutoTokenizer.from_pretrained(self.model_dict[model_name]["ORIG"])
                        model = AutoModelForCausalLM.from_pretrained(self.model_dict[model_name]["ORIG"])
                        model.load_adapter(model_path, leave_out=leave_out)

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

                    if model.config.pad_token_id:
                        pad_token_id = model.config.pad_token_id
                    else:
                        pad_token_id = model.config.eos_token_id
                    model.eval()

                    batch_input_ids = [
                        tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
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
                                pad_token_id=pad_token_id,
                                no_repeat_ngram_size=no_repeat_ngram_size
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
                                top_k=top_k,
                                top_p=top_p,
                                pad_token_id=pad_token_id,
                                no_repeat_ngram_size=no_repeat_ngram_size
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
                                pad_token_id=pad_token_id,
                                no_repeat_ngram_size=no_repeat_ngram_size,
                                early_stopping=early_stopping
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

        df = pd.DataFrame(data=data, columns=columns)
        df.to_json(output_path)

    def rsa(
        self,
        model_name,
        leave_out=None,
        num_sample_texts=30,
        num_sample_tokens=500,
        seed=40,
        use_cpu=True,
        input_path="../datasets/aligned German simplification/mdr_aligned_news.csv",
        output_path="../evaluation",
    ):
        print("-"*50)
        print("Analyzing representational similarity:")

        output_path = os.path.join(
            output_path,
            f"{model_name}/similarity.csv"
        )

        dataset_df = pd.read_csv(input_path)
        dataset_df = pd.concat(
            [dataset_df["normal_phrase"], dataset_df["simple_phrase"]],
            ignore_index=True
        ).dropna()

        sample_texts = dataset_df.sample(
            n=num_sample_texts,
            random_state=40
        ).values.tolist()

        columns = [
            "Model Name",
            "Source | Target",
            "Embedding Layer"
        ]

        data = []

        device = "cpu" if use_cpu else self.device

        if model_name not in self.model_dict.keys():
            raise ValueError("Invalid model name.")
        else:
            src_tokenizer = None
            src_model = None
            tgt_tokenizer = None
            tgt_model = None

            for tuning_method, model_path in self.model_dict[model_name].items():

                if model_path:
                    if tuning_method == "ORIG":
                        src_tokenizer = AutoTokenizer.from_pretrained(model_path)
                        src_model = AutoModelForCausalLM.from_pretrained(model_path)
                        for i in range(src_model.config.n_layer):
                            columns.append(f"Attention Layer {i + 1}")

                    elif tuning_method.startswith("ADP"):
                        tgt_tokenizer = AutoTokenizer.from_pretrained(self.model_dict[model_name]["ORIG"])
                        tgt_model = AutoModelForCausalLM.from_pretrained(self.model_dict[model_name]["ORIG"])
                        tgt_model.load_adapter(model_path, leave_out=leave_out)

                        adapter_dicts = tgt_model.adapter_summary(as_dict=True)
                        adapter_names = [
                            adapter_dict["name"] for adapter_dict in adapter_dicts
                            if adapter_dict["name"] != "Full model"
                        ]
                        for name in adapter_names:
                            tgt_model.set_active_adapters(name)

                    else:
                        tgt_tokenizer = AutoTokenizer.from_pretrained(model_path)
                        tgt_model = AutoModelForCausalLM.from_pretrained(model_path)

                    if src_model and tgt_model:

                        src_model.to(device)
                        tgt_model.to(device)

                        src_model.eval()
                        tgt_model.eval()

                        print("-"*50)
                        print(f"Source Model: {model_name} | ORIG")
                        print(f"Target Model: {model_name} | {tuning_method}")

                        src_rep_spaces = get_rep_spaces(
                            model=src_model,
                            tokenizer=src_tokenizer,
                            texts=sample_texts,
                            device=device,
                            num_sample_tokens=num_sample_tokens,
                            seed=seed
                        )
                        tgt_rep_spaces = get_rep_spaces(
                            model=tgt_model,
                            tokenizer=tgt_tokenizer,
                            texts=sample_texts,
                            device=device,
                            num_sample_tokens=num_sample_tokens,
                            seed=seed
                        )

                        scores = get_pearson_scores(src_rep_spaces, tgt_rep_spaces, device)

                        data.append([model_name, f"ORIG | {tuning_method}"] + scores)

                else:
                    print("Model path is not defined.")

        df = pd.DataFrame(data=data, columns=columns)
        df.to_csv(output_path, index=False)


if __name__ == "__main__":
    model_list = [
        "bloom-350m-german",
        # "german-gpt2"
    ]
    model_idx = 0
    evaluate = Evaluate()
    for layer_range in range(25):
        leave_out_layers = [layer for layer in range(layer_range)]
        # evaluate.ppl_eval(model_name=model_list[model_idx], leave_out=leave_out_layers)
        # evaluate.generate_text(model_name=model_list[model_idx], leave_out=leave_out_layers)
        evaluate.simp_val_eval(model_name=model_list[model_idx], leave_out=leave_out_layers)
    # evaluate.rsa(model_name=model_list[model_idx], use_cpu=True)
    print("End")
