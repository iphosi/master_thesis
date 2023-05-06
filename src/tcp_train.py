import torch
import os

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from transformers import EarlyStoppingCallback

from preprocess import get_text_complexity_dataset, split_dataset, specify_config

from transformers.adapters import (
    AdapterConfig,
    PfeifferConfig,
    HoulsbyConfig,
    ParallelConfig,
    CompacterConfig,
    CompacterPlusPlusConfig,
    PrefixTuningConfig,
    LoRAConfig,
    IA3Config
)

from transformers import TrainingArguments, AdapterTrainer, DataCollatorWithPadding

from argparse import ArgumentParser

import wandb


def train_adapter(
    device=None,
    adapter_type=None,
    adapter_name=None,
    adapter_dict=None,
    checkpoint_name=None,
    model_path="malteos/bloom-350m-german",
    data_path="../datasets/TextComplexity/monolingual",
    batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=4e-4,
    warmup_steps=200,
    num_train_epochs=2,
    num_steps=1500,
    early_stopping_patience=4,
    train_head=False,
    text_column_name="phrase",
    target_label="WLF",
    do_rescaling=False,
    max_value=10,
    min_value=0,
    rescaling_factor=100,
    disable_tqdm=False
):
    assert target_label in ["MOS", "FRE", "WLF"]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.exists(model_path):
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model_name = model_path.split("/")[-2]
    else:
        model, tokenizer = specify_config(model_path, head_type="regression")
        model_name = model_path.split("/")[-1]

    print("-"*50)
    print(f"Baseline model: {model_name}")

    model.to(device)

    # Dataset
    if hasattr(model.config, "n_positions"):
        max_length = model.config.n_positions
    elif hasattr(model.config, "seq_length"):
        max_length = model.config.seq_length
    else:
        max_length = 1024

    print("-" * 50)
    print(f"Dataset: {target_label}")

    dataset = get_text_complexity_dataset(
        tokenizer=tokenizer,
        max_length=max_length,
        text_column_name=text_column_name,
        target_label=target_label,
        do_rescaling=do_rescaling,
        max_value=max_value,
        min_value=min_value,
        rescaling_factor=rescaling_factor,
        input_path=data_path
    )

    sample_label = dataset[0]["labels"]
    print(f"Label of the first sample: {sample_label:.2f}")

    train_set, val_set, test_set = split_dataset(dataset=dataset, val_split=0.1, test_split=0.1)

    # Adapter
    adapter_dict = {
        f"Adapter_Bottleneck_Sequential": AdapterConfig(
            mh_adapter=False,
            output_adapter=True,
            reduction_factor=16,
            non_linearity="gelu"
        ),
        f"Compacter++": CompacterPlusPlusConfig(
            reduction_factor=8,
            phm_dim=4,
            phm_rank=16,
            shared_phm_rule=False
        )
    } if adapter_dict is None else adapter_dict

    adapter_config = adapter_dict[adapter_type]

    adapter_name = f"{adapter_type}_{target_label}" if adapter_name is None else adapter_name

    if checkpoint_name:
        checkpoint_path = f"../adapters/{model_name}/{adapter_name}/pretrained/{checkpoint_name}/{adapter_name}"
    else:
        checkpoint_path = ""

    print("-" * 50)
    if os.path.exists(checkpoint_path):
        print("Resume from checkpoint.")
        model.load_adapter(adapter_name_or_path=checkpoint_path)
    else:
        print("Train the adapter from scratch.")
        model.add_adapter(adapter_name=adapter_name, config=adapter_config, overwrite_ok=True)

    assert adapter_name in model.adapter_summary()

    # Training
    model.train_adapter(adapter_setup=adapter_name)
    print(model.adapter_summary())

    if train_head:
        for module in model.named_modules():
            if module[0] == "lm_head":
                for parameter in module[1].parameters():
                    parameter.requires_grad = True
                break

    training_args = TrainingArguments(
        report_to=["wandb"],
        output_dir=f"../adapters/{model_name}/{adapter_name}/checkpoints",
        do_train=True,
        remove_unused_columns=False,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type="linear",
        weight_decay=0.01,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=True,
        warmup_steps=warmup_steps,
        num_train_epochs=num_train_epochs,
        save_strategy="steps",
        evaluation_strategy="steps",
        save_steps=num_steps,
        eval_steps=num_steps,
        logging_steps=num_steps,
        save_total_limit=2,
        overwrite_output_dir=True,
        load_best_model_at_end=True,
        disable_tqdm=disable_tqdm
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=data_collator
    )

    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    trainer.train()

    wandb.finish()

    model.save_adapter(
        save_directory=f"../adapters/{model_name}/{adapter_name}/model",
        adapter_name=adapter_name
    )


if __name__ == "__main__":
    wandb.login()

    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--adapter_type", type=str, default="Adapter_Bottleneck_Sequential")
    parser.add_argument("--adapter_name", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="malteos/bloom-350m-german")
    parser.add_argument("--data_path", type=str, default="../datasets/TextComplexity/monolingual")
    parser.add_argument("--checkpoint_name", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=4e-4)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--num_train_epochs", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=1500)
    parser.add_argument("--early_stopping_patience", type=int, default=4)
    parser.add_argument("--train_head", action="store_true")

    parser.add_argument("--text_column_name", type=str, default="phrase")
    parser.add_argument("--target_label", type=str, default="WLF")
    parser.add_argument("--do_rescaling", action="store_true")
    parser.add_argument("--max_value", type=float, default=100)
    parser.add_argument("--min_value", type=float, default=0)
    parser.add_argument("--rescaling_factor", type=float, default=10)

    parser.add_argument("--disable_tqdm", action="store_true")

    args = parser.parse_args()

    train_adapter(
        device=args.device,
        adapter_type=args.adapter_type,
        adapter_name=args.adapter_name,
        model_path=args.model_path,
        data_path=args.data_path,
        checkpoint_name=args.checkpoint_name,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        num_steps=args.num_steps,
        early_stopping_patience=args.early_stopping_patience,
        text_column_name=args.text_column_name,
        target_label=args.target_label,
        do_rescaling=args.do_rescaling,
        max_value=args.max_value,
        min_value=args.min_value,
        rescaling_factor=args.rescaling_factor,
        disable_tqdm=args.disable_tqdm
    )





