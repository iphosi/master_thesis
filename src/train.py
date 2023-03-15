import torch
import datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AdapterConfig,
    PfeifferConfig,
    HoulsbyConfig,
    ParallelConfig,
    CompacterConfig,
    CompacterPlusPlusConfig,
    PrefixTuningConfig,
    LoRAConfig,
    IA3Config,
    TrainingArguments
)
from preprocess import *
from datasets import Dataset
from trainer import ContrastiveTrainer

if __name__ == "__main__":
    # Device
    assert torch.cuda.is_available() is True

    # Model
    model_path = "dbmdz/german-gpt2"
    model_name = "german-gpt2"
    #model_path = "benjamin/gerpt2"
    #model_name = "gerpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    for p in model.parameters():
        p.requires_grad = False

    # Dataset
    dataset_df = concat_datasets()

    dataset = Dataset.from_pandas(dataset_df, preserve_index=False)
    column_names = dataset.column_names

    dataset = dataset.train_test_split(
        test_size=0.1,
        shuffle=True,
        seed=40
    )

    dataset = dataset.map(
        lambda batch: tokenizer(batch["phrase"]),
        remove_columns=column_names,
        batched=True
    )

    dataset = dataset.map(group_texts, batched=True)

    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Adapter
    adapter_name = "Prefix_Tuning"
    adapter_config = PrefixTuningConfig(
        flat=False,
        prefix_length=600,
        bottleneck_size=16,
        non_linearity="gelu",
        dropout=0.5
    )

    if adapter_name not in model.adapter_summary():
        model.add_adapter(adapter_name=adapter_name, config=adapter_config)
    else:
        pass

    assert adapter_name in model.adapter_summary()

    # Training
    model.train_adapter(adapter_setup=adapter_name)

    training_args = TrainingArguments(
        output_dir=f"../adapters/{model_name}/{adapter_name}/checkpoints",
        do_train=True,
        remove_unused_columns=False,
        label_smoothing_factor=0.1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=8e-3,
        weight_decay=0.01,
        gradient_accumulation_steps=4,
        num_train_epochs=6,
        save_strategy="steps",
        evaluation_strategy="steps",
        save_steps=1000,
        eval_steps=1000,
        logging_steps=1000,
        save_total_limit=4,
        overwrite_output_dir=True,
        load_best_model_at_end=True,
        disable_tqdm=True
    )

    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        margin=0.5
    )

    trainer.train()

    # Saving
    model.save_adapter(
        save_directory=f"../adapters/{model_name}/{adapter_name}/model",
        adapter_name=adapter_name
    )
