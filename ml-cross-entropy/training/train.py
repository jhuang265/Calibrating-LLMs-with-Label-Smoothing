# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence, cast

import datasets
import torch
import torch.distributed
import transformers
from torch.utils.data import Dataset

from cut_cross_entropy.transformers import cce_patch

IGNORE_INDEX = -100
SYSTEM_PROMPT = "You are a helpful AI assistant."
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n"
    ),
}

MODEL_NAME_MAP = {
    "gemma2": "google/gemma-2-2b-it",
    "phi3.5": "microsoft/Phi-3.5-mini-instruct",
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3.2-1": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2-3": "meta-llama/Llama-3.231B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral-nemo": "mistralai/Mistral-Nemo-Instruct-2407",
}

DATA_NAME_MAP = {"alpaca": "yahma/alpaca-cleaned"}


@dataclass
class ModelArguments:
    model_name: str
    attn_impl: str | None = None
    cross_entropy_impl: str = "cce"
    label_smoothing: float | None = None


@dataclass
class DataArguments:
    dataset_name: str = "alpaca"
    sequence_length: int = 512


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = False
    torch_compile: bool = False
    fp16: bool = False
    bf16: bool = True
    tf32: bool = True
    gradient_checkpoint: bool = True
    logging_strategy: str = "steps"
    logging_steps: int = 1
    warmup_ratio: float = 0.05
    dataloader_num_workers: int = 12
    dataloader_pin_memory: bool = True
    save_strategy: str = "no"
    save_steps: int = 400
    save_total_limit: int = 3
    num_train_epochs: float = 1.0
    gradient_checkpoint_kwargs: dict[str, Any] = field(
        default_factory=lambda: dict(use_reentrant=True)
    )


def download_hf(name: str, repo_type: str = "model"):
    if not Path(name).exists():
        subprocess.check_call(
            [
                "huggingface-cli",
                "download",
                "--exclude=original/*",
                f"--repo-type={repo_type}",
                name,
            ]
        )


def preprocess(
    source: str,
    target: str,
    tokenizer: transformers.PreTrainedTokenizer,
    uses_system_prompt: bool = True,
) -> dict:
    """Preprocess the data by tokenizing."""
    if uses_system_prompt:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
    else:
        messages = []

    messages.extend(
        (
            {"role": "user", "content": source},
            {"role": "assistant", "content": target},
        )
    )
    tokenization = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        return_dict=True,
    )
    input_ids = torch.as_tensor(tokenization["input_ids"])

    target_ids = tokenizer.encode(target, add_special_tokens=False, return_tensors="pt")[0]

    labels = input_ids.clone()
    for offset in reversed(range(0, len(input_ids) - len(target_ids))):
        if (labels[offset : offset + len(target_ids)] == target_ids).all():
            labels[0:offset] = IGNORE_INDEX
            break

    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_args: DataArguments,
        seed: int,
        tokenizer: transformers.PreTrainedTokenizer,
        split: str = "train",
        uses_system_prompt: bool = True,
    ):
        super().__init__()
        self.dataset = datasets.load_dataset(data_args.dataset_name, split="train")
        self.tokenizer = tokenizer
        self.uses_system_prompt = uses_system_prompt

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        element = self.dataset[i]
        if element["input"] == "":
            prompt_template = PROMPT_DICT["prompt_no_input"]
        else:
            prompt_template = PROMPT_DICT["prompt_input"]

        source = prompt_template.format_map(element)
        target = element["output"]

        return preprocess(source, target, self.tokenizer, self.uses_system_prompt)


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    pad_token_id: int | None
    padding_side: str

    def __call__(self, instances: Sequence[dict]) -> dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        max_len = max(len(v) for v in input_ids)
        assert self.pad_token_id is not None
        padded_input_ids = torch.full((len(input_ids), max_len), self.pad_token_id)
        padded_labels = torch.full((len(input_ids), max_len), IGNORE_INDEX)
        position_ids = (
            torch.arange(0, max_len, dtype=torch.int64).view(1, -1).repeat(len(input_ids), 1)
        )

        for i, (inp, lbl) in enumerate(zip(input_ids, labels, strict=True)):
            if self.padding_side == "right":
                slc = slice(len(inp))
            else:
                slc = slice(-len(inp), None)

            padded_input_ids[i, slc] = inp
            padded_labels[i, slc] = lbl
            position_ids[i, slc] -= position_ids[i, slc][0].item()

        return dict(
            input_ids=padded_input_ids,
            labels=padded_labels,
            attention_mask=padded_input_ids.ne(self.pad_token_id),
            position_ids=position_ids,
        )


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    seed,
    uses_system_prompt: bool = True,
) -> dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(
        data_args,
        seed=seed,
        tokenizer=tokenizer,
        uses_system_prompt=uses_system_prompt,
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer.pad_token_id, tokenizer.padding_side)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args = cast(TrainingArguments, training_args)
    model_args = cast(ModelArguments, model_args)
    data_args = cast(DataArguments, data_args)

    if model_args.model_name in MODEL_NAME_MAP:
        model_args.model_name = MODEL_NAME_MAP[model_args.model_name]

    if data_args.dataset_name in DATA_NAME_MAP:
        data_args.dataset_name = DATA_NAME_MAP[data_args.dataset_name]

    if torch.distributed.is_initialized():
        if training_args.local_rank == 0:
            download_hf(model_args.model_name)
            download_hf(data_args.dataset_name, "dataset")

        torch.distributed.barrier()

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name, use_fast=True)
    config = transformers.AutoConfig.from_pretrained(model_args.model_name)

    if config.model_type == "mistral":
        tokenizer.padding_side = "left"
        tokenizer.pad_token = "<pad>"
    elif config.model_type == "llama":
        tokenizer.pad_token = "<|reserved_special_token_0|>"

    attn_impl = model_args.attn_impl
    if attn_impl is None:
        attn_impl = "flash_attention_2" if config.model_type != "gemma2" else "eager"

    # This could be done instead. That will patch transformers code globally
    # cce_patch(config, model_args.cross_entropy_impl)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    model = model.to(device)

    model = cast(transformers.PreTrainedModel, model)

    model = cce_patch(
        model, 
        model_args.cross_entropy_impl,
        model_args.label_smoothing
    )

    data_module = make_supervised_data_module(
        tokenizer,
        data_args,
        training_args.seed,
        uses_system_prompt=config.model_type not in ("gemma2",),
    )

    os.environ["TOKENIZERS_PARALLELISM"] = "False"

    trainer = transformers.Trainer(
        model,
        training_args,
        tokenizer=tokenizer,
        **data_module,
    )

    trainer.train()

    if data_module.get("eval_dataset") is not None:
        trainer.evaluate()


if __name__ == "__main__":
    main()
