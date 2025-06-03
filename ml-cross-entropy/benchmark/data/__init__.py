# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import functools
from collections.abc import Callable

from .data import Data
from .models import generate_test_data_otf, load_model
from .randn import generate as randn_generate

generators: dict[str, Callable[..., Data]] = {
    "llama3": functools.partial(
        generate_test_data_otf,
        "meta-llama/Meta-Llama-3-8B-Instruct",
    ),
    "llama3.2-1": functools.partial(
        generate_test_data_otf,
        "meta-llama/Llama-3.2-1B-Instruct",
    ),
    "llama3.2-3": functools.partial(
        generate_test_data_otf,
        "meta-llama/Llama-3.2-3B-Instruct",
    ),
    "llama3-70": functools.partial(
        generate_test_data_otf,
        "meta-llama/Meta-Llama-3-70B-Instruct",
    ),
    "llama2": functools.partial(
        generate_test_data_otf,
        "meta-llama/Llama-2-7b-chat-hf",
    ),
    "gemma2": functools.partial(
        generate_test_data_otf, 
        "google/gemma-2-2b-it"
    ),
    "gemma2-9": functools.partial(
        generate_test_data_otf, 
        "google/gemma-2-9b-it"
    ),
    "gemma2-27": functools.partial(
        generate_test_data_otf, 
        "google/gemma-2-27b-it"
    ),
    "phi3.5": functools.partial(
        generate_test_data_otf, 
        "microsoft/Phi-3.5-mini-instruct"
    ),
    "phi4": functools.partial(
        generate_test_data_otf, 
        "microsoft/phi-4"
    ),
    "mistral-nemo": functools.partial(
        generate_test_data_otf, 
        "mistralai/Mistral-Nemo-Instruct-2407"
    ),
    "mistral": functools.partial(
        generate_test_data_otf, 
        "mistralai/Mistral-7B-Instruct-v0.3"
    ),
    "tinyllama": functools.partial(
        generate_test_data_otf, 
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )

}

generators = generators | {
    f"{k}-invalids": functools.partial(v, keep_invalids=True) for k, v in generators.items()
}

generators["randn"] = randn_generate

all_fig1_models = [
    "google/gemma-2-2b",
    "google/gemma-2b",
    "meta-llama/Llama-2-7b-chat-hf",
    "microsoft/Phi-3.5-mini-instruct",
    "meta-llama/Meta-Llama-3-8B",
    "google/gemma-2-9b",
    "meta-llama/Meta-Llama-3-70B",
    "meta-llama/Llama-2-13b-chat-hf",
    "openai-community/gpt2",
    "mistralai/Mistral-7B-v0.1",
    "microsoft/phi-1_5",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-2.7B",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "google/gemma-2-27b-it",
    "microsoft/Phi-3-medium-128k-instruct",
]


def generator(name: str) -> Callable[..., Data]:
    if name not in generators:
        raise ValueError(f"Data generator {name!r} not found.")

    load_model.cache_clear()
    return generators[name]
