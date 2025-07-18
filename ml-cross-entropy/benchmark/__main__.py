# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import functools
import gc
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from fire import Fire

from cut_cross_entropy import linear_cross_entropy
from cut_cross_entropy.constants import IGNORE_INDEX

from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
from torchtune.modules.loss import CEWithChunkedOutputLoss

from . import data, memory

def baseline(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    softcap: float | None = None,
    label_smoothing: float = 0.0,
    ignore_index: int = IGNORE_INDEX,
    reduction: str = "mean",
) -> torch.Tensor:
    logits = e @ c.T

    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap

    return F.cross_entropy(logits.float(), targets, ignore_index=ignore_index, reduction=reduction, label_smoothing=label_smoothing)

def liger(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    softcap: float | None = None,
    label_smoothing: float = 0.0,
    ignore_index: int = IGNORE_INDEX,
    reduction: str = "mean",
) -> torch.Tensor:
    loss_fn = LigerFusedLinearCrossEntropyLoss(ignore_index=ignore_index, reduction=reduction, label_smoothing=label_smoothing)
    return loss_fn(c, e, targets)


def torchtune(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    softcap: float | None = None,
    label_smoothing: float = 0.0,
    ignore_index: int = IGNORE_INDEX,
    reduction: str = "mean",
) -> torch.Tensor:
    logits = e @ c.T
    logits = logits.unsqueeze(0)
    logits = [chunk for chunk in logits.chunk(8, dim=1)]
    loss_fn = CEWithChunkedOutputLoss(ignore_index=ignore_index)
    return loss_fn(logits, targets.unsqueeze(0))


def clear_grad_fn(E, C, *args, **kwargs):
    E.grad = C.grad = None


def benchmark(
    methods: list[str] | str | None = None,
    test_data: list[str] | str | None = None,
    n_iteration: int = 50,
    n_rep: int = 0,
    dtype: str = "bfloat16",
    output: str | None = "benchmark.csv",
    kinds: list[str] | str | None = None,
    softcap: float | None = None,
    label_smoothing: float = 0.0,
):
    torch.set_float32_matmul_precision("high")

    if methods is None:
        methods = ["cce_orig", "cce", "cce_fused", "liger", "torchtune", "torch_compile", "baseline"]
    elif isinstance(methods, str):
        methods = methods.split(",")

    if kinds is None:
        kinds = ["loss-fw", "loss-bw", "loss-fw-bw"]
    elif isinstance(kinds, str):
        kinds = kinds.split(",")

    if test_data is None:
        test_data = ["gemma2", "llama2", "mistral", "llama3", "llama3.2-1", "tinyllama"]#, "mistral-nemo", "llama3.2-3", "llama3-70", "gemma2-9", "gemma2-27"]#, "phi3.5", "phi4"]
    elif isinstance(test_data, str):
        test_data = test_data.split(",")

    dtype = getattr(torch, dtype)

    all_stats = []

    for this_test_data in tqdm.tqdm(test_data, desc="Data source", disable=len(test_data) == 1):
        gen = data.generator(this_test_data)
        for rep in tqdm.trange(n_rep + 1, desc="Repetition"):
            D = gen(dtype=dtype)
            for kind in tqdm.tqdm(kinds, desc="Benchmark kind", disable=len(kinds) == 1):
                E, C, T = D.embedding, D.classifier, D.targets

                this_softcap = softcap if softcap is not None else D.softcap

                smoothing_values = [None, 0.0, 0.5, 1.0]#, 0.25, 0.5, 1.0]
                for smoothing in tqdm.tqdm(smoothing_values, desc="Smoothing", disable=len(smoothing_values) == 1):

                    for m in tqdm.tqdm(methods, desc="Method", leave=False):

                        kwargs: dict[str, Any] = {"softcap": this_softcap, "label_smoothing": smoothing}
                        if kind == "loss-fw":
                            E.requires_grad_(True)
                            C.requires_grad_(True)
                            args = (E, C, T)
                        elif kind in {"loss-bw", "loss-fw-bw"}:
                            E.requires_grad_(True)
                            C.requires_grad_(True)

                            args = (E, C, T)
                            kwargs["backward"] = True
                            kwargs["forward"] = kind == "loss-fw-bw"
                            kwargs["pre_fn"] = clear_grad_fn
                        else:
                            raise ValueError(f"Unknown kind {kind=}")

                        if smoothing == None and (m != "cce_orig" and m != "torchtune"):
                            continue

                        if m in "liger" and kind.startswith("lse"):
                            continue

                        # warmup (it==0)
                        stats = memory.Stats.measure(
                            baseline
                            if m == "baseline"
                            else 
                            liger if m == "liger"
                            else 
                            torchtune if m == "torchtune"
                            else functools.partial(linear_cross_entropy, impl=m),
                            *args,
                            n_iteration=n_iteration if rep > 0 else 1,
                            **kwargs,
                        )

                        if rep > 0 or n_rep == 0:
                            this_stats = {
                                "method": m,
                                "kind": kind,
                                "smoothing": smoothing,
                            } | asdict(stats)

                            this_stats["test_data"] = this_test_data

                            all_stats.append(this_stats)

                            torch.cuda.synchronize()
                            time.sleep(1)
                            gc.collect()
                            torch.cuda.empty_cache()
                            time.sleep(1)

        pd.options.display.float_format = "{:.1f}".format
        print(pd.DataFrame(all_stats))

    all_stats = pd.DataFrame(all_stats)
    pd.options.display.float_format = "{:.1f}".format
    print(all_stats)
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        all_stats.to_csv(output_path)


if __name__ == "__main__":
    Fire(benchmark)
