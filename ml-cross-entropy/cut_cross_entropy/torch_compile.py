# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import torch
import torch.nn.functional as F

from cut_cross_entropy.constants import IGNORE_INDEX
from cut_cross_entropy.doc import LINEAR_CROSS_ENTROPY_DOC, add_doc_start
from cut_cross_entropy.utils import (
    _build_flat_valids,
    handle_reduction_none,
    softcapping,
)


@torch.compile(fullgraph=True, dynamic=True)
def torch_compile_linear_cross_entropy_apply(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    softcap: float | None = None,
    *,
    ignore_index: int = IGNORE_INDEX,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    logits = e @ c.T

    if softcap is not None:
        logits = softcapping(logits, softcap)

    loss = F.cross_entropy(logits.float(), targets, ignore_index=ignore_index, reduction=reduction, label_smoothing=label_smoothing)

    return loss


@add_doc_start(LINEAR_CROSS_ENTROPY_DOC)
def torch_compile_linear_cross_entropy(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
    softcap: float | None = None,
    reduction: str = "mean",
    shift: bool = False,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    assert e.size()[0:-1] == targets.size()
    assert e.size(-1) == c.size(1)

    orig_b_size = targets.size()
    e = e.contiguous()
    targets = targets.contiguous()

    valids = _build_flat_valids(targets, ignore_index, shift)

    e = e.flatten(0, -2)
    targets = targets.flatten()

    if valids is not None:
        e = e[valids]
        targets = targets[(valids + 1) if shift else valids]

    loss = torch_compile_linear_cross_entropy_apply(
        e,
        c,
        targets,
        softcap,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing
    )

    if reduction == "none":
        loss = handle_reduction_none(orig_b_size, valids, shift, loss)

        if shift:
            loss = loss[..., 1:]

    # print(f"Torch Compile Params: smoothing={label_smoothing}")
    # print(f" => loss={loss}")

    return loss
