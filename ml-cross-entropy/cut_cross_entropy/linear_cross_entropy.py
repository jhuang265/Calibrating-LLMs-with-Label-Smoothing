# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import enum
import platform
from enum import auto

import torch
import torch.nn as nn

from cut_cross_entropy.constants import IGNORE_INDEX
from cut_cross_entropy.doc import LINEAR_CROSS_ENTROPY_DOC, add_doc_start
from cut_cross_entropy.torch_compile import torch_compile_linear_cross_entropy


class LinearCrossEntropyImpl(enum.IntEnum):
    CCE = auto()
    CCE_FUSED = auto()
    CCE_ORIG = auto()
    TORCH_COMPILE = auto()


if platform.system() != "Darwin":
    from cut_cross_entropy.cce import cce_linear_cross_entropy, fused_cce_linear_cross_entropy

    LCE_IMPL_DEFAULT = LinearCrossEntropyImpl.CCE
else:
    cce_linear_cross_entropy = None
    LCE_IMPL_DEFAULT = LinearCrossEntropyImpl.TORCH_COMPILE


@add_doc_start(LINEAR_CROSS_ENTROPY_DOC)
def linear_cross_entropy(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
    softcap: float | None = None,
    reduction: str = "mean",
    shift: bool = False,
    filter_eps: float | str | None = "auto",
    impl: str | LinearCrossEntropyImpl = LCE_IMPL_DEFAULT,
    label_smoothing: float = 0.0
) -> torch.Tensor:
    """
    :param filter_eps: The threshold value used to determine which locations can be safely ignored
        in gradient computation. The default value of "auto" will automatically choose a value
        based on the input dtype. Only valid for the CCE implementation.
    :param impl: The linear cross entropy implementation to use. Currently supports cce and torch_compile.
    """

    if isinstance(impl, LinearCrossEntropyImpl):
        impl = impl.name.lower()

    match impl:
        case "cce":
            if platform.system() == "Darwin":
                raise RuntimeError(
                    "CCE does not support MacOS. Please use torch_compile when running on MacOS instead."
                )

            assert cce_linear_cross_entropy is not None
            return cce_linear_cross_entropy(
                e, c, targets, ignore_index, softcap, reduction, shift, filter_eps, label_smoothing
            )
        case "cce_fused":
            if platform.system() == "Darwin":
                raise RuntimeError(
                    "CCE does not support MacOS. Please use torch_compile when running on MacOS instead."
                )

            assert cce_linear_cross_entropy is not None
            return fused_cce_linear_cross_entropy(
                e, c, targets, ignore_index, softcap, reduction, shift, filter_eps, label_smoothing
            )
        case "cce_orig":
            if platform.system() == "Darwin":
                raise RuntimeError(
                    "CCE does not support MacOS. Please use torch_compile when running on MacOS instead."
                )

            assert cce_linear_cross_entropy is not None
            return cce_linear_cross_entropy(
                e, c, targets, ignore_index, softcap, reduction, shift, filter_eps, None
            )
        case "torch_compile":
            return torch_compile_linear_cross_entropy(
                e, c, targets, ignore_index, softcap, reduction, shift, label_smoothing
            )
        case _:
            raise NotImplementedError(f"{impl} is not implemented.")


class LinearCrossEntropy(nn.Module):
    def __init__(
        self,
        ignore_index: int = IGNORE_INDEX,
        softcap: float | None = None,
        reduction: str = "mean",
        filter_eps: float | str | None = "auto",
        shift: bool = False,
        impl: str | LinearCrossEntropyImpl = LCE_IMPL_DEFAULT,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.softcap = softcap
        self.reduction = reduction
        self.filter_eps = filter_eps
        self.shift = shift

        self.impl = impl
        self.label_smoothing = label_smoothing

    def forward(self, e: torch.Tensor, c: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return linear_cross_entropy(
            e,
            c,
            targets,
            self.ignore_index,
            self.softcap,
            reduction=self.reduction,
            filter_eps=self.filter_eps,
            shift=self.shift,
            impl=self.impl,
            label_smoothing=self.label_smoothing
        )
