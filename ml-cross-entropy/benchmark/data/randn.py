# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import torch

from .data import Data


def generate(
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    M: int = 256,  # 16 * 1024,
    N: int = 512,
    D: int = 128,
    fraction_invalid_labels: float = 0.2,
) -> Data:
    """Random data generation

    Args:
    ----
        device (str, optional): Cuda device. Defaults to "cuda".
        dtype (torch.dtype, optional): Tensor type (torch.float32 or torch.bfloat16). Defaults to torch.bfloat16.
        M (int, optional): Sequence length. Defaults to 16000.
        N (int, optional): Vocabulary size. Defaults to 128000.
        D (int, optional): Embedding dimension. Defaults to 4096.

    Returns:
    -------
        Data: A data sample.

    """
    W = torch.randn(N, D, device=device, dtype=dtype, requires_grad=False) / D**0.25
    x = torch.randn(M, D, device=device, dtype=dtype, requires_grad=False) / D**0.25
    # Get some values that are non-zero in expectation
    W[:M] = x[: min(N, M)]
    targets = torch.randint(0, N, size=(M,), device=device)
    # targets[0 : int(M * fraction_invalid_labels)] = -100
    targets = targets[torch.randperm(M, device=targets.device)]

    return Data(x, W, targets)
