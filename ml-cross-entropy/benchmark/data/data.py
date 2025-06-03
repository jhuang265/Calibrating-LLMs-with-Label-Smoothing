# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from dataclasses import dataclass

import torch


@dataclass
class Data:
    embedding: torch.Tensor
    classifier: torch.Tensor
    targets: torch.Tensor
    softcap: float | None = None
    smoothing: float | None = None

    @property
    def required_storage(self) -> float:
        return (
            self.embedding.element_size() * self.embedding.numel()
            + self.classifier.element_size() * self.classifier.numel()
        )
