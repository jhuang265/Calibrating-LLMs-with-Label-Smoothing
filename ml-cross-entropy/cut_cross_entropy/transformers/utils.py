# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from dataclasses import dataclass
from typing import TypeVar

import transformers

TransformersModelT = TypeVar("TransformersModelT", bound=transformers.PreTrainedModel)


@dataclass
class PatchOptions:
    impl: str
    reduction: str
    label_smoothing: float
