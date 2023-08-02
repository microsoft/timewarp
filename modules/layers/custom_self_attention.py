import abc
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from utilities.cache import Cache

from utilities.logger import TrainingLogger


class CustomSelfAttentionBase(nn.Module, abc.ABC):
    """
    A self-attention module that uses positional information in some way,
    for instance to only attend over nearby features.
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(
        self,
        src: Tensor,  # [B, num_points, input_dim]
        positions: Tensor,  # [B, num_points, D]
        masked_elements: torch.BoolTensor,  # [B, num_points]
        logger: Optional[TrainingLogger] = None,
        cache: Optional[Cache] = None,
    ) -> Tensor:  # [B, num_points, output_dim]
        pass
