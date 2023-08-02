from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn

from timewarp.modules.layers.custom_self_attention import CustomSelfAttentionBase
from timewarp.modules.layers.kernel_attention import KernelAttention
from utilities.cache import Cache
from utilities.logger import TrainingLogger


class KernelSelfAttention(CustomSelfAttentionBase):
    def __init__(
        self,
        *,
        input_dim: int,
        num_heads: int,
        value_dim: int,
        attention: KernelAttention,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.value_dim = value_dim

        self.values_proj = nn.Linear(input_dim, num_heads * value_dim, bias=False)
        self.attention = attention

    def forward(
        self,
        src: Tensor,  # [B, num_points, input_dim]
        positions: Tensor,  # [B, num_points, D]
        masked_elements: torch.BoolTensor,  # [B, num_points]
        logger: Optional[TrainingLogger] = None,
        cache: Optional[Cache] = None,
    ) -> Tensor:  # [B, num_points, output_dim]
        values = self.values_proj(src)  # [B, seq_len, num_heads * value_dim]
        values = values.reshape(
            (values.shape[0], values.shape[1], self.num_heads, self.value_dim)
        )  # [B, seq_len, num_heads, value_dim]
        out = self.attention(
            query_positions=positions,
            key_positions=positions,
            values=values,
            masked_elements=masked_elements,
            cache=cache,
        )
        return out
