from dataclasses import dataclass
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable, Literal, List, Optional

from timewarp.modules.layers.custom_self_attention import CustomSelfAttentionBase
from timewarp.modules.layers.kernel_attention import (
    KernelAttention,
    LearnableChebyshevKernelAttention,
    LearnableLengthscaleKernelAttention,
)
from timewarp.modules.layers.kernel_self_attention import KernelSelfAttention
from timewarp.modules.layers.local_self_attention import LocalSelfAttention
from utilities.cache import Cache
from utilities.logger import TrainingLogger


ActivationFunctionName = Literal["relu", "gelu"]


class CustomTransformerEncoderLayer(nn.Module):
    r"""CustomTransformerEncoderLayer is made up of self-attn and feedforward network.
    This encoder layer is based on the paper "Attention Is All You Need".

    The one difference is that it allows for using other types of self-attention that
    depend on positional information in some way.

    Args:
        d_model: the number of expected features in the input (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).

    Examples::
        >>> encoder_layer = CustomTransformerEncoderLayer(d_model=512, self_attention=KernelSelfAttention(...))
        >>> src = torch.randn(10, 32, 512)
        >>> positions = torch.randn(10, 32, 3)
        >>> masked_elements = torch.zeros(10, 32, dtype=torch.bool)
        >>> out = encoder_layer(src, positions, masked_elements=masked_elements)

    """
    __constants__ = ["batch_first"]

    def __init__(
        self,
        *,
        d_model: int,
        self_attention: CustomSelfAttentionBase,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: ActivationFunctionName = "relu",
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        # Save the relevant parameters:
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward

        # Initialise other layers
        self.self_attn = self_attention
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        src: Tensor,  # [B, num_points, d_model]
        positions: Tensor,  # [B, num_points, D]
        masked_elements: torch.BoolTensor,  # [B, num_points]
        logger: Optional[TrainingLogger] = None,
        cache: Optional[Cache] = None,
    ) -> Tensor:  # [B, num_points, d_model]
        r"""
        Pass the input through the encoder layer.

        Mostly taken from the torch TransformerEncoderLayer implementation.

        Args:
            src: the sequence to the encoder layer.
            positions: the positional information associated with each element in the source sequence.
                This will be used by the self-attention layer to do position-informed attention in some way.
            masked_elements: True indicates which elements of the source sequence to ignore when doing attention.

        """
        src2 = self.self_attn(
            src=src,
            positions=positions,
            masked_elements=masked_elements,
            logger=logger,
            cache=cache,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


def _get_activation_fn(activation: ActivationFunctionName) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


@dataclass
class CustomAttentionEncoderLayerConfig:
    d_model: int  # Dimension of values, keys, queries in self attention
    dim_feedforward: int  # Dimension of hidden layer in the pointwise MLP in transformer block
    dropout: float  # Dropout rate in transformer block
    num_heads: int  # Number of heads in multihead attention.
    attention_type: str  # should be Literal["local", "kernel", "learnable_kernel", "chebyshev_kernel"], literal doesn't work with omegaconf
    lengthscales: Optional[List[float]] = None  # Only relevant for kernel attention
    max_radius: Optional[float] = None  # Only relevant for local attention
    normalise_kernel_values: Optional[bool] = None  # Only relevant for kernel attention
    cheb_order: Optional[int] = None  # Only relevant for Chebyshev attention
    force_asymptotic_zero: Optional[bool] = None  # Only relevant for Chebyshev attention


def custom_attention_transformer_encoder_constructor(
    config: CustomAttentionEncoderLayerConfig,
) -> CustomTransformerEncoderLayer:
    # forward type declaration
    self_attention: CustomSelfAttentionBase
    if config.attention_type == "local":
        assert config.max_radius is not None
        self_attention = LocalSelfAttention(
            input_dim=config.d_model,
            output_dim=config.d_model,
            num_heads=config.num_heads,
            value_dim=config.d_model,
            key_query_dim=config.d_model,
            max_radius=config.max_radius,
        )
    elif config.attention_type in {"kernel", "learnable_kernel"}:
        assert config.lengthscales is not None
        assert len(config.lengthscales) > 0
        if not config.num_heads == len(config.lengthscales):
            warnings.warn(
                f"Number of lengthscales ({len(config.lengthscales)}) not equal number of heads of the transformer ({config.num_heads}). Using {len(config.lengthscales)} heads instead."
            )

        # Translate from attention_type str to correct class
        attention_cls = {
            "kernel": KernelAttention,
            "learnable_kernel": LearnableLengthscaleKernelAttention,
        }

        def make_attention(cls) -> KernelAttention:
            return cls(
                value_dim=config.d_model,
                output_dim=config.d_model,
                lengthscales=config.lengthscales,
                normalise_kernel_values=config.normalise_kernel_values,
            )

        assert config.normalise_kernel_values is not None
        self_attention = KernelSelfAttention(
            input_dim=config.d_model,
            num_heads=len(config.lengthscales),
            value_dim=config.d_model,
            attention=make_attention(attention_cls[config.attention_type]),
        )
    elif config.attention_type == "chebyshev_kernel":
        assert config.lengthscales is not None
        assert len(config.lengthscales) > 0
        if not config.num_heads == len(config.lengthscales):
            warnings.warn(
                f"Number of lengthscales ({len(config.lengthscales)}) not equal number of heads of the transformer ({config.num_heads}). Using {len(config.lengthscales)} heads instead."
            )
        assert config.cheb_order is not None
        assert config.cheb_order >= 1
        assert config.force_asymptotic_zero is not None

        assert config.normalise_kernel_values is not None
        attention = LearnableChebyshevKernelAttention(
            value_dim=config.d_model,
            output_dim=config.d_model,
            lengthscales=config.lengthscales,
            cheb_order=config.cheb_order,
            normalise_kernel_values=config.normalise_kernel_values,
            force_asymptotic_zero=config.force_asymptotic_zero,
        )
        assert config.normalise_kernel_values is not None
        self_attention = KernelSelfAttention(
            input_dim=config.d_model,
            num_heads=len(config.lengthscales),
            value_dim=config.d_model,
            attention=attention,
        )
    else:
        raise RuntimeError(f"Unknown attention type: {config.attention_type}.")

    return CustomTransformerEncoderLayer(
        d_model=config.d_model,
        self_attention=self_attention,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
    )
