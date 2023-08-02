from typing import Callable, Optional, Sequence
import torch
import torch.nn as nn
from torch import Tensor

from utilities.cache import Cache, NullaryClosure


def gaussian_basis_function(alpha: torch.Tensor) -> torch.Tensor:
    return torch.exp(-(alpha**2))


def chebyshev_basis_function(
    scaled_distances: torch.Tensor,
    cheb_order: int,
    cheb_coeffs: torch.Tensor,
    force_asymptotic_zero: bool,
) -> torch.Tensor:
    # Inputs:
    #   scaled_distances: [B, num_heads, query_len, memory_len]
    #   cheb_coeffs: [num_heads, cheb_order]

    # The expanded function $F: \R_+ \to \R$ satisfies
    # $\lim_{x \to \infty} F(x) = 0$, iff the sum of the coefficients
    # is zero.
    if force_asymptotic_zero:
        cheb_coeffs = cheb_coeffs - torch.mean(cheb_coeffs, dim=1, keepdim=True)

    # First square then expand the scaled distances
    cheb = chebyshev_expansion(scaled_distances**2, cheb_order)
    # cheb: [B, num_heads, cheb_order, query_len, memory_len]

    # Then evaluate the approximant by inner-product with the coefficient
    return torch.einsum("bhcqm,hc->bhqm", cheb, cheb_coeffs)


def chebyshev_expansion(scaled_distances: torch.Tensor, cheb_order: int):
    """Expand a distance Tensor into a Chebyshev rational function basis."""
    assert cheb_order >= 1
    # scaled_distances: [B, num_heads, query_len, memory_len]

    # Linearly construct the recursion
    #    R_{n+1}(x) = 2 ((x-1)/(x+1)) R_n(x) - R_{n-1}(x),
    # where R_0(x) = 1 and R_1(x) = (x-1)/(x+1).
    # This is using the three-term recursion (Algorithm I) for T_n, and
    # the fact that R_n(x) = T_n((x-1)/(x+1)).
    # The three-term recursion is analyzed in
    # [(Smoktunowicz et al., 2018)](https://arxiv.org/pdf/1312.5677.pdf),
    # where it is shown to be numerically stable.
    rprev = torch.ones_like(scaled_distances)
    rfactor = (scaled_distances - 1.0) / (scaled_distances + 1.0)
    rcur = rfactor

    expansion_result = [rprev]
    if cheb_order >= 2:
        expansion_result.append(rcur)

    for ri in range(2, cheb_order):
        rnext = 2.0 * rfactor * rcur - rprev
        expansion_result.append(rnext)
        rcur, rprev = rnext, rcur

    expansion_result = torch.stack(expansion_result, dim=2)
    # [B, num_heads, cheb_order, query_len, memory_len]

    return expansion_result


def compute_kernel_attention_scores(
    query: Tensor,  # [B, query_len, pos_dim]
    key: Tensor,  # [B, memory_len, pos_dim]
    masked_elements: torch.BoolTensor,  # [B, memory_len]
    lengthscales: Tensor,  # [num_heads]
    basis_function: Callable[[Tensor], Tensor] = gaussian_basis_function,
    normalise_kernel_values: bool = True,
    distance_compute_mode: str = "use_mm_for_euclid_dist_if_necessary",
) -> Tensor:  # [B, num_heads, query_len, memory_len]
    """Compute kernel-based attention scores from `query` and `key`.

    Args:
        query: A tensor of shape `[batch_len, query_len, dim]`.
        key: A tensor of shape `[batch_len, memory_len, dim]`.
        masked_elements: A tensor of shape `[batch_len, memory_len]` representing
            which elements to assign non-zero attention to.
        lengthscales: A tensor of shape `[num_heads, ]` representing the length-scales for
            the different heads.
        basis_function: A callable mapping the scaled distances to
            the evaluation of the kernel. Kernel is assumed to be stationary.
        normalise_kernel_values: If `True`, the attention scores will be normalised.
            Otherwise, attention scores will be left as returned from `basis_function`.
        distance_compute_mode: keyword argument passed to `torch.cdist` specifying
            which method to use for computing the distances.

    Returns:
        attention_scores: A tensor of shape `[batch_len, num_heads, query_len, memory_len]`
            representing the attention weights.
    """
    distances = torch.cdist(
        query,
        key,
        compute_mode=distance_compute_mode,
    )  # [batch_size, query_len, memory_len]

    # Number of lengthscales is the number of heads of this transformer
    scaled_distances = distances.unsqueeze(-3).expand(
        -1, len(lengthscales), -1, -1
    )  # [B, num_heads, query_len, memory_len]
    scaled_distances = (
        scaled_distances / lengthscales[None, :, None, None]
    )  # [B, num_heads, query_len, memory_len]
    attention_scores = basis_function(scaled_distances)  # [B, num_heads, query_len, memory_len]

    # Set attention scores to 0. for masked elements
    attention_scores = attention_scores.masked_fill(masked_elements[:, None, None, :], 0.0)

    if normalise_kernel_values:
        attention_scores = attention_scores / (
            torch.abs(attention_scores).sum(dim=-1, keepdim=True) + 1e-5
        )

    return attention_scores  # [B, num_heads, query_len, memory_len]


def attend(
    attention_scores: Tensor,  # [B, num_heads, query_len, memory_len]
    values: Tensor,  # [B, num_heads, memory_len, value_dim]
) -> Tensor:  # [B, num_heads, query_len, value_dim]
    """Compute attended `values` using `attention_scores`.

    Args:
        attention_scores: A tensor of shape `[batch_len, num_heads, query_len, memory_len]`
            e.g. as computed from :func:`compute_attention_scores`.
        values: A tensor of shape `[batch_len, num_heads, memory_len, value_dim]`.

    Returns:
        attended_values: A tensor of shape `[batch_len, num_heads, query_len, query_len, value_dim]`.
    """
    # Attend to values using the attention scores
    return attention_scores @ values


def flatten_multihead(
    multiheaded_weighted_value_sum: Tensor,  # [B, num_heads, query_len, value_dim]
) -> Tensor:  # [B, query_len, num_heads * value_dim]
    multiheaded_weighted_value_sum = multiheaded_weighted_value_sum.transpose(
        -2, -3
    )  # [B, query_len, num_heads, value_dim]
    attention_pre_project = multiheaded_weighted_value_sum.reshape(
        (
            multiheaded_weighted_value_sum.shape[0],
            multiheaded_weighted_value_sum.shape[1],
            -1,
        )
    )  # [B, query_len, num_heads * value_dim]

    return attention_pre_project


class KernelAttention(nn.Module):
    def __init__(
        self,
        *,
        value_dim: int,
        output_dim: int,
        lengthscales: Sequence[float],
        normalise_kernel_values: bool,
    ):
        super().__init__()
        self.register_buffer(
            "lengthscales", torch.tensor(lengthscales, dtype=torch.float32), persistent=True
        )
        num_heads = len(
            lengthscales
        )  # The number of heads of this transformer is the number of lengthscales
        self.basis_function = gaussian_basis_function
        self.normalise_kernel_values = normalise_kernel_values

        self._out_projection = nn.Linear(value_dim * num_heads, output_dim, bias=False)

    def compute_lengthscales(self) -> Tensor:
        # No computation here. Just return it.
        assert isinstance(self.lengthscales, Tensor)
        return self.lengthscales

    def forward(
        self,
        query_positions: Tensor,  # [B, query_len, pos_dim]
        key_positions: Tensor,  # [B, memory_len, pos_dim]
        values: Tensor,  # [B, memory_len, num_heads, value_dim]
        masked_elements: torch.BoolTensor,  # [B, memory_len]
        cache: Optional[Cache] = None,
    ) -> Tensor:
        if cache is None:
            # Create an empty cache with nothing marked as cacheable for simplicity.
            cache = Cache()

        attention_scores = cache.load_or_produce(
            NullaryClosure.create(
                compute_kernel_attention_scores,
                query=query_positions,
                key=key_positions,
                masked_elements=masked_elements,
                lengthscales=self.compute_lengthscales(),
                basis_function=self.basis_function,
            ),
        )
        # Transpose because `attend` expects shape `[B, num_heads, memory_len, value_dim]`.
        values_transposed = values.transpose(1, 2)
        # Attend to the values using the attention scores.
        attended = attend(
            attention_scores, values_transposed
        )  # [B, num_heads, query_len, value_dim]
        attended_flattened = flatten_multihead(attended)  # [B, query_len, num_heads * value_dim]
        return self._out_projection(attended_flattened)


class LearnableLengthscaleKernelAttention(KernelAttention):
    def __init__(
        self,
        *,
        value_dim: int,
        output_dim: int,
        lengthscales: Sequence[float],
        normalise_kernel_values: bool,
    ):
        """
        Args:
            lengthscales: the initial values for the lengthscale parameters
            normalise_kernel_values: Whether the attention weights are normalised to sum up
                to 1
        """

        super().__init__(
            value_dim=value_dim,
            output_dim=output_dim,
            lengthscales=lengthscales,
            normalise_kernel_values=normalise_kernel_values,
        )
        self.log_lengthscales = nn.Parameter(
            torch.log(torch.tensor(lengthscales, dtype=torch.float32)), requires_grad=True
        )

        self.basis_function = gaussian_basis_function
        self.normalise_kernel_values = normalise_kernel_values

        num_heads = len(
            lengthscales
        )  # The number of heads of this transformer is the number of lengthscales
        self._out_projection = nn.Linear(value_dim * num_heads, output_dim, bias=False)

    def compute_lengthscales(self) -> Tensor:
        return torch.exp(self.log_lengthscales)


class LearnableChebyshevKernelAttention(KernelAttention):
    """Learnable attention using Chebyshev rational expansions."""

    def __init__(
        self,
        *,
        value_dim: int,
        output_dim: int,
        lengthscales: Sequence[float],
        cheb_order: int,
        normalise_kernel_values: bool,
        force_asymptotic_zero: bool,
    ):
        """
        Args:
            lengthscales: the initial values for the lengthscale parameters
            cheb_order: the truncation of the Chebyshev rational expansion.
            normalise_kernel_values: Whether the attention weights are normalised to sum up
                to 1
            force_asymptotic_zero: if True, we linearly constrain coefficients such
                that F(x) = 0 for x -> oo.  If False, we do not constrain coefficients.
        """
        assert cheb_order >= 1

        super().__init__(
            value_dim=value_dim,
            output_dim=output_dim,
            lengthscales=lengthscales,
            normalise_kernel_values=normalise_kernel_values,
        )

        num_heads = len(lengthscales)

        # Create cheb_coeffs parameter and initialize to exp(-s) function
        # as in gaussian_basis_function.
        # The coefficients are generated by numerical quadrature to float32
        # numerical precision using `projects/timewarp/notebooks/chebrat.jl`.
        cheb_coeffs_expmx = [
            4.275836e-01,
            -5.464240e-01,
            7.106222e-02,
            5.473271e-02,
            5.744192e-03,
            -7.926410e-03,
            -5.392865e-03,
            -1.210823e-03,
            6.996851e-04,
            8.686655e-04,
            4.459163e-04,
            7.084817e-05,
            -9.620444e-05,
            -1.110469e-04,
            -6.551055e-05,
            -1.875292e-05,
            7.930955e-06,
            1.553729e-05,
            1.246072e-05,
            6.282442e-06,
            1.216243e-06,
            -1.468327e-06,
            -2.141963e-06,
            -1.694741e-06,
            -9.063254e-07,
            -2.337215e-07,
            1.609271e-07,
            2.978384e-07,
            2.700519e-07,
            1.730454e-07,
            7.272222e-08,
            1.192814e-09,
        ]  # type: ignore
        coeffs_take = min(len(cheb_coeffs_expmx), cheb_order)  # type: ignore
        coeffs_pad_zero = max(0, cheb_order - len(cheb_coeffs_expmx))
        cheb_coeffs = cheb_coeffs_expmx[0:coeffs_take] + coeffs_pad_zero * [0.0]
        cheb_coeffs = torch.tensor(cheb_coeffs, dtype=torch.float32)  # type: ignore
        cheb_coeffs = torch.unsqueeze(cheb_coeffs, 0).expand(num_heads, -1)  # type: ignore
        self.cheb_coeffs = nn.Parameter(cheb_coeffs, requires_grad=True)  # type: ignore

        # The heavy lifting is done within the basis function
        self.basis_function = lambda sd: chebyshev_basis_function(
            sd, cheb_order, self.cheb_coeffs, force_asymptotic_zero
        )
        self.normalise_kernel_values = normalise_kernel_values

        self._out_projection = nn.Linear(value_dim * num_heads, output_dim, bias=False)
