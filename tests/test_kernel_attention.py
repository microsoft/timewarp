import torch

from timewarp.modules.layers.kernel_attention import (
    KernelAttention,
    LearnableChebyshevKernelAttention,
    chebyshev_expansion,
    chebyshev_basis_function,
    compute_kernel_attention_scores,
)
from timewarp.modules.layers.kernel_self_attention import KernelSelfAttention
from timewarp.modules.layers.custom_attention_encoder import CustomTransformerEncoderLayer
from timewarp.modules.layers.custom_transformer_block import CustomAttentionTransformerBlock
from utilities.cache import Cache
from utilities.common import Returns

from utilities.training_utils import set_seed


def test_kernel_attention_normalised():
    set_seed(0)
    B = 2
    query_len = 11
    memory_len = 13
    pos_dim = 3
    d_model = 7
    lengthscales = [0.1, 0.2, 0.5, 0.7, 1.0]
    attention = KernelAttention(
        value_dim=d_model,
        output_dim=d_model,
        lengthscales=lengthscales,
        normalise_kernel_values=True,
    )

    with torch.no_grad():
        attention = compute_kernel_attention_scores(
            query=0.1 * torch.randn(B, query_len, pos_dim),
            key=0.1 * torch.randn(B, memory_len, pos_dim),
            masked_elements=torch.full((B, memory_len), False),  # type: ignore
            lengthscales=attention.compute_lengthscales(),
            basis_function=attention.basis_function,
        )

    assert attention.shape == (B, len(lengthscales), query_len, memory_len)
    assert torch.allclose(
        attention.sum(-1), torch.full((B, len(lengthscales), query_len), 1.0), atol=1e-3
    )


def test_attention_score_caching():
    set_seed(0)
    B = 2
    query_len = 11
    memory_len = 13
    pos_dim = 3
    d_model = 7
    value_dim = d_model
    lengthscales = [0.1, 0.2, 0.5, 0.7, 1.0]
    num_heads = len(lengthscales)
    attention = KernelAttention(
        value_dim=value_dim,
        output_dim=d_model,
        lengthscales=lengthscales,
        normalise_kernel_values=True,
    )

    query_positions = 0.1 * torch.randn(B, query_len, pos_dim)
    key_positions = 0.1 * torch.randn(B, memory_len, pos_dim)
    masked_elements = torch.full((B, memory_len), False)
    values = torch.randn(B, memory_len, num_heads, value_dim)

    # Check that passing cached `attention_scores` results in the same computation.
    cache = Cache(
        cacheable={compute_kernel_attention_scores},
    )
    out = attention(query_positions, key_positions, values, masked_elements, cache=cache)
    assert len(cache._lookup) == 1

    cache_info = cache.cache_info(compute_kernel_attention_scores)
    assert cache_info.hits == 0

    # Changed input should result in different output.
    out_cached_attention = attention(
        query_positions + torch.rand_like(query_positions),
        key_positions,
        values,
        masked_elements,
        cache=cache,
    )
    assert (out != out_cached_attention).all()

    cache_info = cache.cache_info(compute_kernel_attention_scores)
    assert cache_info.hits == 0
    assert cache_info.misses == 2

    # If `attention_scores` is not marked as cacheable, it should not be in the `cache`.
    cache = Cache()
    out = attention(query_positions, key_positions, values, masked_elements, cache=cache)
    cache_info = cache.cache_info(compute_kernel_attention_scores)
    assert cache_info.hits == 0
    assert cache_info.misses == 0

    # Wrapping `attention` in `KernelSelfAttention` should result in the same `attention_scores`.
    positions = key_positions
    self_attention = KernelSelfAttention(
        input_dim=value_dim, num_heads=num_heads, value_dim=value_dim, attention=attention
    )

    # Wrapped in `CustomAttentionTransformerBlock`.
    # TODO : Maybe move these tests to a separate file for transformer blocks?
    latent_mlp_hidden_dims = [8]
    dim_feedforward = 8
    transformer_encoder_layers = [
        CustomTransformerEncoderLayer(
            d_model=d_model,
            self_attention=self_attention,
            dim_feedforward=dim_feedforward,
            dropout=0,  # so we can compare using equality
        ),
        CustomTransformerEncoderLayer(
            d_model=d_model,
            self_attention=self_attention,
            dim_feedforward=dim_feedforward,
            dropout=0,  # so we can compare using equality
        ),
    ]
    # With and without `use_shared_attention`.
    # should produce the same results.
    # For simplicity, use `value_dim` as both input- and output-dimensionality.
    transformer_block = CustomAttentionTransformerBlock(
        value_dim,
        value_dim,
        latent_mlp_hidden_dims,
        transformer_encoder_layers,
    )

    # Just use one of the output heads from `values` as an example sequence.
    input_seq = values[:, :, 1, :]
    # Different layers will be using different _instances_ of `lengthscales`
    # but the same values, so we're just going to ignore this by the
    # `lengthscales` argument to `0` upon hashing.
    cache = Cache(
        cacheable={compute_kernel_attention_scores},
        keyword_transforms={compute_kernel_attention_scores: {"lengthscales": Returns(0)}},
    )
    with torch.no_grad():
        out = transformer_block(input_seq, positions, masked_elements)
        out_with_shared = transformer_block(input_seq, positions, masked_elements, cache=cache)
        assert (out == out_with_shared).all()

        # 1st `CustomTransformerEncoderLayer` misses, and then the 2nd hits.
        cache_info = cache.cache_info(compute_kernel_attention_scores)
        assert cache_info.hits == len(transformer_encoder_layers) - 1

        # Perturbing inputs still hits cache, but overall should result in different output.
        out_with_shared_incorrect = transformer_block(
            input_seq + torch.randn_like(input_seq), positions, masked_elements, cache=cache
        )
        cache_info = cache.cache_info(compute_kernel_attention_scores)
        assert cache_info.hits == 2 * len(transformer_encoder_layers) - 1
        assert not torch.allclose(out, out_with_shared_incorrect)


def test_chebyshev_expansion():
    scaled_distances = 0.7 * torch.ones((1, 1, 1, 1))
    cheb = chebyshev_expansion(scaled_distances, 5)
    cheb = cheb.flatten()

    # Reference values from Julia implementation
    cheb_ref = torch.Tensor(
        [1.0, -0.17647058823529416, -0.9377162629757785, 0.507429269285569, 0.7586235796985188]
    )

    assert torch.allclose(cheb, cheb_ref)


def test_chebyshev_basis_function():
    # First six Chebyshev rational basis coefficients for exp(-s)
    cheb_coeffs = torch.Tensor(
        [
            0.42758357,
            -0.54642403,
            0.07106222,
            0.05473271,
            0.00574419,
            -0.00792641,
        ]
    )
    cheb_order = len(cheb_coeffs)
    num_heads = 3
    cheb_coeffs = cheb_coeffs.flatten()
    cheb_coeffs = torch.unsqueeze(cheb_coeffs, 0)
    cheb_coeffs = cheb_coeffs.expand(num_heads, -1)

    # Create random mock data
    set_seed(0)
    scaled_distances = 10.0 * torch.rand((2, num_heads, 64, 128))

    approx_expsq = chebyshev_basis_function(scaled_distances, cheb_order, cheb_coeffs, False)
    exact_expsq = torch.exp(-(scaled_distances**2.0))

    # Maximum error over [0, 10] is 0.0076
    assert torch.allclose(exact_expsq, approx_expsq, atol=1.0e-2, rtol=0.0)

    # Test with asymptotics
    scaled_distances = 1000.0 * torch.ones((1, 1, 1, 1))
    asymptotic = chebyshev_basis_function(scaled_distances, cheb_order, cheb_coeffs, True)
    asymptotic = asymptotic.flatten()
    assert torch.allclose(torch.zeros_like(asymptotic), asymptotic, atol=1.0e-6, rtol=0.0)


def test_chebyshev_kernel_attention_normalised():
    set_seed(0)
    B = 2
    query_len = 11
    memory_len = 13
    pos_dim = 3
    d_model = 7
    lengthscales = [0.1, 0.2, 0.5, 0.7, 1.0]
    attention = LearnableChebyshevKernelAttention(
        value_dim=d_model,
        output_dim=d_model,
        lengthscales=lengthscales,
        cheb_order=6,
        normalise_kernel_values=True,
        force_asymptotic_zero=True,
    )

    with torch.no_grad():
        attention = compute_kernel_attention_scores(
            query=0.1 * torch.randn(B, query_len, pos_dim),
            key=0.1 * torch.randn(B, memory_len, pos_dim),
            masked_elements=torch.full((B, memory_len), False),
            lengthscales=attention.compute_lengthscales(),
            basis_function=attention.basis_function,
        )

    assert attention.shape == (B, len(lengthscales), query_len, memory_len)
    assert torch.allclose(
        torch.abs(attention).sum(-1), torch.full((B, len(lengthscales), query_len), 1.0), atol=1e-3
    )
