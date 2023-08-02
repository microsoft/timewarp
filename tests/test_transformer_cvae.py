import pytest

import torch
import numpy as np
import scipy.stats as stats

from timewarp.dataloader import moldyn_dense_collate_fn
from timewarp.utils.loss_utils import get_log_likelihood
from timewarp.utils.sampling_utils import get_sample
from utilities.training_utils import (
    set_seed,
)
from timewarp.utils.training_utils import (
    load_or_construct_optimizer_lr_scheduler,
)

from timewarp.tests.assets import get_model_config


def test_cvae_sampling(
    dummy_datapoints,
    device: torch.device,
):
    """Test sampling from the conditional VAE model."""
    set_seed(0)
    model, config = get_model_config("transformer_cvae")
    model = model.to(device)
    batch = moldyn_dense_collate_fn(dummy_datapoints)
    y_coords, y_velocs = get_sample(model, batch=batch, num_samples=1, device=device)
    assert torch.all(torch.isfinite(y_coords))
    assert torch.all(torch.isfinite(y_velocs))


def test_cvae_elbo(
    dummy_datapoints,
    device: torch.device,
):
    """Test ELBO evaluation through the model."""
    set_seed(0)
    model, config = get_model_config("transformer_cvae")
    model = model.to(device)
    batch = moldyn_dense_collate_fn(dummy_datapoints)
    with torch.no_grad():
        log_likelihood = get_log_likelihood(model, batch=batch, device=device)  # [B]
        assert torch.all(torch.isfinite(log_likelihood))


def take_log_likelihoods(model, batch, device, num_samples):
    """Take multiple samples of log-likelihoods."""
    with torch.no_grad():
        lls = [get_log_likelihood(model, batch=batch, device=device) for si in range(num_samples)]
        lls = torch.stack(lls, dim=0)  # [S, B]

    lls = lls.clone().detach().cpu().numpy()

    return lls


def test_cvae_elbo_iwae_consistency(
    dummy_datapoints,
    device: torch.device,
):
    """Test that E[IWAE_1] = E[ELBO]."""
    set_seed(0)
    model, config = get_model_config("transformer_cvae")
    model = model.to(device)
    batch = moldyn_dense_collate_fn(dummy_datapoints)

    num_samples = 200

    model.elbo_estimator = "elbo"
    model.num_elbo_samples = 1
    lls_elbo = take_log_likelihoods(model, batch, device, num_samples)

    model.elbo_estimator = "iwae"
    lls_iwae = take_log_likelihoods(model, batch, device, num_samples)

    # Test for the same mean of ELBO and IWAE_1
    tstat, pvalue = stats.ttest_ind(
        lls_elbo,
        lls_iwae,
        axis=0,
        equal_var=True,  # IWAE_1 is ELBO, so same variance
    )
    assert np.all(pvalue > 1.0e-4)


def test_cvae_iwae_monotonicity(
    dummy_datapoints,
    device: torch.device,
):
    """Test that E[IWAE_m] >= E[IWAE_n] for m > n."""
    set_seed(0)
    model, config = get_model_config("transformer_cvae")
    model = model.to(device)
    batch = moldyn_dense_collate_fn(dummy_datapoints)

    num_samples = 200

    model.elbo_estimator = "iwae"
    test_num_elbo_samples = [1, 2, 4, 8, 16]
    lls = []
    for num_elbo_samples in test_num_elbo_samples:
        model.num_elbo_samples = num_elbo_samples
        lls_iwae = take_log_likelihoods(model, batch, device, num_samples)
        lls_iwae = np.mean(lls_iwae, axis=0)  # [B]
        lls.append(lls_iwae)

    lls = np.stack(lls, axis=0)  # [len(test_num_elbo_samples), B]
    lls_var = np.var(lls, axis=1)  # [len(test_num_elbo_samples)]
    lls_mean = np.mean(lls, axis=1)  # [len(test_num_elbo_samples)]

    for si, num_elbo_samples in enumerate(test_num_elbo_samples):
        ll_var_cur = lls_var[si]
        assert ll_var_cur > 0.0, "V[IWAE_%d] = 0, which should not happen." % num_elbo_samples

    for si in range(1, len(test_num_elbo_samples)):
        num_elbo_samples_cur = test_num_elbo_samples[si]
        num_elbo_samples_prev = test_num_elbo_samples[si - 1]
        assert (
            lls_mean[si] >= lls_mean[si - 1]
        ), "(IWAE_%d) %.5f < %.5f (IWAE_%d), which should not happen" % (
            num_elbo_samples_cur,
            lls_mean[si],
            num_elbo_samples_prev,
            lls_mean[si - 1],
        )


def estimate_gradient_variance(optimizer, model, batch, device, num_samples):
    """Estimate the gradient variance."""
    grad = dict()
    for name, param in model.named_parameters():
        grad[name] = []

    # Take multiple samples of the gradient
    lls = []
    for si in range(num_samples):
        optimizer.zero_grad()
        ll = get_log_likelihood(model, batch=batch, device=device)
        ll = ll.mean()  # [1]
        lls.append(float(ll))
        ll.backward()
        for name, param in model.named_parameters():
            grad[name].append(param.grad.clone().detach())

    lls = np.array(lls)
    lls_var = np.var(lls)

    # Estimate the variance
    grad_vars = dict()
    for name, param in model.named_parameters():
        param_grads = torch.stack(grad[name], dim=0)
        grad_var = torch.var(param_grads, dim=0, unbiased=True)
        grad_vars[name] = grad_var.clone().detach().cpu().numpy()

    return lls_var, grad_vars


def get_cvae_ll_stddev(
    elbo_estimator,
    num_elbo_samples,
    dummy_datapoints,
    device: torch.device,
):
    set_seed(0)
    model, config = get_model_config(
        "transformer_cvae",
        learning_rate=1.0e-4,
        warmup_steps=1000,
        weight_decay=0.0,
    )
    model = model.to(device)
    batch = moldyn_dense_collate_fn(dummy_datapoints)

    model.elbo_estimator = elbo_estimator
    model.num_elbo_samples = num_elbo_samples

    num_samples = 100

    lls = []
    for si in range(num_samples):
        ll = get_log_likelihood(model, batch=batch, device=device)
        ll = ll.mean()  # [1]
        lls.append(float(ll))

    lls = np.array(lls)  # type: ignore
    ll_stddev = np.std(lls)

    return ll_stddev


def test_cvae_elbo_ll_stddev(
    dummy_datapoints,
    device: torch.device,
):
    """Test stddev(ELBO) > 0."""
    ll_stddev = get_cvae_ll_stddev("elbo", 1, dummy_datapoints, device)
    assert ll_stddev > 0.0, "Deterministic CVAE ELBO values, should not happen."


def test_cvae_iwae_ll_stddev(
    dummy_datapoints,
    device: torch.device,
):
    """Test stddev(IWAE_m) > 0 for a few m values."""
    ll_stddev_prev = 0.0
    num_elbo_samples_prev = 0
    for num_elbo_samples in [1, 2, 4, 8]:
        ll_stddev = get_cvae_ll_stddev("iwae", num_elbo_samples, dummy_datapoints, device)
        assert ll_stddev > 0.0, (
            "Deterministic CVAE IWAE_%d values, should not happen." % num_elbo_samples
        )
        assert (
            ll_stddev > ll_stddev_prev
        ), "stddev(IWAE_%d) %.5f < %.5f stddev(IWAE_%d), which should not happen." % (
            num_elbo_samples,
            ll_stddev,
            ll_stddev_prev,
            num_elbo_samples_prev,
        )


def get_cvae_gradients(elbo_estimator, num_elbo_samples, dummy_datapoints, device):
    set_seed(0)
    model, config = get_model_config(
        "transformer_cvae",
        learning_rate=1.0e-4,
        warmup_steps=1000,
        weight_decay=0.0,
    )
    model = model.to(device)
    assert model.training, "Model not in training mode."

    optimizer, _ = load_or_construct_optimizer_lr_scheduler(model, config)
    batch = moldyn_dense_collate_fn(dummy_datapoints)

    model.elbo_estimator = elbo_estimator
    model.num_elbo_samples = num_elbo_samples

    grad = dict()
    optimizer.zero_grad()
    model.zero_grad()
    ll = get_log_likelihood(model, batch=batch, device=device)
    ll = ll.mean()  # [1]
    ll.backward()
    for name, param in model.named_parameters():
        grad[name] = param.grad.clone().detach().cpu().numpy()

    return grad


def test_cvae_elbo_gradient(
    dummy_datapoints,
    device: torch.device,
):
    """Test |grad ELBO| > 0."""
    grads = get_cvae_gradients("elbo", 1, dummy_datapoints, device)
    for name, grad in grads.items():
        assert np.linalg.norm(grad) > 0, (
            "grad of '%s' in ELBO is exactly zero, which should not happen" % name
        )


def test_cvae_iwae_gradient(
    dummy_datapoints,
    device: torch.device,
):
    """Test |grad IWAE_m| > 0 for multiple m."""
    for num_elbo_samples in [1, 2, 4, 8]:
        grads = get_cvae_gradients("iwae", num_elbo_samples, dummy_datapoints, device)
        for name, grad in grads.items():
            assert (
                np.linalg.norm(grad) > 0
            ), "grad of '%s' in IWAE_%d is exactly zero, which should not happen" % (
                name,
                num_elbo_samples,
            )


@pytest.mark.skip(reason="Test should compare gradient variance between IWAE and IWAE-DReG.")
def test_cvae_iwae_gradient_variance(
    dummy_datapoints,
    device: torch.device,
):
    """Test that V[grad IWAE_m] >= V[grad IWAE_n] for m > n."""
    set_seed(0)
    model, config = get_model_config(
        "transformer_cvae",
        learning_rate=1.0e-4,
        warmup_steps=1000,
        weight_decay=0.0,
    )
    model = model.to(device)
    assert model.training, "Model not in training mode."

    optimizer, _ = load_or_construct_optimizer_lr_scheduler(model, config)
    batch = moldyn_dense_collate_fn(dummy_datapoints)

    num_samples = 200

    model.elbo_estimator = "iwae"
    test_num_elbo_samples = [4, 8, 16]
    lls_grad_vars = []
    for num_elbo_samples in test_num_elbo_samples:
        model.num_elbo_samples = num_elbo_samples
        lls_var, lls_grad_var = estimate_gradient_variance(
            optimizer, model, batch, device, num_samples
        )
        lls_grad_vars.append(lls_grad_var)
        assert lls_var > 0.0, "V[IWAE_%d] = 0, which should not happen." % num_elbo_samples

    for si, num_elbo_samples in enumerate(test_num_elbo_samples):
        for name, var in lls_grad_vars[si].items():
            assert np.linalg.norm(var) > 0.0, (
                "Gradient variance of IWAE_%d in variable '%s' is "
                "exactly zero, which should not happen."
                % (
                    num_elbo_samples,
                    name,
                )
            )

    variance_higher_fraction_thresh = 0.9
    for si in range(1, len(test_num_elbo_samples)):
        num_elbo_samples_cur = test_num_elbo_samples[si]
        num_elbo_samples_prev = test_num_elbo_samples[si - 1]

        # Test that gradient variance increases along IWAE_m for
        # increasing m
        vars_cur = lls_grad_vars[si]
        vars_prev = lls_grad_vars[si - 1]
        for (name_prev, var_prev), (name_cur, var_cur) in zip(
            vars_prev.items(),
            vars_cur.items(),
        ):
            assert name_prev == name_cur  # ensure we compare the same variables
            if np.size(var_prev) <= 100:
                continue  # Too few elements

            var_cur_higher_than_prev = var_cur >= var_prev
            fraction_higher = np.count_nonzero(var_cur_higher_than_prev) / np.size(
                var_cur_higher_than_prev
            )
            assert fraction_higher > variance_higher_fraction_thresh, (
                "Gradient variance of IWAE_%d compared to IWAE_%d "
                "is higher in only %.1f%% of elements in variable '%s', "
                "which should not happen; we expect to see values "
                ">%.1f%%."
                % (
                    num_elbo_samples_cur,
                    num_elbo_samples_prev,
                    100.0 * fraction_higher,
                    name_cur,
                    100.0 * variance_higher_fraction_thresh,
                )
            )
