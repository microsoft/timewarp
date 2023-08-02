import torch


class Moments:
    """Online estimator for the mean and variance, i.e. the 0th moment and the 1st
    central moment.

    The algorithm here is a batched version of the one described at [1].

    [1] https://www.johndcook.com/blog/standard_deviation/
    """

    def __init__(self):
        self.n = 0
        self.m = None
        self.s = None

    def fit(self, x: torch.Tensor):
        # Ensure that there is a batch dimension in the front
        x = torch.atleast_2d(x)

        k = len(x)
        assert k >= 1
        if self.m is None:
            self.n = k
            self.m = x.mean(dim=0)
            if k == 1:
                self.s = torch.zeros_like(self.m)
            else:
                self.s = self.n * x.var(dim=0, correction=0)
        else:
            self.n += k

            # Compute all intermediate means that would occur if the values were
            # observed sequentially
            nmk_to_n = torch.arange(self.n - k + 1, self.n + 1, device=x.device)
            m_k = torch.einsum("i, ... -> i...", (self.n - k) / nmk_to_n, self.m) + torch.einsum(
                "i, i... -> i...", nmk_to_n.reciprocal(), x.cumsum(dim=0)
            )

            # Update the variance estimate
            s_first = (x[0] - self.m) * (x[0] - m_k[0])
            self.s = self.s + s_first
            if k > 1:
                self.s = self.s + ((x[1:] - m_k[:-1]) * (x - m_k)[1:]).sum(dim=0)

            # Clone the last mean to release the larger tensor of intermediate values
            self.m = m_k[-1].clone()

    def mean(self) -> torch.Tensor:
        if self.n == 0:
            raise RuntimeError("Need at least 1 observation to compute the sample mean.")
        return self.m

    def std(self) -> torch.Tensor:
        return self.var(correction=0).sqrt()

    def var(self, correction: int = 0) -> torch.Tensor:
        if self.n <= correction:
            raise RuntimeError(
                f"Need at least {correction + 1} observations to compute the sample variance."
            )
        return self.s / (self.n - correction)
