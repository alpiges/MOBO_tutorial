"""ParEGO acquisition function.

ParEGO (Knowles, 2005) converts the multi-objective problem into a
single-objective one by applying a random Chebyshev scalarization of the
GP posterior means, then computes standard analytic Expected Improvement
on that scalar objective.

Reference
---------
Knowles, J. (2005). ParEGO: a hybrid algorithm with on-line landscape
approximation for expensive multiobjective optimization problems.
IEEE Transactions on Evolutionary Computation, 9(4), 353-375.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from surrogate import _ExactGPModel, predict


def _chebyshev_scalarize(mu: Tensor, sigma: Tensor, weights: Tensor, rho: float = 0.05) -> tuple[Tensor, Tensor]:
    """Augmented Chebyshev scalarization of GP moments.

    For a weight vector w and **maximisation** objectives, the scalar objective is

        s(f) = min_m w_m * f_m + rho * sum_m w_m * f_m

    (Standard minimisation ParEGO would use max_m; flipping to min_m is the
    correct adaptation when all objectives are being maximised.)

    Parameters
    ----------
    mu:     (..., M) predictive means.
    sigma:  (..., M) predictive standard deviations.
    weights: (M,) non-negative weight vector (will be normalised).
    rho:    Augmentation coefficient.

    Returns
    -------
    s_mu, s_sigma: scalar mean and std at each point, shape (...,).
    """
    w = weights / weights.sum()
    # Weighted objectives for the min term and the augmentation term.
    # NOTE: the standard Knowles (2005) formulation uses max_m for *minimisation*.
    # Here objectives are *maximised*, so the bottleneck is the smallest weighted
    # objective → we use min_m instead of max_m.
    weighted = w * mu                            # (..., M)
    min_term = weighted.min(dim=-1).values       # (...,)  ← min, not max
    aug_term = rho * weighted.sum(dim=-1)        # (...,)
    s_mu = min_term + aug_term

    # Variance: approximate as (w_max^2 + rho^2 * ||w||^2) * sigma^2 for
    # simplicity (worst-case single active objective).
    w_max = w.max()
    var_coeff = w_max ** 2 + rho ** 2 * (w ** 2).sum()
    s_sigma = torch.sqrt(var_coeff * (w * sigma).pow(2).sum(dim=-1))

    return s_mu, s_sigma


def _analytic_ei(mu: Tensor, sigma: Tensor, best: float) -> Tensor:
    """Standard analytic Expected Improvement (log-stable version).

    Parameters
    ----------
    mu, sigma: scalar predictive moments, shape (...,).
    best:      current best scalar incumbent.

    Returns
    -------
    EI values, shape (...,).
    """
    from torch.distributions import Normal
    normal = Normal(0, 1)
    z = (mu - best) / sigma.clamp(min=1e-9)
    ei = sigma * (normal.log_prob(z).exp() + z * normal.cdf(z))
    return ei.clamp(min=0.0)


class ParEGO:
    """ParEGO acquisition: random Chebyshev scalarization + analytic EI.

    Parameters
    ----------
    models:
        List of M trained GP models (one per objective, see surrogate.py).
    train_y:
        (n, M) observed objectives used to compute the incumbent.
    rho:
        Augmented Chebyshev coefficient (default 0.05 as in the paper).
    """

    name = "ParEGO"

    def __init__(
        self,
        models: list[_ExactGPModel],
        train_y: Tensor,
        rho: float = 0.05,
    ) -> None:
        self.models = models
        self.train_y = train_y
        self.rho = rho
        M = train_y.shape[-1]
        # Sample a random weight vector from the simplex
        raw = torch.rand(M)
        self.weights = raw / raw.sum()

    def __call__(self, x: Tensor) -> Tensor:
        """Evaluate ParEGO acquisition at candidate points x.

        Parameters
        ----------
        x:
            (n, d) tensor of candidates in the normalised search space.

        Returns
        -------
        (n,) acquisition values (higher is better).
        """
        mu, sigma = predict(self.models, x)
        s_mu, s_sigma = _chebyshev_scalarize(mu, sigma, self.weights, self.rho)

        # Incumbent: best scalarised observed value
        s_mu_obs, _ = _chebyshev_scalarize(
            self.train_y,
            torch.zeros_like(self.train_y),
            self.weights,
            self.rho,
        )
        best = s_mu_obs.max().item()

        return _analytic_ei(s_mu, s_sigma, best)

    def _train_x_cache(self) -> Tensor:
        # Training inputs are stored inside the first GP model
        return self.models[0].train_inputs[0]
