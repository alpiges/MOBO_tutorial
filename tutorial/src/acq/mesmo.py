"""Max-value Entropy Search for Multi-objective BO (MESMO).

MESMO (Belakaria et al., 2019) estimates the information gain about the
Pareto front by sampling a candidate Pareto set from the posterior and
computing the reduction in differential entropy for each objective GP.

This is a pedagogical implementation that is intentionally readable rather
than maximally efficient.  The key approximation steps are:

1. Sample S candidate Pareto sets by drawing objective vectors at a
   discrete grid of points and keeping non-dominated samples.
2. For each candidate x, compute the approximate information gain as
      alpha(x) ≈ sum_m [ H[f_m(x)] - E_{Y*}[ H[f_m(x) | Y*_m] ] ]
   where the conditional entropy reduction is approximated via the
   Gaussian approximation of Hernandez-Lobato et al. (2016).

Reference
---------
Belakaria, S., Deshwal, A., & Doppa, J. R. (2019). Max-value entropy
search for multi-objective Bayesian optimization. NeurIPS 2019.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from hypervolume import _nondominated
from surrogate import _ExactGPModel, predict


def _sample_pareto_set(
    models: list[_ExactGPModel],
    bounds: Tensor,
    n_grid: int = 256,
) -> Tensor:
    """Draw one sample Pareto set from the posterior.

    We evaluate a random grid, draw one posterior sample per GP, and keep
    the non-dominated points.

    Returns
    -------
    (p, M) tensor of non-dominated sampled objective vectors.
    """
    d = bounds.shape[1]
    M = len(models)
    # Sobol-like quasi-random grid
    grid = torch.rand(n_grid, d) * (bounds[1] - bounds[0]) + bounds[0]
    samples = []
    for m, model in enumerate(models):
        with torch.no_grad():
            dist = model.likelihood(model(grid))
            s = dist.sample()           # (n_grid,)
        samples.append(s)
    y_samples = torch.stack(samples, dim=-1)   # (n_grid, M)
    return _nondominated(y_samples)[0]


def _entropy_reduction(
    mu_m: float,
    sigma_m: float,
    pareto_m_vals: Tensor,
    n_grid: int = 64,
) -> float:
    """Approximate entropy reduction for one objective dimension.

    Uses the truncated-Gaussian approximation from MES (Wang & Jegelka, 2017)
    extended to use the empirical maximum of the sampled Pareto set as y*.

    Parameters
    ----------
    mu_m, sigma_m: predictive mean and std for objective m at query point.
    pareto_m_vals: (p,) sampled values of objective m in the Pareto set.

    Returns
    -------
    Scalar approximate information gain (initial entropy - conditional entropy).
    """
    if sigma_m < 1e-10:
        return 0.0
    
    # Initial entropy: H[f_m(x)] for Gaussian is 0.5 * log(2πe * σ²)
    h_initial = 0.5 * math.log(2 * math.pi * math.e * sigma_m**2)
    
    y_star = pareto_m_vals.max().item()
    # Clip y_star so that gamma stays in a numerically active range.
    # With few training points the GP posterior is very uncertain and
    # posterior samples can produce y_star many σ above mu_m, driving
    # γ → ∞ and entropy-reduction → 0 everywhere (flat landscape).
    y_star = min(y_star, mu_m + 4.0 * sigma_m)
    gamma = (y_star - mu_m) / sigma_m
    from torch.distributions import Normal
    normal = Normal(0.0, 1.0)
    phi_gamma = math.exp(normal.log_prob(torch.tensor(gamma)).item())
    Phi_gamma = normal.cdf(torch.tensor(gamma)).item()
    if Phi_gamma < 1e-10:
        return h_initial
    
    # Conditional entropy reduction
    h_reduction = (
        gamma * phi_gamma / (2.0 * Phi_gamma)
        - math.log(Phi_gamma)
    )
    
    return h_initial - h_reduction


class MESMO:
    """MESMO acquisition function.

    Parameters
    ----------
    models:
        List of M trained GP models (one per objective).
    bounds:
        (2, d) search space bounds (used to sample candidate Pareto sets).
    n_pareto_samples:
        Number of Pareto set samples to average over (default 10).
    n_grid:
        Grid size used when sampling each Pareto set (default 256).
    """

    name = "MESMO"

    def __init__(
        self,
        models: list[_ExactGPModel],
        bounds: Tensor,
        n_pareto_samples: int = 10,
        n_grid: int = 256,
    ) -> None:
        self.models = models
        self.bounds = bounds
        self.n_pareto_samples = n_pareto_samples
        self.n_grid = n_grid
        # Pre-sample Pareto sets once per acquisition instantiation
        self._pareto_samples = [
            _sample_pareto_set(models, bounds, n_grid)
            for _ in range(n_pareto_samples)
        ]

    def __call__(self, x: Tensor) -> Tensor:
        """Evaluate MESMO acquisition at candidate points x.

        Parameters
        ----------
        x:
            (n, d) candidates in the search space.

        Returns
        -------
        (n,) acquisition values.
        """
        mu, sigma = predict(self.models, x)   # (n, M)
        M = mu.shape[-1]
        values = torch.zeros(x.shape[0])

        for i in range(x.shape[0]):
            acq_i = 0.0
            for pareto in self._pareto_samples:
                for m in range(M):
                    pareto_m = pareto[:, m] if pareto.ndim > 1 else pareto
                    acq_i += _entropy_reduction(
                        mu[i, m].item(),
                        sigma[i, m].item(),
                        pareto_m,
                    )
            values[i] = acq_i / self.n_pareto_samples

        return values
