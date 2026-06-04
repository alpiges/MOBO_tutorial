"""Expected Hypervolume Improvement (EHVI) acquisition function.

For M=2 objectives, we implement the analytic closed-form EHVI derived in
Emmerich et al. (2006), which decomposes the non-dominated space into
axis-aligned cells and computes the expected improvement over each cell.

For general M we provide a simple Monte Carlo fallback.

References
----------
Emmerich, M., Giannakoglou, K., & Naujoks, B. (2006). Single- and
multiobjective evolutionary optimization assisted by Gaussian random field
metamodels. IEEE TEC, 10(4), 421-439.

Daulton, S., Balandat, M., & Bakshy, E. (2020). Differentiable expected
hypervolume improvement for parallel multi-objective Bayesian optimization.
NeurIPS 2020.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.distributions import Normal

from hypervolume import _nondominated
from surrogate import _ExactGPModel, predict


_normal = Normal(0.0, 1.0)


def _psi(mu: float, sigma: float, a: float, b: float) -> float:
    """Ψ(a, b | μ, σ) = ∫_a^b (y - a) φ((y-μ)/σ) dy / σ.

    Used in the analytic EHVI cell integral for one dimension.
    """
    if sigma < 1e-12:
        val = max(min(mu, b) - a, 0.0)
        return val
    ga = (a - mu) / sigma
    gb = (b - mu) / sigma
    phi_a = math.exp(_normal.log_prob(torch.tensor(ga)).item())
    phi_b = math.exp(_normal.log_prob(torch.tensor(gb)).item())
    Phi_a = _normal.cdf(torch.tensor(ga)).item()
    Phi_b = _normal.cdf(torch.tensor(gb)).item()
    return sigma * (phi_a - phi_b) + (mu - a) * (Phi_b - Phi_a)


def _ehvi_2d_single(mu: Tensor, sigma: Tensor, pareto_y: Tensor, ref_point: Tensor) -> Tensor:
    """Analytic EHVI for a single candidate with M=2.

    Parameters
    ----------
    mu:         (2,) predictive mean.
    sigma:      (2,) predictive std.
    pareto_y:   (p, 2) current Pareto set.
    ref_point:  (2,) reference point.

    Returns
    -------
    Scalar EHVI value.
    """
    # Build cell decomposition along dim-0 (sorted by first objective)
    ref = ref_point
    pts = pareto_y
    # Add sentinel points
    upper1 = pts[:, 0].max() + 1e6
    upper2 = pts[:, 1].max() + 1e6

    # Sort Pareto points by obj1 descending → determines cells
    order = pts[:, 0].argsort(descending=True)
    pts_sorted = pts[order]

    # Build breakpoints along obj1 (upper) and obj2 (lower) per cell
    # Cells: between consecutive Pareto points
    cells = []  # (lo1, hi1, lo2, hi2)
    prev_y2 = ref[1].item()
    for i, pt in enumerate(pts_sorted):
        lo1 = pt[0].item()
        hi1 = (pts_sorted[i - 1][0].item() if i > 0 else upper1.item())
        lo2 = prev_y2
        hi2 = pt[1].item()
        if hi1 > lo1 and hi2 > lo2:
            cells.append((lo1, hi1, lo2, hi2))
        prev_y2 = max(prev_y2, pt[1].item())
    # Final cell above the last Pareto point
    if pts_sorted.shape[0] > 0:
        lo1 = ref[0].item()
        hi1 = pts_sorted[-1][0].item()
        lo2 = prev_y2
        hi2 = upper2.item()
        if hi1 > lo1 and hi2 > lo2:
            cells.append((lo1, hi1, lo2, hi2))

    mu0, s0 = mu[0].item(), sigma[0].item()
    mu1, s1 = mu[1].item(), sigma[1].item()

    ehvi = 0.0
    for (lo1, hi1, lo2, hi2) in cells:
        psi1 = _psi(mu0, s0, lo1, hi1)
        psi2 = _psi(mu1, s1, lo2, hi2)
        ehvi += psi1 * psi2

    return torch.tensor(ehvi, dtype=mu.dtype)


def _ehvi_mc(
    mu: Tensor,
    sigma: Tensor,
    pareto_y: Tensor,
    ref_point: Tensor,
    n_samples: int = 512,
) -> Tensor:
    """Monte Carlo EHVI for a single candidate, general M."""
    M = mu.shape[-1]
    eps = torch.randn(n_samples, M) * sigma + mu   # (S, M)
    hv_base = _compute_hv_contrib(pareto_y, ref_point)
    improvements = []
    for s in range(n_samples):
        combined = torch.cat([pareto_y, eps[s].unsqueeze(0)], dim=0)
        combined = _nondominated(combined)[0]
        hv_new = _compute_hv_contrib(combined, ref_point)
        improvements.append(max(hv_new - hv_base, 0.0))
    return torch.tensor(sum(improvements) / n_samples, dtype=mu.dtype)


def _compute_hv_contrib(points: Tensor, ref_point: Tensor) -> float:
    from hypervolume import hypervolume
    return hypervolume(points, ref_point)


class EHVI:
    """Expected Hypervolume Improvement acquisition function.

    Uses the analytic formula for M=2 and a Monte Carlo estimate for M>2.

    Parameters
    ----------
    models:
        List of M trained GP models (one per objective, see surrogate.py).
    pareto_y:
        (p, M) current Pareto front in objective space.
    ref_point:
        (M,) reference point for hypervolume computation.
    n_mc_samples:
        Number of MC samples used when M>2 (default 512).
    """

    name = "EHVI"

    def __init__(
        self,
        models: list[_ExactGPModel],
        pareto_y: Tensor,
        ref_point: Tensor,
        n_mc_samples: int = 512,
    ) -> None:
        self.models = models
        self.pareto_y = pareto_y
        self.ref_point = ref_point
        self.n_mc_samples = n_mc_samples

    def __call__(self, x: Tensor) -> Tensor:
        """Evaluate EHVI at candidate points x.

        Parameters
        ----------
        x:
            (n, d) candidates in the normalised search space.

        Returns
        -------
        (n,) acquisition values.
        """
        mu, sigma = predict(self.models, x)   # (n, M) each
        M = mu.shape[-1]
        values = []
        for i in range(x.shape[0]):
            if M == 2:
                v = _ehvi_2d_single(mu[i], sigma[i], self.pareto_y, self.ref_point)
            else:
                v = _ehvi_mc(mu[i], sigma[i], self.pareto_y, self.ref_point, self.n_mc_samples)
            values.append(v)
        return torch.stack(values)
