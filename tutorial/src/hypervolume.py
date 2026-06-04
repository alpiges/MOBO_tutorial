"""Hypervolume computation utilities.

We implement an exact hypervolume indicator for M=2 objectives in O(n log n)
and fall back to a Monte Carlo estimate for M>2.  Both conventions use
*maximisation* (larger values are better).

Public API
----------
hypervolume(pareto_y, ref_point)     -> float
hypervolume_improvement(y, pareto_y, ref_point) -> Tensor  shape (n,)
"""

from __future__ import annotations

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Exact M=2 hypervolume  (Daulton et al. 2020 style sweep)
# ---------------------------------------------------------------------------

def _hv_2d(points: Tensor, ref: Tensor) -> float:
    """Exact hypervolume for M=2 objectives in maximisation convention."""
    # Filter dominated by reference
    mask = (points > ref).all(dim=-1)
    pts = points[mask]
    if pts.shape[0] == 0:
        return 0.0
    # Sort descending by first objective
    idx = pts[:, 0].argsort(descending=True)
    pts = pts[idx]
    hv = 0.0
    prev_y = ref[1].item()
    for pt in pts:
        x, y = pt[0].item(), pt[1].item()
        if y > prev_y:
            hv += (x - ref[0].item()) * (y - prev_y)
            prev_y = y
    return hv


def _hv_mc(points: Tensor, ref: Tensor, n_samples: int = 100_000) -> float:
    """Monte Carlo hypervolume estimate for M>=2."""
    mask = (points > ref).all(dim=-1)
    pts = points[mask]
    if pts.shape[0] == 0:
        return 0.0
    upper = pts.max(dim=0).values
    # Random uniform samples in the bounding box [ref, upper]
    samples = ref + torch.rand(n_samples, ref.shape[0]) * (upper - ref)
    # A sample is dominated if any point dominates it
    dominated = (pts.unsqueeze(0) >= samples.unsqueeze(1)).all(dim=-1).any(dim=-1)
    vol = (upper - ref).prod().item()
    return vol * dominated.float().mean().item()


def hypervolume(pareto_y: Tensor, ref_point: Tensor) -> float:
    """Compute the hypervolume indicator of a set of points.

    Parameters
    ----------
    pareto_y:
        (n, M) tensor of objective vectors (maximisation).
    ref_point:
        (M,) reference point; all objectives should be worse than any
        point in pareto_y.

    Returns
    -------
    Scalar hypervolume value.
    """
    if pareto_y.shape[-1] == 2:
        return _hv_2d(pareto_y, ref_point)
    return _hv_mc(pareto_y, ref_point)


# ---------------------------------------------------------------------------
# Per-point hypervolume improvement
# ---------------------------------------------------------------------------

def hypervolume_improvement(
    candidates: Tensor,
    current_pareto: Tensor,
    ref_point: Tensor,
) -> Tensor:
    """Compute exact hypervolume improvement for each candidate.

    This is a deterministic computation on *observed* objective vectors
    (used e.g. for greedy selection after fantasy evaluations).  For
    expected HVI over the posterior see acq/ehvi.py.

    Parameters
    ----------
    candidates:
        (n, M) tensor of candidate objective vectors.
    current_pareto:
        (p, M) current Pareto set.
    ref_point:
        (M,) reference point.

    Returns
    -------
    (n,) tensor of hypervolume improvements.
    """
    hv_base = hypervolume(current_pareto, ref_point)
    improvements = []
    for i in range(candidates.shape[0]):
        combined = torch.cat([current_pareto, candidates[i].unsqueeze(0)], dim=0)
        # Keep non-dominated
        combined = _nondominated(combined)[0]
        hv_new = hypervolume(combined, ref_point)
        improvements.append(hv_new - hv_base)
    return torch.tensor(improvements)


def _nondominated(points: Tensor) -> Tensor:
    """Return the non-dominated subset of points (maximisation)."""
    n = points.shape[0]
    is_dominated = torch.zeros(n, dtype=torch.bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (points[j] >= points[i]).all() and (points[j] > points[i]).any():
                is_dominated[i] = True
                break
    return points[~is_dominated], ~is_dominated
