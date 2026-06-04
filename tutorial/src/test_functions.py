"""Multi-objective test functions.

Each function is a callable that maps a torch.Tensor of shape (..., d)
to a tensor of shape (..., M), where d is the input dimension and M is
the number of objectives.  All objectives are *maximised* by convention;
negate objectives as needed inside each class.

Provided functions
------------------
Schaffer1      d=1, M=2
BraninCurrin   d=2, M=2
ZDT1           d=6, M=2
DTLZ2          d=6, M=2
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SearchSpace:
    """Axis-aligned box in R^d."""

    bounds: Tensor  # shape (2, d) – row 0 = lower, row 1 = upper

    @property
    def d(self) -> int:
        return self.bounds.shape[1]

    def normalize(self, x: Tensor) -> Tensor:
        """Map x from bounds to [0, 1]^d."""
        lb, ub = self.bounds[0], self.bounds[1]
        return (x - lb) / (ub - lb)

    def unnormalize(self, x: Tensor) -> Tensor:
        """Map x from [0, 1]^d back to bounds."""
        lb, ub = self.bounds[0], self.bounds[1]
        return lb + x * (ub - lb)


class MOTestFunction(ABC):
    """Base class for multi-objective test functions."""

    #: Input dimension
    d: int
    #: Number of objectives (all maximised)
    M: int
    #: Optional reference point for hypervolume computation (M,)
    ref_point: Tensor

    @property
    @abstractmethod
    def search_space(self) -> SearchSpace:
        ...

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        """Evaluate objectives.

        Parameters
        ----------
        x:
            (..., d) tensor, each row a candidate in the search space.

        Returns
        -------
        Tensor of shape (..., M), all objectives to be maximised.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(d={self.d}, M={self.M})"


# ---------------------------------------------------------------------------
# Sinusoidal  (d=1, M=2)
# ---------------------------------------------------------------------------

class Sinusoidal(MOTestFunction):
    """Sinusoidal functions (d=1, M=2).

    For x in [-2, 4]:
      g1(x) = sin(2πx) * (0.5x)
      g2(x) = 2 * (sin(2π(x + 0.3)) - sin(x) + 1) * exp(-(x - 2)^2 / 3)

    The implementation returns [-g1(x), -g2(x)] to match the
    maximisation convention used in the tutorial.
    """

    d = 1
    M = 2

    def __init__(self) -> None:
        self.ref_point = torch.tensor([-3.0, -3.0])

    @property
    def search_space(self) -> SearchSpace:
        bounds = torch.tensor([[-2.0], [4.0]])
        return SearchSpace(bounds=bounds)

    def __call__(self, x: Tensor) -> Tensor:
        x0 = x[..., 0]
        f1 = torch.sin(2 * torch.pi * x0) * (0.5 * x0)
        f2 = 2*(torch.sin(2 * torch.pi * (x0 + 0.3)) - torch.sin(x0) + 1) * torch.exp(-(x0-2)**2 / 3)
        return torch.stack([-f1, -f2], dim=-1)

# ---------------------------------------------------------------------------
# BraninCurrin  (d=2, M=2)
# ---------------------------------------------------------------------------

class BraninCurrin(MOTestFunction):
    """Two independent objectives: (negative) Branin and (negative) Currin."""

    d = 2
    M = 2

    def __init__(self) -> None:
        self.ref_point = torch.tensor([-200.0, -10.0])

    @property
    def search_space(self) -> SearchSpace:
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        return SearchSpace(bounds=bounds)

    def _branin(self, x: Tensor) -> Tensor:
        # x in [0,1]^2, rescaled internally
        x1 = x[..., 0] * 15.0 - 5.0
        x2 = x[..., 1] * 15.0
        a, b, c = 1.0, 5.1 / (4.0 * torch.pi**2), 5.0 / torch.pi
        r, s, t = 6.0, 10.0, 1.0 / (8.0 * torch.pi)
        return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * torch.cos(x1) + s

    def _currin(self, x: Tensor) -> Tensor:
        x1 = x[..., 0]
        x2 = x[..., 1].clamp(min=1e-6)
        factor1 = 1.0 - torch.exp(-1.0 / (2.0 * x2))
        factor2 = (2300.0 * x1**3 + 1900.0 * x1**2 + 2092.0 * x1 + 60.0) / (
            100.0 * x1**3 + 500.0 * x1**2 + 4.0 * x1 + 20.0
        )
        return factor1 * factor2

    def __call__(self, x: Tensor) -> Tensor:
        return torch.stack([-self._branin(x), -self._currin(x)], dim=-1)


# ---------------------------------------------------------------------------
# ZDT1  (d=6, M=2)
# ---------------------------------------------------------------------------

class ZDT1(MOTestFunction):
    """ZDT1 test problem.  All inputs in [0, 1]^d."""

    d = 6
    M = 2

    def __init__(self) -> None:
        self.ref_point = torch.tensor([-1.1, -1.1])

    @property
    def search_space(self) -> SearchSpace:
        bounds = torch.zeros(2, self.d)
        bounds[1] = 1.0
        return SearchSpace(bounds=bounds)

    def __call__(self, x: Tensor) -> Tensor:
        f1 = x[..., 0]
        g = 1.0 + 9.0 * x[..., 1:].mean(dim=-1)
        h = 1.0 - torch.sqrt(f1 / g)
        f2 = g * h
        # Maximise: return negatives so optima are at large values
        return torch.stack([-f1, -f2], dim=-1)


# ---------------------------------------------------------------------------
# DTLZ2  (d=6, M=2)
# ---------------------------------------------------------------------------

class DTLZ2(MOTestFunction):
    """DTLZ2 test problem (2-objective variant).  Inputs in [0, 1]^d."""

    d = 6
    M = 2

    def __init__(self) -> None:
        self.ref_point = torch.tensor([-1.1, -1.1])

    @property
    def search_space(self) -> SearchSpace:
        bounds = torch.zeros(2, self.d)
        bounds[1] = 1.0
        return SearchSpace(bounds=bounds)

    def __call__(self, x: Tensor) -> Tensor:
        theta = x[..., 0] * (torch.pi / 2.0)
        k = self.d - 1
        g = ((x[..., 1:] - 0.5) ** 2).sum(dim=-1)
        r = 1.0 + g
        f1 = r * torch.cos(theta)
        f2 = r * torch.sin(theta)
        return torch.stack([-f1, -f2], dim=-1)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, type[MOTestFunction]] = {
    "Sinusoidal": Sinusoidal,
    "BraninCurrin": BraninCurrin,
    "ZDT1": ZDT1,
    "DTLZ2": DTLZ2,
}


def get_test_function(name: str) -> MOTestFunction:
    """Instantiate a test function by name."""
    if name not in REGISTRY:
        raise ValueError(f"Unknown test function '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name]()
