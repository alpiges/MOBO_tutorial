"""MOBO loop.

Ties together the surrogate model, acquisition function, and test function
into a clean iteration loop that can be driven from a notebook or script.

Public API
----------
run_mobo(...)  -> MOBOResult
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch
from torch import Tensor

from hypervolume import _nondominated, hypervolume
from surrogate import fit_gp, predict
from test_functions import MOTestFunction


@dataclass
class MOBOResult:
    """Container for the result of a MOBO run."""

    # All evaluated inputs, shape (n_init + n_iter, d)
    X: Tensor
    # All evaluated objectives, shape (n_init + n_iter, M)
    Y: Tensor
    # Hypervolume after each iteration (length n_init + n_iter)
    hv_trace: list[float]
    # Wall-clock time per iteration in seconds (length n_iter)
    iter_times: list[float]

    @property
    def pareto_y(self) -> Tensor:
        return _nondominated(self.Y)[0]

    @property
    def final_hv(self) -> float:
        return self.hv_trace[-1]


def _optimize_acquisition(
    acq_fn,
    bounds: Tensor,
    n_restarts: int = 20,
    n_raw: int = 1024,
) -> Tensor:
    """Maximise acquisition function via random restart L-BFGS.

    Parameters
    ----------
    acq_fn:
        Callable (n, d) -> (n,).
    bounds:
        (2, d) search space bounds.
    n_restarts:
        Number of gradient ascent restarts.
    n_raw:
        Number of random candidates used to seed restarts.

    Returns
    -------
    (1, d) best candidate found.
    """
    d = bounds.shape[1]
    lb, ub = bounds[0], bounds[1]

    # 1. Evaluate many random points to seed restarts
    raw = lb + torch.rand(n_raw, d) * (ub - lb)
    with torch.no_grad():
        raw_vals = acq_fn(raw)
    _, top_idx = raw_vals.topk(min(n_restarts, n_raw))
    x_init = raw[top_idx].detach().clone()

    best_x, best_val = None, -float("inf")

    for i in range(x_init.shape[0]):
        x = x_init[i].clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [x],
            lr=0.1,
            max_iter=50,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            # Project into bounds
            with torch.no_grad():
                x.clamp_(lb, ub)
            val = acq_fn(x.unsqueeze(0)).squeeze()
            loss = -val
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
        except Exception:
            pass

        with torch.no_grad():
            x.clamp_(lb, ub)
            val = acq_fn(x.unsqueeze(0)).squeeze().item()

        if val > best_val:
            best_val = val
            best_x = x.detach().clone()

    if best_x is None:
        best_x = x_init[0]
    return best_x.unsqueeze(0)


def run_mobo(
    test_fn: MOTestFunction,
    acq_name: str,
    n_init: int = 10,
    n_iter: int = 20,
    n_restarts: int = 20,
    n_raw: int = 512,
    gp_n_iter: int = 100,
    seed: int | None = None,
    verbose: bool = True,
) -> MOBOResult:
    """Run MOBO for n_iter iterations.

    Parameters
    ----------
    test_fn:
        An instantiated MOTestFunction (see test_functions.py).
    acq_name:
        One of "EHVI", "MESMO", "ParEGO".
    n_init:
        Number of initial Sobol points (evaluated before the loop).
    n_iter:
        Number of BO iterations.
    n_restarts:
        Gradient-restart restarts for acquisition optimisation.
    n_raw:
        Random candidates to seed acquisition optimisation.
    gp_n_iter:
        L-BFGS iterations for GP hyperparameter optimisation.
    seed:
        Optional random seed for reproducibility.
    verbose:
        Print progress.

    Returns
    -------
    MOBOResult
    """
    from acq import get_acquisition

    if seed is not None:
        torch.manual_seed(seed)

    ss = test_fn.search_space
    bounds = ss.bounds
    ref_point = test_fn.ref_point

    # ------------------------------------------------------------------
    # Initial design
    # ------------------------------------------------------------------
    d = bounds.shape[1]
    lb, ub = bounds[0], bounds[1]
    X = lb + torch.rand(n_init, d) * (ub - lb)
    Y = test_fn(X)

    hv_trace: list[float] = []
    iter_times: list[float] = []

    def _hv_now():
        nd = _nondominated(Y)[0]
        return hypervolume(nd, ref_point)

    hv_trace.append(_hv_now())

    # ------------------------------------------------------------------
    # BO loop
    # ------------------------------------------------------------------
    for i in range(n_iter):
        t_start = time.perf_counter()

        # Normalise inputs, standardise outputs for GP stability
        X_mean, X_std = X.mean(0), X.std(0).clamp(min=1e-6)
        Y_mean, Y_std = Y.mean(0), Y.std(0).clamp(min=1e-6)
        X_norm = (X - X_mean) / X_std
        Y_norm = (Y - Y_mean) / Y_std
        bounds_norm = torch.stack([(lb - X_mean) / X_std, (ub - X_mean) / X_std])

        # Fit GPs
        models = fit_gp(X_norm, Y_norm, n_iter=gp_n_iter)

        # Build acquisition
        pareto_y = _nondominated(Y)[0]
        pareto_y_norm = (pareto_y - Y_mean) / Y_std
        ref_norm = (ref_point - Y_mean) / Y_std

        if acq_name == "EHVI":
            acq = get_acquisition("EHVI", models=models,
                                  pareto_y=pareto_y_norm, ref_point=ref_norm)
        elif acq_name == "MESMO":
            acq = get_acquisition("MESMO", models=models, bounds=bounds_norm)
        elif acq_name == "ParEGO":
            acq = get_acquisition("ParEGO", models=models, train_y=Y_norm)
        elif acq_name == "qLogNEHVI":
            acq = get_acquisition("qLogNEHVI", models=models, ref_point=ref_norm)
        elif acq_name == "qLogEHVI":
            acq = get_acquisition("qLogEHVI", models=models,
                                  ref_point=ref_norm, pareto_y=pareto_y_norm)
        elif acq_name == "qMOPES":
            acq = get_acquisition("qMOPES", models=models, bounds=bounds_norm)
        elif acq_name == "qLBMOMES":
            acq = get_acquisition("qLBMOMES", models=models,
                                  ref_point=ref_norm, pareto_y=pareto_y_norm)
        else:
            raise ValueError(f"Unknown acquisition: {acq_name}")

        # Optimise acquisition in normalised space
        x_next_norm = _optimize_acquisition(acq, bounds_norm, n_restarts, n_raw)

        # Un-normalise
        x_next = x_next_norm * X_std + X_mean
        x_next = x_next.clamp(lb, ub)

        # Evaluate
        y_next = test_fn(x_next)
        X = torch.cat([X, x_next], dim=0)
        Y = torch.cat([Y, y_next], dim=0)

        elapsed = time.perf_counter() - t_start
        iter_times.append(elapsed)
        hv_trace.append(_hv_now())

        if verbose:
            hv_now = hv_trace[-1]
            log_hv = math.log(hv_now) if hv_now > 0 else float("-inf")
            print(f"  iter {i+1:3d}/{n_iter}  HV={hv_now:.6g}  log(HV)={log_hv:.4f}  "
                  f"({elapsed:.1f}s)")

    return MOBOResult(X=X, Y=Y, hv_trace=hv_trace, iter_times=iter_times)
