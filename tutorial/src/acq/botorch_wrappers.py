"""BoTorch acquisition wrappers for the MOBO tutorial framework.

Bridges BoTorch multi-objective acquisitions to the ``(n, d) → (n,)``
interface expected by ``run_mobo`` / ``_optimize_acquisition``.

How it works
------------
BoTorch acquisitions operate on a :class:`botorch.models.model.Model` and
expect inputs of shape ``(batch, q, d)`` instead of our flat ``(n, d)``.
:class:`_WrappedModelList` adapts our already-trained GPyTorch
``_ExactGPModel`` list to satisfy the BoTorch ``Model`` interface without
re-fitting.  :class:`BoTorchAcquisition` then handles the ``(n, d) ↔
(n, 1, d)`` reshape so the rest of the framework sees the familiar
signature.

Public API
----------
BoTorchAcquisition          – generic adapter; pass any BoTorch acq class
qLogNEHVI(...)              – qLogNoisyExpectedHypervolumeImprovement
qLogEHVI(...)               – qLogExpectedHypervolumeImprovement
qMOPES(...)                 – qMultiObjectivePredictiveEntropySearch
qLBMOMES(...)               – qLowerBoundMultiObjectiveMaxValueEntropySearch
"""

from __future__ import annotations

import torch
from torch import Tensor

from botorch.models.model_list_gp_regression import ModelListGP as _ModelListGP
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultitaskMultivariateNormal

from surrogate import _ExactGPModel


# ---------------------------------------------------------------------------
# BoTorch model adapter
# ---------------------------------------------------------------------------

class _WrappedModelList(_ModelListGP):
    """Thin BoTorch ``Model`` adapter over a list of GPyTorch ExactGP models.

    BoTorch acquisitions require a ``Model`` with a ``posterior()`` that
    returns a :class:`~botorch.posteriors.gpytorch.GPyTorchPosterior`.
    We delegate to the already-trained GPyTorch models, so no re-fitting
    is needed.

    We inherit from ``ModelListGP`` so that acquisitions that use
    ``isinstance(model, ModelListGP)`` as a dispatch condition (e.g.
    ``qLBMOMES``) take the correct code path.  ``ModelListGP.__init__`` is
    bypassed because it expects ``SingleTaskGP`` instances; we call
    ``torch.nn.Module.__init__`` directly instead.

    Parameters
    ----------
    models:
        List of M trained :class:`~surrogate._ExactGPModel` instances,
        one per objective.
    """

    def __init__(self, models: list[_ExactGPModel]) -> None:
        # Bypass ModelListGP.__init__ (requires SingleTaskGP); initialise
        # the Module directly and register the sub-models manually.
        torch.nn.Module.__init__(self)
        self.models = torch.nn.ModuleList(models)
        self._models = list(models)

    @property
    def num_outputs(self) -> int:
        return len(self._models)

    @property
    def train_inputs(self) -> tuple:
        """Expose training inputs in ModelListGP format: ``((X,), (X,), ...)``.

        Some BoTorch acquisitions (e.g. ``qMOPES``) access
        ``model.train_inputs[m][0]`` to retrieve the training data of the
        m-th output.  Each ``_ExactGPModel.train_inputs`` is already a
        1-tuple ``(train_x,)``, so stacking them gives the expected layout.
        """
        return tuple(m.train_inputs for m in self._models)

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool = False,
        **kwargs,
    ) -> GPyTorchPosterior:
        """Compute the joint posterior over all (or selected) objectives.

        Parameters
        ----------
        X:
            ``(*batch_shape, q, d)`` query tensor — BoTorch convention.
        output_indices:
            Subset of objective indices to return.  ``None`` means all.
        observation_noise:
            If ``True``, folds the likelihood noise into the posterior
            variance (equivalent to calling ``model.likelihood(model(X))``).

        Returns
        -------
        :class:`~botorch.posteriors.gpytorch.GPyTorchPosterior`
        """
        selected = (
            self._models if output_indices is None
            else [self._models[i] for i in output_indices]
        )
        mvns = []
        for model in selected:
            mvn = model.likelihood(model(X)) if observation_noise else model(X)
            mvns.append(mvn)

        if len(mvns) == 1:
            return GPyTorchPosterior(mvns[0])

        joint = MultitaskMultivariateNormal.from_independent_mvns(mvns)
        return GPyTorchPosterior(joint)


# ---------------------------------------------------------------------------
# Generic acquisition wrapper
# ---------------------------------------------------------------------------

class BoTorchAcquisition:
    """Adapts any BoTorch multi-objective acquisition to the MOBO framework.

    The framework expects::

        acq(x: Tensor[n, d]) -> Tensor[n]

    BoTorch acquisitions expect::

        acq.forward(X: Tensor[batch, q, d]) -> Tensor[batch]

    This class:

    * Wraps the GP model list in :class:`_WrappedModelList`.
    * Instantiates the requested BoTorch acquisition class.
    * Handles the ``(n, d) ↔ (n, 1, d)`` reshape in ``__call__``.

    Parameters
    ----------
    models:
        List of M trained GP models (``surrogate.py``).
    botorch_acq_class:
        Any BoTorch acquisition class whose constructor accepts ``model``
        and ``ref_point`` plus whatever is in ``acq_kwargs``.
    ref_point:
        ``(M,)`` reference point in normalised objective space.
    **acq_kwargs:
        All remaining keyword arguments forwarded verbatim to the BoTorch
        acquisition constructor.  Prefer the convenience functions below.

    Examples
    --------
    >>> from botorch.acquisition.multi_objective.logei import (
    ...     qLogNoisyExpectedHypervolumeImprovement,
    ... )
    >>> acq = BoTorchAcquisition(
    ...     models=models,
    ...     botorch_acq_class=qLogNoisyExpectedHypervolumeImprovement,
    ...     ref_point=ref_norm,
    ...     X_baseline=X_norm,
    ... )
    >>> values = acq(x_candidates)   # (n, d) -> (n,)
    """

    name = "BoTorch"

    def __init__(
        self,
        models: list[_ExactGPModel],
        botorch_acq_class,
        ref_point: Tensor | None = None,
        **acq_kwargs,
    ) -> None:
        self._bt_model = _WrappedModelList(models)
        if ref_point is not None:
            acq_kwargs = {"ref_point": ref_point, **acq_kwargs}
        self._acq = botorch_acq_class(
            model=self._bt_model,
            **acq_kwargs,
        )

    def __call__(self, x: Tensor) -> Tensor:
        """Evaluate the acquisition function.

        Parameters
        ----------
        x:
            ``(n, d)`` candidates in normalised input space.

        Returns
        -------
        ``(n,)`` acquisition values (higher is better).
        """
        X = x.unsqueeze(-2)  # (n, d) → (n, 1, d)  [q = 1 batch]
        return self._acq(X)


# ---------------------------------------------------------------------------
# Helper: sample Pareto input sets from the GP posterior
# ---------------------------------------------------------------------------

def _sample_pareto_inputs(
    models: list[_ExactGPModel],
    bounds: Tensor,
    num_pareto_samples: int = 10,
    num_grid: int = 256,
) -> Tensor:
    """Sample approximate Pareto-optimal input sets from the GP posterior.

    For each of ``num_pareto_samples`` independent posterior function draws,
    evaluates a random grid of candidates, identifies the non-dominated
    objective vectors (maximisation convention), and collects the
    corresponding input points.  Variable-length Pareto sets are padded to
    the maximum observed size by repeating the last row.

    Parameters
    ----------
    models:
        List of M trained GP models.
    bounds:
        ``(2, d)`` lower/upper bounds of the (normalised) search space.
    num_pareto_samples:
        Number of independent Pareto-set samples to draw.
    num_grid:
        Number of random candidates to evaluate per sample.

    Returns
    -------
    Tensor of shape ``(num_pareto_samples, max_pareto_size, d)``.
    """
    from hypervolume import _nondominated

    d = bounds.shape[1]
    all_pareto_x: list[Tensor] = []

    with torch.no_grad():
        for _ in range(num_pareto_samples):
            grid = bounds[0] + torch.rand(num_grid, d) * (bounds[1] - bounds[0])
            samples = []
            for model in models:
                dist = model.likelihood(model(grid))
                samples.append(dist.sample())   # (num_grid,)
            y_sample = torch.stack(samples, dim=-1)   # (num_grid, M)
            _, mask = _nondominated(y_sample)
            all_pareto_x.append(grid[mask])

    # Pad samples to a uniform shape by repeating the last Pareto point
    max_p = max(px.shape[0] for px in all_pareto_x)
    padded = []
    for px in all_pareto_x:
        if px.shape[0] < max_p:
            pad = px[-1:].expand(max_p - px.shape[0], -1)
            px = torch.cat([px, pad], dim=0)
        padded.append(px)
    return torch.stack(padded, dim=0)  # (num_pareto_samples, max_p, d)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def qLogNEHVI(
    models: list[_ExactGPModel],
    ref_point: Tensor,
    X_baseline: Tensor | None = None,
    **acq_kwargs,
) -> BoTorchAcquisition:
    """``qLogNoisyExpectedHypervolumeImprovement`` (BoTorch).

    The current state-of-the-art MOBO acquisition in BoTorch.  Uses a noisy
    EHVI formulation that is robust to observation noise and does not require
    an explicit Pareto-front decomposition to be passed in.

    Training inputs are extracted from the GP models automatically when
    ``X_baseline`` is not provided.

    Parameters
    ----------
    models:
        List of M trained GP models.
    ref_point:
        ``(M,)`` reference point in normalised objective space.
    X_baseline:
        ``(n, d)`` training inputs used as the noisy EHVI baseline.
        Defaults to the data stored inside the first GP model.
    **acq_kwargs:
        Forwarded to ``qLogNoisyExpectedHypervolumeImprovement``
        (e.g. ``prune_baseline=True``, ``cache_root=True``).
    """
    from botorch.acquisition.multi_objective.logei import (
        qLogNoisyExpectedHypervolumeImprovement,
    )
    if X_baseline is None:
        X_baseline = models[0].train_inputs[0]
    return BoTorchAcquisition(
        models=models,
        botorch_acq_class=qLogNoisyExpectedHypervolumeImprovement,
        ref_point=ref_point,
        X_baseline=X_baseline,
        **acq_kwargs,
    )


def qLogEHVI(
    models: list[_ExactGPModel],
    ref_point: Tensor,
    pareto_y: Tensor,
    **acq_kwargs,
) -> BoTorchAcquisition:
    """``qLogExpectedHypervolumeImprovement`` (BoTorch).

    The analytic q-EHVI variant.  Requires the current Pareto front to build
    the non-dominated box decomposition.

    Parameters
    ----------
    models:
        List of M trained GP models.
    ref_point:
        ``(M,)`` reference point in normalised objective space.
    pareto_y:
        ``(p, M)`` current Pareto front in normalised objective space.
    **acq_kwargs:
        Forwarded to ``qLogExpectedHypervolumeImprovement``.
    """
    from botorch.acquisition.multi_objective.logei import (
        qLogExpectedHypervolumeImprovement,
    )
    from botorch.utils.multi_objective.box_decompositions.non_dominated import (
        FastNondominatedPartitioning,
    )
    partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=pareto_y)
    return BoTorchAcquisition(
        models=models,
        botorch_acq_class=qLogExpectedHypervolumeImprovement,
        ref_point=ref_point,
        partitioning=partitioning,
        **acq_kwargs,
    )


def qMOPES(
    models: list[_ExactGPModel],
    bounds: Tensor,
    num_pareto_samples: int = 10,
    num_grid: int = 256,
    **acq_kwargs,
) -> BoTorchAcquisition:
    """``qMultiObjectivePredictiveEntropySearch`` (BoTorch).

    Predictive-entropy-search acquisition for multi-objective BO.  Selects
    points expected to maximally reduce uncertainty about the Pareto-optimal
    input set.

    Pareto input-set samples are drawn automatically from the GP posterior
    by evaluating a random grid and extracting non-dominated points.  A
    fresh grid is used for each sample to improve diversity.

    Parameters
    ----------
    models:
        List of M trained GP models.
    bounds:
        ``(2, d)`` lower/upper bounds of the normalised search space.
        Used to generate the sampling grid.
    num_pareto_samples:
        Number of Pareto-set posterior samples (``pareto_sets`` first dim).
    num_grid:
        Grid size per sample used when drawing Pareto-set samples.
    **acq_kwargs:
        Forwarded to ``qMultiObjectivePredictiveEntropySearch``
        (e.g. ``max_ep_iterations``, ``ep_jitter``).
    """
    from botorch.acquisition.multi_objective.predictive_entropy_search import (
        qMultiObjectivePredictiveEntropySearch,
    )
    pareto_sets = _sample_pareto_inputs(
        models, bounds, num_pareto_samples=num_pareto_samples, num_grid=num_grid
    )
    return BoTorchAcquisition(
        models=models,
        botorch_acq_class=qMultiObjectivePredictiveEntropySearch,
        pareto_sets=pareto_sets,
        **acq_kwargs,
    )


def qLBMOMES(
    models: list[_ExactGPModel],
    ref_point: Tensor,
    pareto_y: Tensor,
    **acq_kwargs,
) -> BoTorchAcquisition:
    """``qLowerBoundMultiObjectiveMaxValueEntropySearch`` (BoTorch).

    Lower-bound max-value entropy search for multi-objective BO (LB-MOMЕС).
    Related to MESMO: selects points that maximally reduce entropy about the
    Pareto-optimal objective values.  Uses a lower-bound approximation that
    is cheaper to evaluate than the exact MOMЕС.

    Parameters
    ----------
    models:
        List of M trained GP models.
    ref_point:
        ``(M,)`` reference point in normalised objective space.
        Used to build the non-dominated box decomposition.
    pareto_y:
        ``(p, M)`` current Pareto front in normalised objective space.
    **acq_kwargs:
        Forwarded to ``qLowerBoundMultiObjectiveMaxValueEntropySearch``
        (e.g. ``estimation_type``, ``num_samples``).
    """
    from botorch.acquisition.multi_objective.max_value_entropy_search import (
        qLowerBoundMultiObjectiveMaxValueEntropySearch,
    )
    from botorch.utils.multi_objective.box_decompositions.non_dominated import (
        FastNondominatedPartitioning,
    )
    partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=pareto_y)
    # qLBMOMES requires shape (num_pareto_samples, 2, num_boxes, M);
    # unsqueeze adds the leading num_pareto_samples=1 dimension.
    hypercell_bounds = partitioning.hypercell_bounds.unsqueeze(0)
    return BoTorchAcquisition(
        models=models,
        botorch_acq_class=qLowerBoundMultiObjectiveMaxValueEntropySearch,
        hypercell_bounds=hypercell_bounds,
        **acq_kwargs,
    )
