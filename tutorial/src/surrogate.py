"""GP surrogate model for MOBO.

Wraps GPyTorch to train one independent GP per objective, which is the
standard multi-output approach used by BoTorch internally.  Hyperparameters
are optimised with marginal-likelihood maximisation via L-BFGS.

Public API
----------
fit_gp(train_x, train_y)   -> list[ExactGPModel]
predict(models, x)         -> tuple[mu, sigma]  shapes (..., M) each
"""

from __future__ import annotations

import torch
import gpytorch
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.constraints import GreaterThan
from gpytorch.priors import LogNormalPrior
from torch import Tensor


class _ExactGPModel(ExactGP):
    def __init__(
        self,
        train_x: Tensor,
        train_y: Tensor,
        likelihood: GaussianLikelihood,
        lengthscale_prior: gpytorch.priors.Prior | None = None,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=train_x.shape[-1],
                lengthscale_prior=lengthscale_prior,
            )
        )

    def forward(self, x: Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def fit_gp(
    train_x: Tensor,
    train_y: Tensor,
    n_iter: int = 100,
    lr: float = 0.1,
    lengthscale_prior: gpytorch.priors.Prior | None = None,
    min_noise: float = 1e-4,
) -> list[_ExactGPModel]:
    """Train one GP per objective column in train_y.

    Parameters
    ----------
    train_x:
        (n, d) input tensor (already normalised to [0,1]^d is recommended).
    train_y:
        (n, M) objective tensor (standardised is recommended).
    n_iter:
        Number of L-BFGS steps for hyperparameter optimisation.
    lr:
        Learning rate for L-BFGS.
    lengthscale_prior:
        Optional GPyTorch prior placed on the Matern kernel lengthscale.
        E.g. ``LogNormalPrior(loc=0.0, scale=1.0)``.
    min_noise:
        Lower bound on the observation noise variance (default 1e-4).
        Prevents the Cholesky from becoming singular when observations
        are nearly noiseless.

    Returns
    -------
    List of M trained GP models, one per objective.
    """
    M = train_y.shape[-1]
    models = []
    for m in range(M):
        y_m = train_y[:, m]
        likelihood = GaussianLikelihood(noise_constraint=GreaterThan(min_noise))
        model = _ExactGPModel(train_x, y_m, likelihood, lengthscale_prior=lengthscale_prior)
        model.train()
        likelihood.train()

        optimizer = torch.optim.LBFGS(
            list(model.parameters()) + list(likelihood.parameters()),
            lr=lr,
            max_iter=n_iter,
            line_search_fn="strong_wolfe",
        )
        mll = ExactMarginalLogLikelihood(likelihood, model)

        def closure():
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, y_m)
            loss.backward()
            return loss

        optimizer.step(closure)

        model.eval()
        likelihood.eval()
        models.append(model)

    return models


@torch.no_grad()
def predict(models: list[_ExactGPModel], x: Tensor) -> tuple[Tensor, Tensor]:
    """Predictive mean and standard deviation for all objectives.

    Parameters
    ----------
    models:
        List of M trained GP models.
    x:
        (..., d) query tensor.

    Returns
    -------
    mu:   (..., M) predictive means.
    sigma: (..., M) predictive standard deviations.
    """
    mus, sigmas = [], []
    for model in models:
        dist = model.likelihood(model(x))
        mus.append(dist.mean)
        sigmas.append(dist.variance.sqrt())
    mu = torch.stack(mus, dim=-1)
    sigma = torch.stack(sigmas, dim=-1)
    return mu, sigma
