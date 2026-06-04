"""Acquisition function registry and shared __init__.

Usage
-----
from acq import get_acquisition, REGISTRY

acq = get_acquisition("EHVI", models=models, pareto_y=pareto_y,
                       ref_point=ref_point)
values = acq(x_candidates)

Built-in acquisitions
---------------------
"EHVI"       – analytic / MC expected hypervolume improvement (our impl.)
"MESMO"      – max-value entropy search for multi-objective BO (our impl.)
"ParEGO"     – random Chebyshev scalarization + EI (our impl.)
"qLogNEHVI"  – qLogNoisyExpectedHypervolumeImprovement (BoTorch)
"qLogEHVI"   – qLogExpectedHypervolumeImprovement (BoTorch)
"qMOPES"     – qMultiObjectivePredictiveEntropySearch (BoTorch)
"qLBMOMES"   – qLowerBoundMultiObjectiveMaxValueEntropySearch (BoTorch)
"""

from __future__ import annotations

from acq.ehvi import EHVI
from acq.mesmo import MESMO
from acq.parego import ParEGO
from acq.botorch_wrappers import qLogNEHVI, qLogEHVI, qMOPES, qLBMOMES


REGISTRY: dict[str, type] = {
    "EHVI": EHVI,
    "MESMO": MESMO,
    "ParEGO": ParEGO,
}

FULL_REGISTRY: dict[str, type] = {
    "EHVI": EHVI,
    "MESMO": MESMO,
    "ParEGO": ParEGO,
    # BoTorch wrappers — these are factory functions, not classes,
    # so they are called with **kwargs just like the class constructors.
    "qLogNEHVI": qLogNEHVI,
    "qLogEHVI": qLogEHVI,
    "qMOPES": qMOPES,
    "qLBMOMES": qLBMOMES,
}


def get_acquisition(name: str, **kwargs):
    """Instantiate an acquisition function by name.

    Parameters
    ----------
    name:
        One of "EHVI", "MESMO", "ParEGO", "qLogNEHVI", "qLogEHVI".
    **kwargs:
        Passed directly to the acquisition constructor / factory.

    Returns
    -------
    Callable acquisition object.
    """
    if name not in REGISTRY:
        raise ValueError(f"Unknown acquisition '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)
