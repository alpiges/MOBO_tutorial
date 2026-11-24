# MOBO Tutorial

A Python tutorial for **multi-objective Bayesian optimization (MOBO)** originally developped for [NUMRAD26 Summer School](https://numrad.epfl.ch/) at EPFL, Lausanne (CH) and updated for the [Winter School on Robotics and AI for Materials Chemistry 2026](https://aichemy.ac.uk/event/winter-school-robotics-and-ai-for-materials-chemistry-2026/) at University of Liverpool (UK). The tutorial combines from-scratch implementations of key MOBO components and Marimo notebook tutorials for illustration.

The main idea of the tutorial is to showcase the key building blocks of multi-objective Bayesian optimization. The implementation of Gaussian process inference uses [gpytorch](https://gpytorch.ai/), but the main MOBO loop and some acquisition functions are implemented from scratch for demonstration purposes.
The directly implemented acquistion functions are complemented by additional ones wrapped from [botorch](https://botorch.org/).

---

## Slides
The slides with the relevant mathematical background are in `slides/`, assuming prior knowledge about Gaussian processes and single-objective Bayesian optimization.

---

## Installation

Python ≥ 3.11 is required. The project uses [uv](https://github.com/astral-sh/uv) for environment and dependency management.

```bash
# Clone the repository and enter the tutorial directory
cd tutorial

# Create a virtual environment and install all dependencies
uv sync

# Activate the environment
source .venv/bin/activate
```

To also install the development dependencies (marimo, pytest, ruff):

```bash
uv sync --group dev
```

### Running the notebooks

With the environment activated:

```bash
cd tutorial
marimo edit notebooks/mobo_1D.py        # 1D toy problem
marimo edit notebooks/mobo_tutorial.py  # full tutorial
```

---

## Code overview

All source code lives under `tutorial/src/`.

### `mobo.py` — Optimization loop

The main entry point is `run_mobo()`. It:

1. Draws an initial random design.
2. Fits independent Gaussian process surrogates (one per objective).
3. Constructs and maximizes an acquisition function via random-restart L-BFGS.
4. Evaluates the next point and logs the hypervolume indicator.

Returns a `MOBOResult` dataclass containing inputs, objectives, hypervolume trace, and per-iteration wall times.

### `surrogate.py` — Gaussian process surrogate

`fit_gp()` trains M independent GPs (one per objective column) using GPyTorch with a Matérn-5/2 ARD kernel and a constant mean. `predict()` returns predictive means and standard deviations stacked across objectives.

### `hypervolume.py` — Hypervolume indicator

`hypervolume(pareto_y, ref_point)` computes the hypervolume dominated by a Pareto front relative to a reference point. An exact O(n log n) sweep is used for M=2; a Monte Carlo estimate (100k samples) is used for M>2. The module uses the **maximization convention** (larger objective values are better).

### `test_functions.py` — Benchmark problems

Provides four multi-objective test functions (all negated for maximization):

| Class | d | M | Description |
|---|---|---|---|
| `Sinusoidal` | 1 | 2 | Sine-based toy problem |
| `BraninCurrin` | 2 | 2 | Independent Branin and Currin functions |
| `ZDT1` | 6 | 2 | ZDT benchmark suite problem 1 |
| `DTLZ2` | 6 | 2 | Parametric spherical Pareto front |

All test functions expose a `SearchSpace` object with `normalize` / `unnormalize` helpers and a `ref_point` suitable for hypervolume computation.

### `acq/` — Acquisition functions

| Module | Class | Description |
|---|---|---|
| `ehvi.py` | `EHVI` | Expected Hypervolume Improvement. Analytic for M=2 via cell decomposition; Monte Carlo for M>2. |
| `parego.py` | `ParEGO` | Random Chebyshev scalarization + analytic Expected Improvement. A new weight vector is sampled each iteration. |
| `mesmo.py` | `MESMO` | Max-value Entropy Search for Multi-objective BO. Estimates information gain relative to sampled Pareto sets. |
| `botorch_wrappers.py` | `BoTorchAcquisition` | Adapts the tutorial's GPyTorch models to BoTorch's interface, exposing `qLogNEHVI`, `qLogEHVI`, `qMOPES`, and `qLBMOMES`. |

### `benchmark.py` — Benchmarking driver

`run_benchmark()` runs a full grid of test functions × acquisition functions × random seeds, collects hypervolume traces and timing, and saves results to CSV. `plot_results()` produces mean ± std hypervolume curves.
