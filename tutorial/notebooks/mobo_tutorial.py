import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _intro():
    import marimo as mo
    mo.output.append(mo.md(r"""
    # Multi-objective Bayesian Optimization Tutorial

    This notebook walks through the main components of a MOBO pipeline:

    1. **Test functions** – multi-objective black-box problems.
    2. **GP surrogate** – one independent GP per objective fitted with GPyTorch.
    3. **Acquisition functions** – ParEGO, EHVI, MESMO, implemented from scratch.
    4. **MOBO loop** – tie everything together and track hypervolume.
    5. **Visualisation** – Pareto fronts and hypervolume traces.

    All source code lives in `../src/`. Make sure your kernel uses the `uv` environment:

    ```bash
    cd tutorial
    uv run marimo edit notebooks/mobo_tutorial.py
    ```
    """))
    return (mo,)


@app.cell
def _setup():
    import sys, pathlib

    # Add src/ to the path so imports work regardless of the working directory.
    src = (pathlib.Path(__file__).parent.parent / "src").resolve()
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    import torch
    import matplotlib.pyplot as plt

    print("torch", torch.__version__)
    print("src path:", src)
    return plt, torch


@app.cell
def _controls(mo):
    fn_dropdown = mo.ui.dropdown(
        options=["BraninCurrin", "ZDT1", "DTLZ2"],
        value="BraninCurrin",
        label="Test function",
    )
    acq_dropdown = mo.ui.dropdown(
        options=["EHVI", "ParEGO", "MESMO"],
        value="EHVI",
        label="Acquisition function",
    )
    run_btn = mo.ui.run_button(label="Run MOBO")
    mo.output.append(mo.hstack([fn_dropdown, acq_dropdown, run_btn], gap=1.0))
    return acq_dropdown, fn_dropdown, run_btn


@app.cell
def _sec1():
    return


@app.cell
def _test_fns(torch):
    from test_functions import get_test_function as _get_test_function, REGISTRY as TEST_FN_REGISTRY

    print("Available test functions:", list(TEST_FN_REGISTRY.keys()))

    # Instantiate and inspect DTLZ2
    fn = _get_test_function("DTLZ2")
    print(fn)
    print("Search space bounds:\n", fn.search_space.bounds)
    print("Reference point:", fn.ref_point)

    # Quick sanity-check: evaluate 5 random points
    x_test = torch.rand(5, fn.d)
    y_test = fn(x_test)
    print("\nRandom evaluations (shape):", y_test.shape)
    print(y_test)
    return


@app.cell
def _plot_test_fn(fn_dropdown, mo, plt, torch):
    from test_functions import get_test_function as _get_tf
    from hypervolume import _nondominated

    _fn_name = fn_dropdown.value
    _fn = _get_tf(_fn_name)
    _xs = torch.linspace(0, 1, 20)
    _grid_x1, _grid_x2 = torch.meshgrid(_xs, _xs, indexing="ij")

    # For d>2 functions (ZDT1, DTLZ2) fix extra dims at 0.5 — a 2D slice
    _X_grid = torch.full((20 * 20, _fn.d), 0.5)
    _X_grid[:, 0] = _grid_x1.flatten()
    _X_grid[:, 1] = _grid_x2.flatten()

    _Y_grid = _fn(_X_grid)   # (400, 2)
    _f1 = _Y_grid[:, 0].reshape(20, 20).numpy()
    _f2 = _Y_grid[:, 1].reshape(20, 20).numpy()
    _x1_np = _grid_x1.numpy()
    _x2_np = _grid_x2.numpy()

    _Y_pareto, _ = _nondominated(_Y_grid)
    _slice_note = "" if _fn.d == 2 else f" (x₃…x{_fn.d} = 0.5)"

    fig_tf, _axes = plt.subplots(1, 3, figsize=(13, 3.5))

    _cf1 = _axes[0].contourf(_x1_np, _x2_np, _f1, levels=15, cmap="RdYlGn")
    plt.colorbar(_cf1, ax=_axes[0])
    _axes[0].set_xlabel("x₁")
    _axes[0].set_ylabel("x₂")
    _axes[0].set_title(f"f₁(x₁, x₂){_slice_note}")

    _cf2 = _axes[1].contourf(_x1_np, _x2_np, _f2, levels=15, cmap="RdYlGn")
    plt.colorbar(_cf2, ax=_axes[1])
    _axes[1].set_xlabel("x₁")
    _axes[1].set_ylabel("x₂")
    _axes[1].set_title(f"f₂(x₁, x₂){_slice_note}")

    _sc = _axes[2].scatter(
        _Y_grid[:, 0].numpy(), _Y_grid[:, 1].numpy(),
        c=_grid_x1.flatten().numpy(), cmap="viridis", alpha=0.5, s=20,
    )
    plt.colorbar(_sc, ax=_axes[2], label="x₁")
    _axes[2].scatter(_Y_pareto[:, 0].numpy(), _Y_pareto[:, 1].numpy(),
                     color="#0f766e", s=40, zorder=5, label="Pareto front")
    _axes[2].set_xlabel("f₁")
    _axes[2].set_ylabel("f₂")
    _axes[2].set_title(f"Objective space ({_fn_name})")
    _axes[2].legend(fontsize=8)

    fig_tf.tight_layout()
    mo.output.append(fig_tf)
    return


@app.cell
def _sec2():
    return


@app.cell
def _gp(fn_dropdown, torch):
    from test_functions import get_test_function as _get_test_function
    from surrogate import fit_gp as _fit_gp, predict as _predict

    from gpytorch.priors import LogNormalPrior as _LogNormalPrior

    # Generate a small training set
    fn_gp = _get_test_function(fn_dropdown.value)
    torch.manual_seed(0)
    X_train = torch.rand(5, fn_gp.d)
    Y_train = fn_gp(X_train)

    # Standardise for GP stability
    Y_mean, Y_std = Y_train.mean(0), Y_train.std(0).clamp(min=1e-6)
    Y_norm = (Y_train - Y_mean) / Y_std

    # Fit one GP per objective; prior on lengthscale prevents degenerate optima
    _ls_prior = _LogNormalPrior(loc=0.0, scale=1.0)
    models = _fit_gp(X_train, Y_norm, n_iter=100, lengthscale_prior=_ls_prior)
    print(f"Fitted {len(models)} GPs")

    # Predict on a grid
    _x_grid = torch.rand(200, fn_gp.d)
    _mu_grid, _sig_grid = _predict(models, _x_grid)
    print("Predictive mean shape:", _mu_grid.shape)
    print("Predictive std  shape:", _sig_grid.shape)
    return X_train, Y_mean, Y_norm, Y_std, fn_gp, models


@app.cell
def _plot_gp(Y_mean, Y_std, fn_gp, mo, models, plt, torch):
    from surrogate import predict as _predict

    # 1D slice: vary x1 across [0,1], fix all other dims at 0.5
    _x0 = torch.linspace(0, 1, 100)
    _X_slice = torch.full((100, fn_gp.d), 0.5)
    _X_slice[:, 0] = _x0
    _mu, _sigma = _predict(models, _X_slice)      # (100, 2)
    _Y_true = fn_gp(_X_slice)                     # (100, 2)

    _slice_note = "" if fn_gp.d == 2 else f", x₂…x{fn_gp.d} = 0.5"
    _colors = ["#e66739", "#1d3557"]
    fig_gp, _axes = plt.subplots(1, 2, figsize=(11, 3.5))

    for _obj in range(2):
        _ax = _axes[_obj]
        _mu_raw = _mu[:, _obj].detach().numpy() * Y_std[_obj].item() + Y_mean[_obj].item()
        _sig_raw = _sigma[:, _obj].detach().numpy() * Y_std[_obj].item()
        _x0_np = _x0.numpy()

        _ax.plot(_x0_np, _Y_true[:, _obj].numpy(),
                 color=_colors[_obj], linewidth=1.5, label=f"true f{_obj+1}")
        _ax.plot(_x0_np, _mu_raw, "--", color=_colors[_obj], linewidth=2,
                 alpha=0.85, label="GP mean")
        _ax.fill_between(_x0_np, _mu_raw - 1.96 * _sig_raw, _mu_raw + 1.96 * _sig_raw,
                         alpha=0.2, color=_colors[_obj], label="95% CI")
        # Training points are NOT on this slice (they were observed at random x₂...x_d),
        # so showing them here would make the GP look like it doesn't fit the data.
        _ax.set_xlabel("x₁")
        _ax.set_title(f"Objective {_obj+1} — GP fit (x₁ varies{_slice_note})")
        _ax.legend(fontsize=8)

    fig_gp.tight_layout()
    mo.output.append(fig_gp)
    return


@app.cell
def _sec3():
    return


@app.cell
def _acq_demo(X_train, Y_mean, Y_norm, Y_std, fn_gp, models, torch):
    from acq import get_acquisition, REGISTRY as ACQ_REGISTRY
    from hypervolume import _nondominated, hypervolume

    print("Available acquisition functions:", list(ACQ_REGISTRY.keys()))

    # Example: evaluate each acquisition on random candidates
    pareto_y = _nondominated(Y_norm)[0]
    ref_norm = (fn_gp.ref_point - Y_mean) / Y_std
    bounds_norm = torch.stack([
        (torch.zeros(fn_gp.d) - X_train.mean(0)) / X_train.std(0).clamp(min=1e-6),
        (torch.ones(fn_gp.d)  - X_train.mean(0)) / X_train.std(0).clamp(min=1e-6),
    ])
    x_cands = torch.rand(8, fn_gp.d)

    for _acq_name in ACQ_REGISTRY:
        if _acq_name == "EHVI":
            acq = get_acquisition(_acq_name, models=models, pareto_y=pareto_y, ref_point=ref_norm)
        elif _acq_name == "MESMO":
            acq = get_acquisition(_acq_name, models=models, bounds=bounds_norm, n_grid=64)
        elif _acq_name == "ParEGO":
            acq = get_acquisition(_acq_name, models=models, train_y=Y_norm)
        vals = acq(x_cands)
        print(f"{_acq_name:8s}  values: {vals.detach().numpy().round(4)}")
    return (ACQ_REGISTRY,)


@app.cell
def _sec4():
    return


@app.cell
def _mobo_state(mo):
    get_mobo_result, set_mobo_result = mo.state(None)
    return get_mobo_result, set_mobo_result


@app.cell
def _mobo_run(acq_dropdown, fn_dropdown, mo, run_btn, set_mobo_result, torch):
    from test_functions import get_test_function as _get_test_function
    from mobo import run_mobo as _run_mobo

    mo.stop(not run_btn.value, mo.md("Click **Run MOBO** to execute the loop."))

    _TEST_FUNCTION = fn_dropdown.value
    _ACQUISITION   = acq_dropdown.value
    _N_INIT = 5
    _N_ITER = 15
    _SEED   = 42

    torch.manual_seed(_SEED)
    _fn_mobo = _get_test_function(_TEST_FUNCTION)

    _result = _run_mobo(
        test_fn=_fn_mobo,
        acq_name=_ACQUISITION,
        n_init=_N_INIT,
        n_iter=_N_ITER,
        n_restarts=10,
        n_raw=256,
        gp_n_iter=80,
        seed=_SEED,
        verbose=True,
    )
    print(f"\nFinal hypervolume: {_result.final_hv:.4f}")
    set_mobo_result({
        "result": _result,
        "fn_mobo": _fn_mobo,
        "TEST_FUNCTION": _TEST_FUNCTION,
        "ACQUISITION": _ACQUISITION,
        "N_INIT": _N_INIT,
    })
    return


@app.cell
def _sec5a():
    return


@app.cell
def _plot_hv(get_mobo_result, mo, plt):
    _mobo = get_mobo_result()
    mo.stop(_mobo is None, mo.md("Run MOBO above to see this plot."))
    _result = _mobo["result"]
    _N_INIT = _mobo["N_INIT"]
    _TEST_FUNCTION = _mobo["TEST_FUNCTION"]
    _ACQUISITION = _mobo["ACQUISITION"]
    fig_hv, _ax = plt.subplots(figsize=(7, 3.5))
    _ax.plot(_result.hv_trace, marker="o", markersize=4, linewidth=1.8, color="#0f766e")
    _ax.axvline(_N_INIT, color="grey", linestyle=":", linewidth=1.2, label=f"end of init (n={_N_INIT})")
    _ax.set_xlabel("Evaluation index")
    _ax.set_ylabel("Hypervolume")
    _ax.set_title(f"{_TEST_FUNCTION} + {_ACQUISITION}")
    _ax.legend()
    fig_hv.tight_layout()
    mo.output.append(fig_hv)
    return


@app.cell
def _sec5b():
    return


@app.cell
def _plot_pareto(get_mobo_result, mo, plt):
    from hypervolume import _nondominated
    _mobo = get_mobo_result()
    mo.stop(_mobo is None, mo.md("Run MOBO above to see this plot."))
    _result = _mobo["result"]
    _fn_mobo = _mobo["fn_mobo"]
    _TEST_FUNCTION = _mobo["TEST_FUNCTION"]
    _ACQUISITION = _mobo["ACQUISITION"]

    _Y_all = _result.Y.detach()
    _Y_pareto = _nondominated(_Y_all)[0].detach()

    # Sort Pareto front for step plot
    _order = _Y_pareto[:, 0].argsort()
    _Y_pareto_sorted = _Y_pareto[_order]

    fig_pareto, _ax = plt.subplots(figsize=(6, 5))
    _ax.scatter(_Y_all[:, 0], _Y_all[:, 1], c="steelblue", alpha=0.4, s=25, label="all evaluations")
    _ax.scatter(_Y_pareto[:, 0], _Y_pareto[:, 1], c="#0f766e", s=60, zorder=5, label="Pareto front")
    _ax.scatter(_fn_mobo.ref_point[0], _fn_mobo.ref_point[1], c="#241a74", s=60, zorder=5, label="ref point")

    _ax.step(
        _Y_pareto_sorted[:, 0].numpy(),
        _Y_pareto_sorted[:, 1].numpy(),
        where="post", color="#0f766e", linewidth=1.4, alpha=0.6,
    )
    _ax.set_xlabel("Objective 1 (↑)")
    _ax.set_ylabel("Objective 2 (↑)")
    _ax.set_title(f"Pareto front – {_TEST_FUNCTION} + {_ACQUISITION}")
    _ax.legend()
    fig_pareto.tight_layout()
    mo.output.append(fig_pareto)
    return


@app.cell
def _sec6():
    return


@app.cell
def _bench_state(mo):
    get_bench_result, set_bench_result = mo.state(None)
    return get_bench_result, set_bench_result


@app.cell
def _benchmark_run(ACQ_REGISTRY, mo, set_bench_result):
    import time
    from test_functions import get_test_function as _get_test_function
    from mobo import run_mobo as _run_mobo

    # ── Benchmark configuration ───────────────────────────────────────────────────
    _RUN_BENCHMARK = False   # set to True to execute (takes several minutes)
    _N_BENCH_INIT  = 5
    _N_BENCH_ITER  = 20    # increase for a proper comparison
    _N_SEEDS       = 5    # increase for smoother curves
    _BENCH_FNS     = ["BraninCurrin"]          # add "ZDT1", "DTLZ2" as desired
    _BENCH_ACQS    = list(ACQ_REGISTRY.keys())
    # ─────────────────────────────────────────────────────────────────────────────

    mo.stop(not _RUN_BENCHMARK, mo.md(
        "> **Benchmark disabled.** Set `_RUN_BENCHMARK = True` in this cell to run "
        "(expect several minutes for the default configuration)."
    ))

    _results_bench = {}   # (fn_name, acq_name, seed) -> MOBOResult

    for _fn_name in _BENCH_FNS:
        for _acq_name in _BENCH_ACQS:
            for _seed in range(_N_SEEDS):
                _key = (_fn_name, _acq_name, _seed)
                print(f"  {_fn_name} + {_acq_name} seed={_seed} ...", end=" ", flush=True)
                _t0 = time.perf_counter()
                _results_bench[_key] = _run_mobo(
                    test_fn=_get_test_function(_fn_name),
                    acq_name=_acq_name,
                    n_init=_N_BENCH_INIT,
                    n_iter=_N_BENCH_ITER,
                    n_restarts=8,
                    n_raw=128,
                    gp_n_iter=50,
                    seed=_seed,
                    verbose=False,
                )
                print(f"done ({time.perf_counter()-_t0:.1f}s)")
    set_bench_result({
        "results_bench": _results_bench,
        "BENCH_FNS": _BENCH_FNS,
        "BENCH_ACQS": _BENCH_ACQS,
        "N_SEEDS": _N_SEEDS,
        "N_BENCH_INIT": _N_BENCH_INIT,
    })
    return


@app.cell
def _plot_bench(get_bench_result, mo, plt, torch):
    _bench = get_bench_result()
    mo.stop(_bench is None, mo.md("Run the benchmark above to see this plot."))
    _results_bench = _bench["results_bench"]
    _BENCH_FNS = _bench["BENCH_FNS"]
    _BENCH_ACQS = _bench["BENCH_ACQS"]
    _N_SEEDS = _bench["N_SEEDS"]
    _N_BENCH_INIT = _bench["N_BENCH_INIT"]

    colors = {"EHVI": "#0f766e", "ParEGO": "#1d4ed8", "MESMO": "#b45309"}
    _n_fns = len(_BENCH_FNS)
    fig_bench, _axes = plt.subplots(1, _n_fns, figsize=(5.5 * _n_fns, 4), squeeze=False)

    for _col, _fn_name in enumerate(_BENCH_FNS):
        _ax = _axes[0][_col]
        for _acq_name in _BENCH_ACQS:
            _traces = [_results_bench[(_fn_name, _acq_name, s)].hv_trace for s in range(_N_SEEDS)]
            _traces_t = torch.tensor(_traces, dtype=torch.float32)
            _mean = _traces_t.mean(0).numpy()
            _std  = _traces_t.std(0).numpy()
            _xs = range(len(_mean))
            _c = colors.get(_acq_name, "grey")
            _ax.plot(_xs, _mean, label=_acq_name, color=_c, linewidth=2)
            _ax.fill_between(_xs, _mean - _std, _mean + _std, alpha=0.15, color=_c)
        _ax.axvline(_N_BENCH_INIT, color="grey", linestyle=":", linewidth=1.1, label="end of init")
        _ax.set_title(_fn_name)
        _ax.set_xlabel("Evaluation index")
        _ax.set_ylabel("Hypervolume")
        _ax.legend(fontsize=9)

    fig_bench.suptitle("Benchmark – hypervolume vs. evaluations", y=1.02)
    fig_bench.tight_layout()
    mo.output.append(fig_bench)

    # Wall-clock summary
    print("\nMean iteration time (seconds):")
    print(f"{'Acquisition':10s}", end="")
    for _fn_name in _BENCH_FNS:
        print(f"  {_fn_name:>15s}", end="")
    print()
    for _acq_name in _BENCH_ACQS:
        print(f"{_acq_name:10s}", end="")
        for _fn_name in _BENCH_FNS:
            _all_times = []
            for _s in range(_N_SEEDS):
                _all_times.extend(_results_bench[(_fn_name, _acq_name, _s)].iter_times)
            _mean_t = sum(_all_times) / len(_all_times)
            print(f"  {_mean_t:>15.2f}", end="")
        print()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
