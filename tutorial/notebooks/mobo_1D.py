import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _intro():
    import marimo as mo
    mo.md(r"""
    # Multi-objective Bayesian Optimization Tutorial 1: Acquisition functions

    All source code lives in `../src/`. Make sure your kernel uses the `uv` environment:

    ```bash
    cd tutorial
    uv run marimo edit notebooks/mobo_1D.py
    ```
    """)
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
        options=["Sinusoidal"],
        value="Sinusoidal",
        label="Test function",
    )
    acq_multiselect = mo.ui.multiselect(
        options=["EHVI", "MESMO", "ParEGO"],
        value=["EHVI", "MESMO", "ParEGO"],
        label="Acquisition functions",
    )
    mo.output.append(mo.vstack([fn_dropdown, acq_multiselect]))
    return acq_multiselect, fn_dropdown


@app.cell
def _section1(mo):
    mo.md(r"""
    ## 1. D=1, M=2 objective functions
    """)
    return


@app.cell
def _plot_objectives(fn_dropdown, mo, plt, torch):
    from hypervolume import _nondominated
    from test_functions import get_test_function

    fn = get_test_function(fn_dropdown.value)

    bounds = fn.search_space.bounds  # shape: (2, d)
    x = torch.linspace(bounds[0, 0], bounds[1, 0], 400).unsqueeze(1)
    y = fn(x)

    fig_obj, _axes = plt.subplots(1, 2, figsize=(11, 3))
    base_colors = ["#e66739", "#1d3557"]

    # Non-dominated points
    y_pareto, _nondom_idx = _nondominated(y)

    # Objectives vs input
    _axes[0].plot(x[:, 0].numpy(), y[:, 0].numpy(), label="f1(x)", linewidth=2, color=base_colors[0])
    _axes[0].scatter(x[_nondom_idx, 0].numpy(), y[_nondom_idx, 0].numpy(), color=base_colors[0])
    _axes[0].plot(x[:, 0].numpy(), y[:, 1].numpy(), label="f2(x)", linewidth=2, color=base_colors[1])
    _axes[0].scatter(x[_nondom_idx, 0].numpy(), y[_nondom_idx, 1].numpy(), color=base_colors[1])
    _axes[0].set_xlabel("x")
    _axes[0].set_ylabel("objective value")
    _axes[0].set_title("Objectives vs input")
    _axes[0].legend()

    # Objective-space curve
    _axes[1].plot(y[:, 0].numpy(), y[:, 1].numpy(), color="#50bbb2", linewidth=2)
    _axes[1].scatter(y_pareto[:, 0].numpy(), y_pareto[:, 1].numpy(), color="#0f766e")
    _axes[1].set_xlabel("f1(x)")
    _axes[1].set_ylabel("f2(x)")
    _axes[1].set_title("Objective space")

    fig_obj.tight_layout()
    mo.output.append(fig_obj)
    return base_colors, fn, x, y


@app.cell
def _section2(mo):
    mo.md(r"""
    ## 2. Acquisition functions
    """)
    return


@app.cell
def _plot_acq(acq_multiselect, base_colors, fn, mo, plt, torch, x, y):
    from hypervolume import _nondominated
    from surrogate import fit_gp, predict, LogNormalPrior
    from acq import get_acquisition

    # Collect observations
    X_obs = torch.tensor([-1.5, -1.0, 0.5, 3.3]).unsqueeze(1)
    Y_obs = fn(X_obs)

    # Standardise observations for GP
    Y_obs_mean, Y_obs_std = Y_obs.mean(0), Y_obs.std(0).clamp(min=1e-6)
    Y_obs_norm = (Y_obs - Y_obs_mean) / Y_obs_std

    # Train GPs on observations
    gp_models = fit_gp(X_obs, Y_obs_norm, n_iter=100, min_noise=1e-6, fix_noise=True, lengthscale_prior=LogNormalPrior(0.0, 0.5))

    # Make predictions on the grid
    mu_grid, sigma_grid = predict(gp_models, x)

    # Prepare for acquisition function evaluation
    pareto_y_obs, _nondom_idx = _nondominated(Y_obs_norm)
    ref_norm_obs = (fn.ref_point - Y_obs_mean) / Y_obs_std

    # Create figure with 2 subplots
    fig_acq, _axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

    # Subplot 1: Objectives with GP surrogates
    for obj_idx in range(2):
        mu_obj = mu_grid[:, obj_idx].detach().numpy() * Y_obs_std[obj_idx].item() + Y_obs_mean[obj_idx].item()
        sigma_obj = sigma_grid[:, obj_idx].detach().numpy()
        y_obj = y[:, obj_idx].numpy()

        _axes[0].plot(x[:, 0].numpy(), y_obj, label=f"f{obj_idx+1}(x)", linewidth=1, color=base_colors[obj_idx])
        _axes[0].plot(x[:, 0].numpy(), mu_obj, "--", linewidth=2, alpha=0.8, color=base_colors[obj_idx])
        _axes[0].fill_between(
            x[:, 0].numpy(),
            mu_obj - 1.96 * sigma_obj,
            mu_obj + 1.96 * sigma_obj,
            alpha=0.2, color=base_colors[obj_idx]
        )
        _axes[0].scatter(X_obs[:, 0].numpy(), Y_obs[:, obj_idx].numpy(), s=100, zorder=5, marker='.', color=base_colors[obj_idx])

    _axes[0].set_ylabel("objective value")
    _axes[0].legend()

    # Subplot 2: Acquisition functions
    x_cands_acq = torch.linspace(fn.search_space.bounds[0, 0], fn.search_space.bounds[1, 0], 150).unsqueeze(1)

    acq_names = acq_multiselect.value
    acq_colors = {acq_name: color for acq_name, color in zip(acq_names, ["#92b82b", "#420f55", "#50bbb2"])}

    for i, acq_name in enumerate(acq_names):
        if i == 0:
            ax = _axes[1]
            _axes[1].set_xlabel("x")
        else:
            ax = _axes[1].twinx()

        c = acq_colors.get(acq_name, "black")
        ax.spines["right"].set_color(c)
        ax.tick_params(axis="y", colors=c, labelleft=False, labelright=False)
        ax.tick_params(axis="y", labelleft=False, labelright=False)

        if acq_name == "EHVI":
            acq = get_acquisition(acq_name, models=gp_models, pareto_y=pareto_y_obs, ref_point=ref_norm_obs)
        elif acq_name == "MESMO":
            bounds_norm_obs = torch.stack([
                (torch.zeros(fn.d) - X_obs.mean(0)) / X_obs.std(0).clamp(min=1e-6),
                (torch.ones(fn.d)  - X_obs.mean(0)) / X_obs.std(0).clamp(min=1e-6),
            ])
            acq = get_acquisition(acq_name, models=gp_models, bounds=bounds_norm_obs, n_grid=32)
        elif acq_name == "ParEGO":
            acq = get_acquisition(acq_name, models=gp_models, train_y=Y_obs_norm)

        acq_vals = acq(x_cands_acq).detach().numpy()
        acq_max_idx = acq_vals.argmax()

        ax.plot(x_cands_acq[:, 0].numpy(), acq_vals, label=acq_name, linewidth=2, color=acq_colors[acq_name])
        ax.axvline(x=x_cands_acq[acq_max_idx, 0].item(), color=acq_colors[acq_name], linestyle="--", alpha=0.8)
        _axes[0].axvline(x=x_cands_acq[acq_max_idx, 0].item(), color=acq_colors[acq_name], linestyle="--", alpha=0.8)

    handles, labels = [], []
    for a in fig_acq.axes:
        h, l = a.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll in acq_names and ll not in labels:
                handles.append(hh)
                labels.append(ll)
    _axes[1].legend(handles, labels, loc="upper right")

    fig_acq.tight_layout()
    mo.output.append(fig_acq)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
