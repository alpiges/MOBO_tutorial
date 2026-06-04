"""Benchmark: run all acquisition functions on all test functions.

Usage
-----
    cd tutorial
    uv run python src/benchmark.py                 # defaults
    uv run python src/benchmark.py --n_iter 30 --n_seeds 5

Results are written to benchmark_results.csv and a summary plot to
benchmark_results.png.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import matplotlib.pyplot as plt

# Make src importable when run directly
sys.path.insert(0, str(Path(__file__).parent))

from acq import REGISTRY as ACQ_REGISTRY
from test_functions import REGISTRY as FN_REGISTRY
from mobo import run_mobo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MOBO benchmark")
    p.add_argument("--n_init", type=int, default=10, help="Initial design size")
    p.add_argument("--n_iter", type=int, default=20, help="BO iterations per run")
    p.add_argument("--n_seeds", type=int, default=3, help="Random seeds (replicates)")
    p.add_argument("--n_restarts", type=int, default=10, help="AF optimisation restarts")
    p.add_argument("--gp_n_iter", type=int, default=50, help="GP hyperparameter steps")
    p.add_argument("--out_csv", type=str, default="benchmark_results.csv")
    p.add_argument("--out_plot", type=str, default="benchmark_results.png")
    p.add_argument("--test_fns", nargs="+", default=list(FN_REGISTRY.keys()))
    p.add_argument("--acqs", nargs="+", default=list(ACQ_REGISTRY.keys()))
    return p.parse_args()


def run_benchmark(args: argparse.Namespace) -> list[dict]:
    records = []
    total = len(args.test_fns) * len(args.acqs) * args.n_seeds
    done = 0
    for fn_name in args.test_fns:
        for acq_name in args.acqs:
            for seed in range(args.n_seeds):
                done += 1
                print(f"[{done}/{total}] {fn_name} + {acq_name} seed={seed} ...", flush=True)
                test_fn = FN_REGISTRY[fn_name]()
                t_wall = time.perf_counter()
                try:
                    result = run_mobo(
                        test_fn=test_fn,
                        acq_name=acq_name,
                        n_init=args.n_init,
                        n_iter=args.n_iter,
                        n_restarts=args.n_restarts,
                        gp_n_iter=args.gp_n_iter,
                        seed=seed,
                        verbose=False,
                    )
                    total_wall = time.perf_counter() - t_wall
                    for it, hv in enumerate(result.hv_trace):
                        records.append({
                            "test_fn": fn_name,
                            "acq": acq_name,
                            "seed": seed,
                            "iteration": it,
                            "hypervolume": hv,
                            "iter_time": (
                                result.iter_times[it - args.n_init]
                                if it >= args.n_init and (it - args.n_init) < len(result.iter_times)
                                else float("nan")
                            ),
                            "total_wall_s": total_wall,
                        })
                except Exception as e:
                    print(f"  ERROR: {e}")
    return records


def save_csv(records: list[dict], path: str) -> None:
    if not records:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"Results saved to {path}")


def plot_results(records: list[dict], path: str, n_init: int) -> None:
    import collections
    # Group: test_fn -> acq -> list of hv_trace per seed
    data: dict[str, dict[str, list]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    max_iter: dict[str, int] = collections.defaultdict(int)
    for r in records:
        fn, acq, it, hv = r["test_fn"], r["acq"], r["iteration"], r["hypervolume"]
        data[fn][acq].append((it, hv))
        max_iter[fn] = max(max_iter[fn], it)

    fns = sorted(data.keys())
    n_fns = len(fns)
    fig, axes = plt.subplots(1, n_fns, figsize=(5 * n_fns, 4), squeeze=False)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    acq_names = sorted({r["acq"] for r in records})
    color_map = {a: colors[i % len(colors)] for i, a in enumerate(acq_names)}

    for col, fn in enumerate(fns):
        ax = axes[0][col]
        for acq in acq_names:
            pts = data[fn][acq]
            if not pts:
                continue
            # Aggregate over seeds
            from collections import defaultdict
            by_iter: dict[int, list[float]] = defaultdict(list)
            for it, hv in pts:
                by_iter[it].append(hv)
            iters = sorted(by_iter)
            means = [torch.tensor(by_iter[i]).mean().item() for i in iters]
            stds = [torch.tensor(by_iter[i]).std().item() for i in iters]
            ax.plot(iters, means, label=acq, color=color_map[acq])
            ax.fill_between(
                iters,
                [m - s for m, s in zip(means, stds)],
                [m + s for m, s in zip(means, stds)],
                alpha=0.15,
                color=color_map[acq],
            )
        ax.axvline(n_init, color="grey", linestyle=":", linewidth=1, label="end of init")
        ax.set_title(fn)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Hypervolume")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Plot saved to {path}")


if __name__ == "__main__":
    args = parse_args()
    records = run_benchmark(args)
    save_csv(records, args.out_csv)
    plot_results(records, args.out_plot, args.n_init)
