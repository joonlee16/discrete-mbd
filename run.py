"""Run the solvers on random unconstrained benchmark problems.

Each trial samples a new problem instance, runs every selected method on it,
and the average cost and runtime over all trials are printed per method.

Examples
--------
    python run.py
    python run.py --problem UPO --N 50 --K 10 --methods MBD CEM SA --trials 10
"""

import argparse
import time
from dataclasses import dataclass

import jax
import numpy as np

from cost_function import uiqp, upo
from solver.cem import cem
from solver.lns import lns
from solver.sa import sa
from solver.smc import smc
from solver.rounded_mbd import rounded_mbd
from solver.discrete_mbd import discrete_mbd

PROBLEMS = {"UIQP": uiqp, "UPO": upo}

SOLVERS = {
    "MBD": discrete_mbd,
    "Rounded-MBD": rounded_mbd,
    "CEM": cem,
    "SA": sa,
    "SMC": smc,
    "LNS": lns,
}

# "Gurobi" can also be selected as a reference solver; it needs a Gurobi license.
METHODS = list(SOLVERS) + ["Gurobi"]


@dataclass(unsafe_hash=True)
class Args:
    """Problem size and solver hyperparameters. Hashable because it is a static argument of ``jax.jit``."""

    N: int = 10  # number of variables
    K: int = 10  # number of values per variable
    Ndiffuse: int = 100  # number of iterations (diffusion steps for MBD)
    Nsample: int = 2048  # samples per iteration
    temp_sample: float = 1e-5  # MBD: temperature of the cost weights
    sample_ratio: float = 0.75  # Discrete MBD: fraction of samples drawn in the first stage
    elite_frac: float = 0.01  # CEM: fraction of samples kept as elites
    T0: float = 1.0  # SA, SMC: initial temperature
    alpha: float = 0.95  # SA, SMC: geometric cooling rate
    destroy_k_ratio: float = 0.1  # SMC, LNS: fraction of variables changed per move


def run_method(name, args, problem, cost_fn, data, seed):
    """Run one method on one instance and return ``(cost, runtime in seconds)``."""
    if name == "Gurobi":
        start = time.perf_counter()
        _, cost = problem.actual_optimal(args, *data)
        return cost, time.perf_counter() - start

    solver = SOLVERS[name]
    # The first call compiles the solver, so only the second call is timed.
    solver(args, cost_fn, seed=seed).block_until_ready()
    start = time.perf_counter()
    history = solver(args, cost_fn, seed=seed).block_until_ready()
    runtime = time.perf_counter() - start
    return float(cost_fn(history[-1])), runtime


def run(problem="UIQP", N=10, K=10, methods=tuple(SOLVERS), num_trials=5):
    """Run ``methods`` on ``num_trials`` random instances and print the averages.

    Parameters
    ----------
    problem : {"UIQP", "UPO"}, optional
        Benchmark problem. Default is "UIQP".
    N : int, optional
        Number of variables. Default is 10.
    K : int, optional
        Number of values each variable can take. Default is 10.
    methods : sequence of str, optional
        Names from ``METHODS``. Default is every solver except Gurobi.
    num_trials : int, optional
        Number of random instances. Default is 5.

    Returns
    -------
    dict
        Maps each method name to a list of ``(cost, runtime)`` pairs, one per trial.
    """
    args = Args(N=N, K=K)
    problem_module = PROBLEMS[problem]
    if "Gurobi" in methods:
        import gurobipy

        gurobipy.setParam("OutputFlag", 0)

    results = {name: [] for name in methods}
    for trial in range(num_trials):
        cost_fn, *data = problem_module.build_cost(args, seed=trial)
        for name in methods:
            results[name].append(run_method(name, args, problem_module, cost_fn, data, seed=trial))
        jax.clear_caches()

    print(f"{problem} with N = {N}, K = {K}, averaged over {num_trials} trials")
    for name, runs in results.items():
        costs = np.array([cost for cost, _ in runs if cost is not None])
        runtimes = np.array([runtime for _, runtime in runs])
        mean_cost = f"{costs.mean():.4e}" if len(costs) > 0 else "no solution"
        print(f"  {name:<12} cost {mean_cost:>12}   runtime {runtimes.mean():8.3f} s")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problem", choices=list(PROBLEMS), default="UIQP")
    parser.add_argument("--N", type=int, default=10, help="number of variables")
    parser.add_argument("--K", type=int, default=10, help="number of values per variable")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(SOLVERS))
    parser.add_argument("--trials", type=int, default=5, help="number of random instances")
    cli = parser.parse_args()
    run(cli.problem, cli.N, cli.K, cli.methods, cli.trials)


if __name__ == "__main__":
    main()
