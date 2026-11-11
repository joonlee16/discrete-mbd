"""Unconstrained Integer Quadratic Program (UIQP).

Solves

    min_{x in {0, 1, ..., K-1}^N}  x^T Q x + c^T x,

where Q is a random symmetric Gaussian matrix and c is a random Gaussian vector.
The fields read from ``args`` are ``N`` (number of variables n) and ``K``
(number of values each variable can take).
"""

import jax
import jax.numpy as jnp
import numpy as np


def build_cost(args, seed=0):
    """Sample a random UIQP instance and return its JIT-compiled cost function.

    Parameters
    ----------
    args : object
        Configuration with attribute ``N`` (number of variables).
    seed : int, optional
        Seed for the JAX PRNG that generates ``Q`` and ``c``. Default is 0.

    Returns
    -------
    cost_fn : callable
        JIT-compiled function mapping ``x`` to the scalar cost
        ``x^T Q x + c^T x``. ``x`` is either an integer vector of shape
        ``(N,)`` or a one-hot array of shape ``(N, K)``.
    Q : jax.Array, shape (N, N)
        Symmetric quadratic coefficient matrix.
    c : jax.Array, shape (N,)
        Linear coefficient vector.
    """
    n = args.N
    Q_key, c_key = jax.random.split(jax.random.PRNGKey(seed))

    Q = jax.random.normal(Q_key, (n, n))
    Q = (Q + Q.T) / 2.0  # symmetrize
    c = jax.random.normal(c_key, (n,))

    @jax.jit
    def cost_fn(x):
        # A one-hot input of shape (n, K) is converted to integer values in {0, ..., K-1}^n.
        if x.ndim == 2:
            x = jnp.argmax(x, axis=-1)
        quadratic = jnp.dot(x, jnp.dot(Q, x))
        linear = jnp.dot(c, x)
        return quadratic + linear

    return cost_fn, Q, c


def actual_optimal(args, Q, c, seed=None, time_limit=30, mip_gap=None, return_info=False):
    """Solve the UIQP instance with Gurobi as a reference solution.

    The problem is solved as a nonconvex integer quadratic program
    (``NonConvex = 2``). Gurobi stops at the time limit or once the
    relative MIP gap falls below ``mip_gap``, so the result is not
    guaranteed to be globally optimal; check ``info["optimal"]``.

    Parameters
    ----------
    args : object
        Configuration with attributes ``N`` (number of variables) and
        ``K`` (number of values per variable).
    Q : array_like, shape (N, N)
        Quadratic coefficient matrix returned by :func:`build_cost`.
    c : array_like, shape (N,)
        Linear coefficient vector returned by :func:`build_cost`.
    seed : int or None, optional
        If given, sets Gurobi's random seed and a random MIP start
        drawn from ``{0, ..., K-1}^N``. Default is None.
    time_limit : float or None, optional
        Gurobi ``TimeLimit`` in seconds. None means no limit. Default is 30.
    mip_gap : float or None, optional
        Gurobi ``MIPGap``. None keeps Gurobi's default of 1e-4, and 0.0
        asks Gurobi to prove exact optimality. Default is None.
    return_info : bool, optional
        If True, also return a dictionary of solver statistics. Default is False.

    Returns
    -------
    x_sol : list of int or None
        Best assignment found, or None if Gurobi found no feasible solution.
    obj_val : float or None
        Gurobi's objective value ``ObjVal`` for ``x_sol``, or None.
    info : dict
        Only returned if ``return_info`` is True. Keys are ``"runtime"``
        (Gurobi solve time in seconds, excluding model building),
        ``"status"`` (Gurobi status code), ``"optimal"`` (whether optimality
        was proven) and ``"gap"`` (final MIP gap, or inf if no solution).
    """
    # Imported here so that build_cost can be used without a Gurobi installation.
    import gurobipy as gp
    from gurobipy import GRB

    n = args.N
    K = args.K

    model = gp.Model("uiqp")
    x = model.addVars(n, vtype=GRB.INTEGER, lb=0, ub=K - 1, name="x")

    quadratic = gp.quicksum(float(Q[i, j]) * x[i] * x[j] for i in range(n) for j in range(n))
    linear = gp.quicksum(float(c[i]) * x[i] for i in range(n))
    model.setObjective(quadratic + linear, GRB.MINIMIZE)

    # Q is indefinite in general, so Gurobi must be told to accept a nonconvex objective.
    model.Params.NonConvex = 2
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap

    # Make the solve depend on the seed through Gurobi's seed and a random MIP start.
    if seed is not None:
        model.Params.Seed = seed
        x0 = np.random.default_rng(seed).integers(0, K, size=n)
        for i in range(n):
            x[i].Start = int(x0[i])

    model.optimize()

    info = {
        "runtime": model.Runtime,
        "status": model.Status,
        "optimal": model.Status == GRB.OPTIMAL,
        "gap": model.MIPGap if model.SolCount > 0 else float("inf"),
    }

    if model.SolCount > 0:
        x_sol = [round(x[i].X) for i in range(n)]
        return (x_sol, model.ObjVal, info) if return_info else (x_sol, model.ObjVal)

    return (None, None, info) if return_info else (None, None)
