"""Unconstrained Polynomial Optimization (UPO).

Solves

    min_{x in {0, 1, ..., K-1}^N}  f(x),    f(x) = sum_k c_k prod_{j in P_k} x_j,

where each of the ``NUM_TERMS`` monomials uses ``NUM_DEGREES`` distinct
variables chosen at random, and the coefficients c_k are standard Gaussian.
The fields read from ``args`` are ``N`` (number of variables n) and ``K``
(number of values each variable can take).
"""

import jax
import jax.numpy as jnp
import numpy as np

NUM_TERMS = 250  # number of monomials in f
NUM_DEGREES = 5  # number of distinct variables in each monomial


def build_cost(args, seed=0):
    """Sample a random UPO instance and return its JIT-compiled cost function.

    Parameters
    ----------
    args : object
        Configuration with attribute ``N`` (number of variables). ``N``
        must be at least ``NUM_DEGREES``.
    seed : int, optional
        Seed for the JAX PRNG that generates the instance. Default is 0.

    Returns
    -------
    cost_fn : callable
        JIT-compiled function mapping ``x`` to the scalar cost ``f(x)``.
        ``x`` is either an integer vector of shape ``(N,)`` or a one-hot
        array of shape ``(N, K)``.
    coeffs : jax.Array, shape (NUM_TERMS,)
        Coefficient c_k of each monomial.
    indices : jax.Array, shape (NUM_TERMS, NUM_DEGREES)
        Variable indices P_k of each monomial.
    degrees : jax.Array, shape (NUM_TERMS,)
        Degree of each monomial; only the first ``degrees[k]`` entries of
        ``indices[k]`` are used. Currently every entry equals ``NUM_DEGREES``.
    """
    n = args.N
    if n < NUM_DEGREES:
        raise ValueError(f"UPO needs N >= NUM_DEGREES ({NUM_DEGREES}), got N = {n}.")

    key = jax.random.PRNGKey(seed)
    coeff_key, index_key = jax.random.split(key, 2)

    coeffs = jax.random.normal(coeff_key, (NUM_TERMS,))
    degrees = jnp.full((NUM_TERMS,), NUM_DEGREES)

    # Each monomial uses NUM_DEGREES distinct variables, drawn without replacement.
    indices = jax.vmap(lambda key: jax.random.permutation(key, n)[:NUM_DEGREES])(
        jax.random.split(index_key, NUM_TERMS)
    )

    # mask[k, p] is True if position p of monomial k is within its degree.
    positions = jnp.arange(NUM_DEGREES)
    mask = positions[None, :] < degrees[:, None]

    @jax.jit
    def cost_fn(x):
        # A one-hot input of shape (n, K) is converted to integer values in {0, ..., K-1}^n.
        if x.ndim == 2:
            x = jnp.argmax(x, axis=-1)
        selected = x[indices].astype(jnp.float32)
        selected = jnp.where(mask, selected, 1)
        monomials = jnp.prod(selected, axis=1)
        return jnp.sum(coeffs * monomials)

    return cost_fn, coeffs, indices, degrees


def actual_optimal(args, coeffs, indices, degrees, seed=None, time_limit=30, return_info=False):
    """Solve the UPO instance with Gurobi as a reference solution.

    Each monomial is written as a chain of bilinear constraints on
    ``y = x / (K-1)`` in [0, 1], which keeps the auxiliary variables bounded
    and numerically well scaled. Gurobi stops at the time limit or at its
    default relative MIP gap of 1e-4, so check ``info["optimal"]``.

    Parameters
    ----------
    args : object
        Configuration with attributes ``N`` (number of variables) and
        ``K`` (number of values per variable).
    coeffs : array_like, shape (NUM_TERMS,)
        Monomial coefficients returned by :func:`build_cost`.
    indices : array_like, shape (NUM_TERMS, NUM_DEGREES)
        Monomial variable indices returned by :func:`build_cost`.
    degrees : array_like, shape (NUM_TERMS,)
        Monomial degrees returned by :func:`build_cost`.
    seed : int or None, optional
        If given, sets Gurobi's random seed and a random MIP start
        drawn from ``{0, ..., K-1}^N``. Default is None.
    time_limit : float or None, optional
        Gurobi ``TimeLimit`` in seconds. None means no limit. Default is 30.
    return_info : bool, optional
        If True, also return a dictionary of solver statistics. Default is False.

    Returns
    -------
    x_sol : list of int or None
        Best assignment found, or None if Gurobi found no feasible solution.
    obj_val : float or None
        Gurobi's objective value ``ObjVal``, rescaled to the units of f(x).
        None if no solution was found.
    info : dict
        Only returned if ``return_info`` is True. Keys are ``"runtime"``
        (Gurobi solve time in seconds, excluding model building),
        ``"status"`` (Gurobi status code), ``"optimal"`` (whether optimality
        was proven) and ``"gap"`` (final MIP gap, or inf if no solution).
    """
    # Imported here so that build_cost can be used without a Gurobi installation.
    import gurobipy as gp
    from gurobipy import GRB

    n, K = args.N, args.K
    coeffs = np.asarray(coeffs, dtype=np.float64)
    indices = np.asarray(indices)
    degrees = np.asarray(degrees)
    dmax = int(degrees.max())

    model = gp.Model("upo")
    x = model.addVars(n, vtype=GRB.INTEGER, lb=0, ub=K - 1, name="x")
    y = model.addVars(n, lb=0.0, ub=1.0, name="y")
    model.addConstrs((y[i] * (K - 1) == x[i] for i in range(n)), name="scale")

    terms = []
    for k in range(len(coeffs)):
        d = int(degrees[k])
        P = [int(j) for j in indices[k, :d]]
        # z_{k,j} = y_{P_0} * ... * y_{P_j}, built one bilinear product at a time.
        z = y[P[0]]
        for j in range(1, d):
            z_new = model.addVar(lb=0.0, ub=1.0, name=f"z_{k}_{j}")
            model.addQConstr(z_new == z * y[P[j]], name=f"prod_{k}_{j}")
            z = z_new
        # c_k * prod x = c_k * (K-1)^d * prod y. Every term is divided by (K-1)^dmax to keep the objective well scaled.
        terms.append(float(coeffs[k]) * float(K - 1) ** (d - dmax) * z)

    model.setObjective(gp.quicksum(terms), GRB.MINIMIZE)
    model.Params.NonConvex = 2
    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    # Make the solve depend on the seed through Gurobi's seed and a random MIP start.
    # Gurobi completes the start values of y and z.
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

    if model.SolCount == 0:
        return (None, None, info) if return_info else (None, None)

    x_sol = [int(round(x[i].X)) for i in range(n)]
    # Undo the (K-1)^dmax normalization so obj_val is in the units of f(x).
    obj_val = model.ObjVal * float(K - 1) ** dmax
    return (x_sol, obj_val, info) if return_info else (x_sol, obj_val)
