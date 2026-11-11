"""Simulated Annealing (SA) with parallel chains."""

from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=(0, 1))
def sa(args, cost_fn, seed=42):
    """Simulated Annealing for categorical assignment.

    Runs ``Nsample`` independent chains from uniformly random assignments. At
    each iteration every chain proposes to change one random variable to a
    different random value, and accepts it with the Metropolis rule at
    temperature ``T0 * alpha**t``.

    Parameters
    ----------
    args : object
        Hashable configuration with fields ``N``, ``K``, ``Nsample``,
        ``Ndiffuse`` (number of iterations), ``T0`` (initial temperature) and
        ``alpha`` (geometric cooling rate).
    cost_fn : callable
        Maps a one-hot array of shape ``(N, K)`` to a scalar cost.
    seed : int, optional
        Random seed. Default is 42.

    Returns
    -------
    Ybars : jax.Array, shape (Ndiffuse, N, K)
        Row ``t`` is the lowest-cost assignment found by any chain up to
        iteration ``t``.
    """
    rng = jax.random.PRNGKey(seed)

    lambdas = args.T0 * (args.alpha ** jnp.arange(args.Ndiffuse))

    rng, init_rng = jax.random.split(rng)
    a = jax.random.randint(init_rng, (args.Nsample, args.N), 0, args.K)
    Y0 = jax.nn.one_hot(a, args.K)
    cost = jax.vmap(cost_fn)(Y0)

    def sa_iteration(carry, lambda_i):
        rng, a, cost, best_a, best_cost = carry
        rng, var_rng, value_rng, acc_rng = jax.random.split(rng, 4)

        # Each chain shifts one random variable by 1..K-1, so the proposed value always differs.
        k = jax.random.randint(var_rng, (args.Nsample,), 0, args.N)
        shift = jax.random.randint(value_rng, (args.Nsample,), 1, args.K)
        cur = jnp.take_along_axis(a, k[:, None], axis=1)[:, 0]
        a_prop = a.at[jnp.arange(args.Nsample), k].set((cur + shift) % args.K)
        Y_prop = jax.nn.one_hot(a_prop, args.K)

        cost_prop = jax.vmap(cost_fn)(Y_prop)

        # Metropolis rule: accept with probability min(1, exp(-dcost / lambda_i)).
        dcost = cost_prop - cost
        u = jax.random.uniform(acc_rng, (args.Nsample,))
        accept = jnp.log(u + 1e-12) < -dcost / lambda_i

        a = jnp.where(accept[:, None], a_prop, a)
        cost = jnp.where(accept, cost_prop, cost)

        improved = cost < best_cost
        best_a = jnp.where(improved[:, None], a, best_a)
        best_cost = jnp.where(improved, cost, best_cost)

        k = jnp.argmin(best_cost)
        return (rng, a, cost, best_a, best_cost), jax.nn.one_hot(best_a[k], args.K)

    _, Ybars = jax.lax.scan(sa_iteration, (rng, a, cost, a, cost), lambdas)
    return Ybars
