"""Tempered Sequential Monte Carlo (SMC)."""

from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=(0, 1))
def smc(args, cost_fn, seed=42):
    """Sequential Monte Carlo for categorical assignment.

    Targets the sequence of distributions proportional to
    ``exp(-cost / lambda_t)`` with ``lambda_t = T0 * alpha**t``. Each
    iteration reweights the particles from ``lambda_{t-1}`` to ``lambda_t``,
    resamples them, and applies one Metropolis mutation step at ``lambda_t``.
    The mutation reassigns ``max(1, int(destroy_k_ratio * N))`` distinct
    random variables to uniformly random values.

    The particles are initialized and evaluated at ``lambda_0``, so there are
    ``Ndiffuse - 1`` iterations and ``Ndiffuse * Nsample`` cost evaluations
    in total.

    Parameters
    ----------
    args : object
        Hashable configuration with fields ``N``, ``K``, ``Nsample`` (number
        of particles), ``Ndiffuse``, ``T0`` (initial temperature), ``alpha``
        (geometric cooling rate) and ``destroy_k_ratio`` (fraction of
        variables changed by a mutation).
    cost_fn : callable
        Maps a one-hot array of shape ``(N, K)`` to a scalar cost.
    seed : int, optional
        Random seed. Default is 42.

    Returns
    -------
    Ybars : jax.Array, shape (Ndiffuse - 1, N, K)
        Row ``t`` is the lowest-cost particle after iteration ``t``.
    """
    rng = jax.random.PRNGKey(seed)

    rng, init_rng = jax.random.split(rng)
    particles = jax.random.randint(init_rng, (args.Nsample, args.N), 0, args.K)
    costs = jax.vmap(cost_fn)(jax.nn.one_hot(particles, args.K))

    lambdas = args.T0 * (args.alpha ** jnp.arange(args.Ndiffuse))

    def mutate(rng, a, cost, lambda_i):
        rng, var_rng, value_rng, accept_rng = jax.random.split(rng, 4)
        n_block = max(1, int(args.destroy_k_ratio * args.N))
        # Each particle picks n_block distinct variables, shape (Nsample, n_block).
        k = jax.vmap(lambda kk: jax.random.permutation(kk, args.N)[:n_block])( jax.random.split(var_rng, args.Nsample))
        value = jax.random.randint(value_rng, (args.Nsample, n_block), 0, args.K)
        proposals = a.at[jnp.arange(args.Nsample)[:, None], k].set(value)

        Y_prop = jax.nn.one_hot(proposals, args.K)
        cost_prop = jax.vmap(cost_fn)(Y_prop)

        dcost = cost_prop - cost
        u = jax.random.uniform(accept_rng, (args.Nsample,))
        accept = jnp.log(u) < -dcost / lambda_i
        a = jnp.where(accept[:, None], proposals, a)
        cost = jnp.where(accept, cost_prop, cost)

        return rng, a, cost

    def smc_iteration(carry, lambda_i):
        rng, particles, costs, lambda_prev = carry

        # Incremental importance weights from lambda_prev to lambda_i.
        log_weights = -(1 / lambda_i - 1 / lambda_prev) * costs
        weights = jax.nn.softmax(log_weights)

        rng, resample_rng = jax.random.split(rng)
        idx = jax.random.choice(resample_rng, args.Nsample, shape=(args.Nsample,), p=weights)
        particles, costs = particles[idx], costs[idx]

        rng, particles, costs = mutate(rng, particles, costs, lambda_i)

        best_idx = jnp.argmin(costs)
        best_particle = jax.nn.one_hot(particles[best_idx], args.K)

        return (rng, particles, costs, lambda_i), best_particle

    _, Ybars = jax.lax.scan(smc_iteration, (rng, particles, costs, lambdas[0]), lambdas[1:])

    return Ybars
