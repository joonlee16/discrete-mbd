"""Large Neighborhood Search (LNS) with a single incumbent."""

from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=(0, 1))
def lns(args, cost_fn, seed=42):
    """Large Neighborhood Search for categorical assignment.

    Keeps one incumbent assignment, starting from a uniformly random one.
    Each iteration builds ``Nsample`` neighbors, each of which changes
    ``max(1, int(destroy_k_ratio * N))`` distinct random variables of the
    incumbent to different values. The best neighbor replaces the incumbent
    only if it is strictly better.

    Parameters
    ----------
    args : object
        Hashable configuration with fields ``N``, ``K``, ``Nsample``,
        ``Ndiffuse`` (number of iterations) and ``destroy_k_ratio``
        (fraction of variables changed in each neighbor).
    cost_fn : callable
        Maps a one-hot array of shape ``(N, K)`` to a scalar cost.
    seed : int, optional
        Random seed. Default is 42.

    Returns
    -------
    jax.Array, shape (Ndiffuse, N, K)
        Row ``t`` is the incumbent after iteration ``t``, which is the best
        assignment found so far.
    """
    N, K, S = args.N, args.K, args.Nsample
    k = max(1, int(args.destroy_k_ratio * N))  # number of variables changed per neighbor
    batched_cost = jax.vmap(lambda a: cost_fn(jax.nn.one_hot(a, K)))
    rows = jnp.arange(S)[:, None]

    key, k0 = jax.random.split(jax.random.PRNGKey(seed))
    keys = jax.random.split(key, args.Ndiffuse)
    a0 = jax.random.randint(k0, (N,), 0, K)
    carry0 = (a0, cost_fn(jax.nn.one_hot(a0, K)))

    def step(carry, key):
        a_best, c_best = carry
        k_dest, k_rep = jax.random.split(key)

        # Destroy: each neighbor picks k distinct variables of the incumbent.
        D = jax.vmap(lambda kk: jax.random.permutation(kk, N)[:k])(jax.random.split(k_dest, S))
        # Repair: shift each picked variable by 1..K-1, so its value always changes.
        shift = jax.random.randint(k_rep, (S, k), 1, K)
        A = jnp.tile(a_best, (S, 1))
        A = A.at[rows, D].set((A[rows, D] + shift) % K)

        c = batched_cost(A)
        i = jnp.argmin(c)
        better = c[i] < c_best
        a_best = jnp.where(better, A[i], a_best)
        c_best = jnp.where(better, c[i], c_best)
        return (a_best, c_best), a_best

    _, a_hist = jax.lax.scan(step, carry0, keys)
    return jax.nn.one_hot(a_hist, K)
