"""Cross-Entropy Method (CEM) over independent categorical distributions."""

from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=(0, 1))
def cem(args, cost_fn, seed=42):
    """Cross-Entropy Method for categorical assignment.

    Keeps an independent categorical distribution for each variable, starting
    from uniform. Each iteration samples ``Nsample`` assignments and refits
    the distribution to the empirical frequencies of the lowest-cost
    ``elite_frac`` fraction of them.

    Parameters
    ----------
    args : object
        Hashable configuration with fields ``N``, ``K``, ``Nsample``,
        ``Ndiffuse`` (number of iterations) and ``elite_frac``.
    cost_fn : callable
        Maps a one-hot array of shape ``(N, K)`` to a scalar cost.
    seed : int, optional
        Random seed. Default is 42.

    Returns
    -------
    Ybars : jax.Array, shape (Ndiffuse, N, K)
        Row ``t`` is the lowest-cost sample of iteration ``t``.
    """
    rng = jax.random.PRNGKey(seed)

    # probs[i, k] is the probability that variable i takes value k.
    probs = jnp.ones((args.N, args.K)) / args.K
    Nelite = max(1, int(args.Nsample * args.elite_frac))

    def cem_iteration(carry, _):
        rng, probs = carry

        rng, sample_rng = jax.random.split(rng)
        logits = jnp.log(probs + 1e-8)
        logits = jnp.broadcast_to(logits, (args.Nsample, args.N, args.K))
        assignment_indices = jax.random.categorical(sample_rng, logits=logits, axis=-1)
        Y0s = jax.nn.one_hot(assignment_indices, args.K)  # (Nsample, N, K)

        costs = jax.vmap(cost_fn)(Y0s)
        elite_idx = jnp.argsort(costs)[:Nelite]
        elite_samples = Y0s[elite_idx]

        # Refit to the elite frequencies.
        probs = jnp.mean(elite_samples, axis=0)

        best_idx = jnp.argmin(costs)
        best_assignment = Y0s[best_idx]

        return (rng, probs), best_assignment

    _, Ybars = jax.lax.scan(cem_iteration, (rng, probs), None, length=args.Ndiffuse)

    return Ybars
