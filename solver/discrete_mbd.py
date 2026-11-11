"""Discrete Model-Based Diffusion (MBD) over categorical variables."""

from functools import partial

import jax
import jax.numpy as jnp


def construct_Qs(Ndiffuse, K, a_min=1e-3):
    """Construct the transition matrices of the uniform categorical diffusion.

    The cumulative signal level follows the cosine schedule of Nichol &
    Dhariwal (2021), rescaled so that it never falls below ``a_min``:

        f(tau)      = cos^2(pi/2 * (tau + s) / (1 + s)),   s = 0.008,
        alpha_bar_i = a_min + (1 - a_min) * f(i / M) / f(0),   i = 0, ..., M,

    with M = ``Ndiffuse``. This gives alpha_bar_0 = 1 and alpha_bar_M =
    ``a_min``. Keeping alpha_bar_M > 0 makes every beta_i lie in (0, 1)
    without clipping. ``rounded_mbd.construct_alpha_bars`` uses the same
    schedule, so both MBD solvers see the same noise levels.

    Parameters
    ----------
    Ndiffuse : int
        Number of diffusion steps.
    K : int
        Number of categories.
    a_min : float, optional
        Final cumulative signal level alpha_bar_M, in (0, 1). Default 1e-3.

    Returns
    -------
    Qs : jax.Array, shape (Ndiffuse, K, K)
        One-step transition matrices. ``Qs[i]`` is Q_{i+1} =
        (1 - beta_{i+1}) I + beta_{i+1} 11^T / K.
    Qbars : jax.Array, shape (Ndiffuse, K, K)
        Cumulative transition matrices. ``Qbars[i]`` is Qbar_{i+1} =
        alpha_bar_{i+1} I + (1 - alpha_bar_{i+1}) 11^T / K.
    """
    s = 0.008

    tau = jnp.arange(Ndiffuse + 1) / Ndiffuse
    f = jnp.cos(0.5 * jnp.pi * (tau + s) / (1 + s)) ** 2
    alpha_bars = a_min + (1.0 - a_min) * f / f[0]

    # beta_i is chosen so that alpha_bar_i = alpha_bar_{i-1} (1 - beta_i).
    # alpha_bars > 0 and strictly decreasing, so beta_i lies in (0, 1).
    betas = 1.0 - alpha_bars[1:] / alpha_bars[:-1]

    eye = jnp.eye(K)
    uniform = jnp.ones((K, K)) / K

    Qs = (1.0 - betas[:, None, None]) * eye[None, :, :] + betas[:, None, None] * uniform[None, :, :]
    Qbars = (alpha_bars[1:, None, None] * eye[None, :, :] + (1.0 - alpha_bars[1:, None, None]) * uniform[None, :, :])

    return Qs, Qbars


def sample_categorical(rng, probs, Nsample):
    """Draw independent one-hot samples from per-variable categorical distributions.

    Parameters
    ----------
    rng : jax.Array
        PRNG key.
    probs : jax.Array, shape (N, K)
        Row ``i`` is the distribution of variable ``i``.
    Nsample : int
        Number of samples to draw.

    Returns
    -------
    jax.Array, shape (Nsample, N, K)
        One-hot samples.
    """
    N, K = probs.shape
    logits = jnp.log(probs + 1e-12)
    logits = jnp.broadcast_to(logits[None, :, :], (Nsample, N, K))
    idx = jax.random.categorical(rng, logits, axis=-1)
    return jax.nn.one_hot(idx, K)


@partial(jax.jit, static_argnums=(0, 1))
def discrete_mbd(args, cost_fn, seed=42):
    """Discrete Model-Based Diffusion (MBD).

    Starting from a uniformly random assignment x_M, each of the ``Ndiffuse``
    reverse steps i = M, ..., 1 does the following:

    1. Sample ``int(sample_ratio * Nsample)`` candidates for x_0, one-hot per
       variable, from ``x_i Qbar_i^T``. Each variable keeps its current value
       with probability alpha_bar_i + (1 - alpha_bar_i) / K and otherwise
       moves to a uniformly random value.
    2. Weight the candidates by ``softmax(-cost / temp_sample)``. The weighted
       average W is a per-variable categorical estimate of x_0. Averaging the
       categorical posterior q(x_{i-1} | x_i, x_0) over W gives the reverse
       distribution p_i(x_{i-1} | x_i). The costs are not normalized, so a
       small ``temp_sample`` makes the weights close to one-hot on the best
       candidate.
    3. Draw the remaining ``Nsample - int(sample_ratio * Nsample)`` samples
       from p_i, and take the lowest-cost assignment among all ``Nsample``
       samples of this step as the next iterate x_{i-1}.

    The iterate of a step is the best sample of that step only. It can cost
    more than an earlier iterate.

    Parameters
    ----------
    args : object
        Hashable configuration with fields ``N``, ``K``, ``Nsample``,
        ``Ndiffuse``, ``temp_sample`` (temperature of the cost weights) and
        ``sample_ratio`` (fraction of ``Nsample`` used in stage 1).
    cost_fn : callable
        Maps a one-hot array of shape ``(N, K)`` to a scalar cost.
    seed : int, optional
        Random seed. Default is 42.

    Returns
    -------
    Y_history : jax.Array, shape (Ndiffuse, N, K)
        Row ``t`` is the lowest-cost sample drawn at reverse step ``t``,
        which is also the iterate carried to the next step.
    """
    Qs, Qbars = construct_Qs(args.Ndiffuse, args.K)
    Qbars_prev = jnp.concatenate([jnp.eye(args.K)[None], Qbars[:-1]], axis=0)

    rng = jax.random.PRNGKey(seed)
    p_M = jnp.ones((args.N, args.K)) / args.K

    sample_number1 = int(args.Nsample * args.sample_ratio)
    sample_number2 = args.Nsample - sample_number1
    rng, init_rng = jax.random.split(rng)
    x_i = sample_categorical(init_rng, p_M, 1)[0]

    def reverse_step(carry, t_rev):
        rng, x_i = carry

        rng, sample_rng, final_sample_rng = jax.random.split(rng, 3)
        Qbar_i = Qbars[t_rev]

        # 1. Sample candidates from the forward marginal around the current iterate.
        phi = x_i @ Qbar_i.T
        phi = phi / (jnp.sum(phi, axis=-1, keepdims=True))
        x_candidates = sample_categorical(sample_rng, phi, sample_number1)

        # 2. Estimate the reverse distribution p(x_{i-1} | x_i).
        # posterior[r, j, k] = q(x_{i-1} = k | x_i, x_0 = j) for variable r.
        posterior = jnp.einsum("rt,kt,jt->rjk", x_i, Qs[t_rev], 1.0 / jnp.clip(Qbar_i, 1e-30))
        posterior = posterior * Qbars_prev[t_rev, None]

        costs = jax.vmap(cost_fn)(x_candidates)
        log_p0 = -costs / args.temp_sample
        weights = jax.nn.softmax(log_p0)

        # W[r, j] is the weighted probability that variable r takes value j in x_0.
        W = jnp.einsum("s,srj->rj", weights, x_candidates)
        p_i = jnp.einsum("rj,rjk->rk", W, posterior)

        # 3. Sample from p_i and keep the best sample of this step.
        sampled_x = sample_categorical(final_sample_rng, p_i, sample_number2)
        sampled_costs = jax.vmap(cost_fn)(sampled_x)

        all_samples = jnp.concatenate([x_candidates, sampled_x], axis=0)
        all_costs = jnp.concatenate([costs, sampled_costs], axis=0)
        xnew = all_samples[jnp.argmin(all_costs)]

        return (rng, xnew), xnew

    _, Y_history = jax.lax.scan(reverse_step, (rng, x_i), jnp.arange(args.Ndiffuse - 1, -1, -1))

    return Y_history
