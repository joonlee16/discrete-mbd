"""Rounded Model-Based Diffusion: a continuous-relaxation baseline.

Runs the Gaussian Model-Based Diffusion of Pan et al. (2024) on a continuous
relaxation of the one-hot assignment in [-1, 1]^{N x K}. Every sample is
rounded to a one-hot assignment before its cost is evaluated, following the
rounding of DIFUSCO (Sun & Yang, 2023).
"""

from functools import partial

import jax
import jax.numpy as jnp


def construct_alpha_bars(Ndiffuse, s=0.008, a_min=1e-3):
    """Cumulative signal levels of the floored cosine schedule.

    This is the same schedule as ``construct_Qs`` in ``discrete_mbd.py``, so
    the rounded and discrete MBD solvers use the same noise levels:

        f(tau)      = cos^2(pi/2 * (tau + s) / (1 + s)),
        alpha_bar_i = a_min + (1 - a_min) * f(i / M) / f(0),   i = 0, ..., M,

    with M = ``Ndiffuse`` (cosine schedule of Nichol & Dhariwal, 2021,
    floored at ``a_min``). The largest noise variance used by the reverse
    process is then 1 / a_min - 1.

    Parameters
    ----------
    Ndiffuse : int
        Number of diffusion steps.
    s : float, optional
        Offset of the cosine schedule. Default is 0.008, as in
        ``construct_Qs``.
    a_min : float, optional
        Final cumulative signal level alpha_bar_M, in (0, 1). Default is
        1e-3, as in ``construct_Qs``.

    Returns
    -------
    alpha_bars : jax.Array, shape (Ndiffuse + 1,)
        ``alpha_bars[i]`` is alpha_bar_i, with ``alpha_bars[0] = 1`` and
        ``alpha_bars[Ndiffuse] = a_min``.
    """
    tau = jnp.arange(Ndiffuse + 1) / Ndiffuse
    f = jnp.cos(0.5 * jnp.pi * (tau + s) / (1 + s)) ** 2
    alpha_bars = a_min + (1.0 - a_min) * f / f[0]
    return alpha_bars


def round_to_onehot(Y0):
    """Round a continuous sample in [-1, 1]^{N x K} to a one-hot assignment.

    This mirrors, on purpose, the decoding of the continuous (Gaussian)
    variant of DIFUSCO [1]. The sample is mapped to the heatmap
    ``0.5 * (Y0 + 1)`` and binarized at 0.5. Assignments are then decoded
    greedily by heatmap score: in each row, the highest-scoring entry above
    the threshold is kept, or the highest-scoring entry overall if no entry
    is above it.

    For the one-value-per-variable constraint used here, this decoding is the
    same as the row-wise argmax of ``Y0``. The DIFUSCO form is kept so that
    the baseline follows the published decoding step by step.

    Parameters
    ----------
    Y0 : jax.Array, shape (N, K)
        Continuous sample in [-1, 1].

    Returns
    -------
    Y : jax.Array, shape (N, K)
        One-hot assignment.

    References
    ----------
    .. [1] Z. Sun and Y. Yang, "DIFUSCO: Graph-based Diffusion Solvers for
       Combinatorial Optimization," NeurIPS 2023. arXiv:2302.08224.
    """
    heatmap = jnp.clip(0.5 * (Y0 + 1.0), 0.0, 1.0)
    Ybin = heatmap > 0.5
    # Shift entries below the threshold down by 1 so that entries above it always rank first.
    score = jnp.where(Ybin, heatmap, heatmap - 1.0)
    return jax.nn.one_hot(jnp.argmax(score, axis=-1), Y0.shape[-1])


@partial(jax.jit, static_argnums=(0, 1))
def rounded_mbd(args, cost_fn, seed=42):
    """Model-Based Diffusion on the [-1, 1] relaxation with rounding.

    Starting from Y_N = 0, each reverse step samples ``Nsample`` points from
    N(Y_i / sqrt(alpha_bar_i), (1 / alpha_bar_i - 1) I), clips them to
    [-1, 1] and rounds each one before evaluating its cost. The
    ``softmax(-cost / temp_sample)``-weighted mean of the unrounded samples
    estimates Y_0, which gives the score used to compute Y_{i-1}.

    Parameters
    ----------
    args : object
        Hashable configuration with fields ``N``, ``K``, ``Nsample``,
        ``Ndiffuse`` and ``temp_sample`` (temperature of the cost weights).
    cost_fn : callable
        Maps a one-hot array of shape ``(N, K)`` to a scalar cost.
    seed : int, optional
        Random seed. Default is 42.

    Returns
    -------
    Ybars : jax.Array, shape (Ndiffuse, N, K)
        Row ``t`` is the rounded iterate Y_{i-1} after reverse step ``t``.
        It is not necessarily the lowest-cost sample of that step.
    """
    rng = jax.random.PRNGKey(seed)

    YN = jnp.zeros((args.N, args.K))

    alphas_bar = construct_alpha_bars(args.Ndiffuse)  # (Ndiffuse + 1,), alphas_bar[0] = 1
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = jnp.concatenate([jnp.ones(1), alphas])  # alphas[i] = alpha_i

    def project(Y0s):
        """Round every continuous sample (B, N, K) to a one-hot assignment (B, N, K)."""
        return jax.vmap(round_to_onehot)(Y0s)

    def reverse_once(rng, Yi, i):
        """One reverse step from Y_i. Returns ``(rng, Y_{i-1})`` and the rounded Y_{i-1}."""
        rng, Y0s_rng = jax.random.split(rng)

        # Sample from q(Y0 | Yi) = N(Yi / sqrt(alpha_bar_i), (1 / alpha_bar_i - 1) I) and clip to [-1, 1].
        samples = jax.random.normal(Y0s_rng, (args.Nsample, args.N, args.K))
        Y0s = samples * jnp.sqrt(1.0 / alphas_bar[i] - 1.0) + Yi / jnp.sqrt(alphas_bar[i])
        Y0s = jnp.clip(Y0s, -1.0, 1.0)

        # Costs are evaluated on the rounded samples, but the weighted mean uses the continuous ones.
        Y0s_round = project(Y0s)
        costs = jax.vmap(cost_fn)(Y0s_round)
        logp0 = -costs / args.temp_sample
        weights = jax.nn.softmax(logp0)
        Ybar = jnp.einsum("s,snk->nk", weights, Y0s)

        score = 1 / (1.0 - alphas_bar[i]) * (-Yi + jnp.sqrt(alphas_bar[i]) * Ybar)
        Yim1 = 1 / jnp.sqrt(alphas[i]) * (Yi + (1.0 - alphas_bar[i]) * score)
        Yim1_rounded = round_to_onehot(Yim1)
        return (rng, Yim1), Yim1_rounded

    def scan_body(carry, i):
        rng, Yi = carry
        return reverse_once(rng, Yi, i)

    # Reverse steps i = Ndiffuse, ..., 1. Row t of Ybars is the rounded iterate after step t.
    _, Ybars = jax.lax.scan(scan_body, (rng, YN), jnp.arange(args.Ndiffuse, 0, -1))
    return Ybars
