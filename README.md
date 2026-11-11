# Discrete Model-Based Diffusion


Code for the our paper ["Discrete Model-Based Diffusion for Combinatorial Optimization in Multi-Robot Task Allocation"](https://arxiv.org/pdf/2611.13382).

Discrete Model-Based Diffusion (Discrete MBD) is a sampling-based, zeroth-order solver for discrete optimization problems

```math
\min_{\mathbf{y} \in \{0, 1, \dots, K-1\}^N} J(\mathbf{y}),
```

where $`J`$ is an objective function. We make no assumptions about the continuity or smoothness of $`J`$. This repository contains a JAX implementation of Discrete MBD, five sampling-based baselines, and two random benchmark problems to compare them on.

## Method

Model-Based Diffusion (Pan et al., 2024) solves optimization problems by running the reverse process of a diffusion model whose score is estimated from samples of the cost function instead of a trained network. Discrete MBD carries this idea over to categorical variables.

**Notation.** $`\mathbf{y} \in \{0, \dots, K-1\}^N`$ is an integer assignment, and $`y_n`$ is the value of variable $`n`$. $`\mathbf{x} \in \{0, 1\}^{N \times K}`$ is its one-hot encoding: row $`n`$ of $`\mathbf{x}`$ is the one-hot vector of $`y_n`$. We write $`J(\mathbf{x})`$ for the cost of the assignment that $`\mathbf{x}`$ encodes. Diffusion steps are indexed by $`i = 0, \dots, M`$, so $`\mathbf{x}^{(i)}`$ is the one-hot array at step $`i`$ and $`\mathbf{x}^{(0)}`$ is a clean assignment.

**Forward process.** Each variable is corrupted independently by a uniform categorical diffusion (Austin et al., 2021). After $`i`$ steps,

```math
q(\mathbf{x}^{(i)} \mid \mathbf{x}^{(0)}) = \mathrm{Cat}\left(\mathbf{x}^{(i)};\ \mathbf{x}^{(0)} \bar Q_i\right), \qquad \bar Q_i = \bar\alpha_i I + (1 - \bar\alpha_i) \frac{\mathbf 1 \mathbf 1^\top}{K},
```

where $`\mathrm{Cat}`$ is applied to each row. A variable keeps its value with probability $`\bar\alpha_i + (1 - \bar\alpha_i)/K`$ and otherwise moves to a uniformly random value. The signal level $`\bar\alpha_i`$ follows a cosine schedule (Nichol & Dhariwal, 2021) that decreases from $`\bar\alpha_0 = 1`$ to $`\bar\alpha_M = 10^{-3}`$.

**Reverse process.** Starting from a uniformly random assignment $`\mathbf{x}^{(M)}`$, each reverse step $`i = M, \dots, 1`$:

1. samples candidate solutions $`\mathbf{x}^{(0)}`$ around the current iterate from $`\mathrm{Cat}(\mathbf{x}^{(i)} \bar Q_i)`$,
2. weights the candidates by $`\exp(-J(\mathbf{x}^{(0)})/\lambda)`$, which gives a per-variable categorical estimate of $`\mathbf{x}^{(0)}`$,
3. combines this estimate with the exact categorical posterior $`q(\mathbf{x}^{(i-1)} \mid \mathbf{x}^{(i)}, \mathbf{x}^{(0)})`$ to get the reverse distribution $`p(\mathbf{x}^{(i-1)} \mid \mathbf{x}^{(i)})`$,
4. draws further samples from $`p(\mathbf{x}^{(i-1)} \mid \mathbf{x}^{(i)})`$, and takes the lowest-cost sample of the step as $`\mathbf{x}^{(i-1)}`$.

As $`i`$ decreases, the noise level falls and the search moves from global exploration to local refinement around the best solutions found. Every step only evaluates $`J`$ on a batch of samples, which runs in parallel with `jax.vmap`, and the whole solver is compiled with `jax.jit`.

## Benchmark problems

Both problems are sampled at random from a seed, so every trial is a new instance.

| Problem | Objective | Instance |
|---|---|---|
| **UIQP**, Unconstrained Integer Quadratic Program (`cost_function/uiqp.py`) | $`J(\mathbf{y}) = \mathbf{y}^\top Q \mathbf{y} + c^\top \mathbf{y}`$ | $`Q`$ symmetric with Gaussian entries, $`c`$ Gaussian |
| **UPO**, Unconstrained Polynomial Optimization (`cost_function/upo.py`) | $`J(\mathbf{y}) = \sum_{k=1}^{250} c_k \prod_{j \in P_k} y_j`$ | 250 monomials, each over 5 distinct random variables, Gaussian $`c_k`$ |

In both problems $`\mathbf{y} \in \{0, \dots, K-1\}^N`$ and $`Q`$ is indefinite in general, so the objectives are nonconvex. Each problem module also provides `actual_optimal`, which solves the same instance with Gurobi as a reference. Gurobi stops at a time limit (30 s by default), so its answer is not always proven optimal.

## Solvers

| Name | File | Description |
|---|---|---|
| `MBD` | `solver/discrete_mbd.py` | Discrete Model-Based Diffusion (this work) |
| `Rounded-MBD` | `solver/rounded_mbd.py` | Gaussian MBD on a continuous relaxation in $`[-1, 1]^{N \times K}`$; every sample is rounded to a one-hot assignment as in DIFUSCO (Sun & Yang, 2023) before its cost is evaluated. Uses the same noise schedule as `MBD`. |
| `CEM` | `solver/cem.py` | Cross-Entropy Method with independent categorical distributions |
| `SA` | `solver/sa.py` | Simulated Annealing with parallel chains |
| `SMC` | `solver/smc.py` | Tempered Sequential Monte Carlo |
| `LNS` | `solver/lns.py` | Large Neighborhood Search |
| `Gurobi` | `cost_function/*.py` | Reference solution from Gurobi (needs a license) |

All sampling-based solvers have the same interface and the same budget: `Nsample` cost evaluations per iteration for `Ndiffuse` iterations.

## Repository structure

```
.
├── README.md
├── pyproject.toml          # Dependencies (read by uv)
├── uv.lock                 # Exact versions used for the experiments
├── run.py                  # Benchmark script: runs the solvers on random UIQP / UPO instances
├── cost_function/
│   ├── uiqp.py             # UIQP instances and Gurobi reference
│   └── upo.py              # UPO instances and Gurobi reference
└── solver/
    ├── discrete_mbd.py     # Discrete Model-Based Diffusion
    ├── rounded_mbd.py      # Rounded (continuous-relaxation) MBD baseline
    ├── cem.py              # Cross-Entropy Method
    ├── sa.py               # Simulated Annealing
    ├── smc.py              # Sequential Monte Carlo
    └── lns.py              # Large Neighborhood Search
```

## Setup

The code was developed with Python 3.13 and tested on GPU with JAX and CUDA 12.

```bash
uv sync --extra cuda12 --extra gurobi 
```

## Running the benchmark

Run every sampling-based solver on 5 random UIQP instances with $`N = 10`$ and $`K = 10`$:

```bash
python run.py
```

Choose the problem, its size, the methods and the number of trials from the command line:

```bash
python run.py --problem UPO --N 50 --K 10 --methods MBD CEM SA --trials 10
python run.py --problem UIQP --N 50 --K 50 --methods MBD Gurobi
```

| Option | Default | Meaning |
|---|---|---|
| `--problem` | `UIQP` | `UIQP` or `UPO` |
| `--N` | `10` | number of variables |
| `--K` | `10` | number of values per variable |
| `--methods` | all but `Gurobi` | any of `MBD Rounded-MBD CEM SA SMC LNS Gurobi` |
| `--trials` | `5` | number of random instances |

Each solver is called once to compile it before the timed call, so the reported runtime excludes JIT compilation.

The solver hyperparameters are the fields of `Args` in `run.py`:

| Field | Default | Used by |
|---|---|---|
| `Ndiffuse` | `100` | all: number of iterations (diffusion steps for MBD) |
| `Nsample` | `2048` | all: samples per iteration |
| `temp_sample` | `1e-5` | MBD, Rounded-MBD: temperature $`\lambda`$ of the cost weights |
| `sample_ratio` | `0.75` | MBD: fraction of samples drawn around $`\mathbf{x}^{(i)}`$ (step 1) |
| `elite_frac` | `0.01` | CEM: fraction of samples kept as elites |
| `T0`, `alpha` | `1.0`, `0.95` | SA, SMC: initial temperature and geometric cooling rate |
| `destroy_k_ratio` | `0.1` | SMC, LNS: fraction of variables changed per move |

## Notes

- The costs are not normalized before the weights $`\exp(-J(\mathbf{x})/\lambda)`$ are computed, so with the default `temp_sample = 1e-5` the weights are close to one-hot on the best candidate. On UPO the scaled costs are very large; on CPU this can overflow `float32` inside the jitted softmax. We ran all experiments on GPU, where this does not occur.

## References

- C. Pan, Z. Yi, G. Shi, and G. Qu. Model-Based Diffusion for Trajectory Optimization. NeurIPS 2024.
- J. Austin, D. D. Johnson, J. Ho, D. Tarlow, and R. van den Berg. Structured Denoising Diffusion Models in Discrete State-Spaces. NeurIPS 2021.
- A. Q. Nichol and P. Dhariwal. Improved Denoising Diffusion Probabilistic Models. ICML 2021.
- Z. Sun and Y. Yang. DIFUSCO: Graph-based Diffusion Solvers for Combinatorial Optimization. NeurIPS 2023.

## Citation

If you find this work useful, please cite:

```bibtex
@misc{lee2026discrete,
  title         = {Discrete Model-Based Diffusion for Combinatorial Optimization in Multi-Robot Task Allocation},
  author        = {Lee, Haejoon and Vielmetti, Thomas Marshall and Wang, Xinyi and Panagou, Dimitra},
  year          = {2026},
  eprint        = {2611.13382},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  doi           = {10.48550/arXiv.2611.13382},
  url           = {https://arxiv.org/abs/2611.13382}
}
```