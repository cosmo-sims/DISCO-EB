# Motivations

Linear perturbation theory describes the universe from inflation until
structure turns nonlinear. In Fourier space the perturbations of each wave
number $k$ obey a set of coupled ordinary differential equations, the
Einstein-Boltzmann system,

$$
\frac{\mathrm{d}\mathbf{y}_k}{\mathrm{d}\tau} = f(\tau, \mathbf{y}_k;\, k, \boldsymbol{\theta}),
$$

where $\mathbf{y}_k$ holds the metric potentials, the density and velocity
of each species and the multipoles of the photon and neutrino distributions,
and $\boldsymbol{\theta}$ are the cosmological parameters. Its solution gives
the matter power spectrum and the CMB anisotropy spectra, the starting point
of nearly every cosmological analysis.

DISCO-EB is written in JAX and its Einstein-Boltzmann solver is fully compatible
with `jax.jvp` and `jax.jacfwd`. The first version of the code ([Hahn et al. 2024](https://arxiv.org/abs/2311.03291)) demonstrated the power of differentiable solves, and in this new version we have:

1. accelerated the solves using fused integration kernels compiled with `numba`,
2. optimised the solver for a wider range of cosmological models using sparsity-aware linear algebra and auto-differentiation techniques, and
3. generalised the code to make it easy to incorporate new species and new physics.

## Why a differentiable solver?

Derivatives of the power spectra are needed for Fisher forecasts and for
gradient-based samplers such as Hamiltonian Monte Carlo. [CAMB](https://camb.info) and [CLASS](https://lesgourg.github.io/class_public/class.html) do not provide derivatives. Their users must resort to finite differences, which costs two solves per parameter and needs a step size tuned
against the solver's own noise.

## Why run solves on a GPU?

An inference run evaluates thousands to millions of cosmologies, and each
evaluation involves a few hundred wave modes. Every (wave mode, cosmology) pair is an
independent stiff system of tens to hundreds of variables depending on the exact model.
This kind of large ensemble of small ODEs can be efficiently integrated on GPUs using 
[modax](https://lawrenceberry.github.io/modax/) (see the [paper](https://arxiv.org/abs/2609.35661) for a detailed discussion).

## Why structure the code using components?

Most analyses extend ΛCDM: dark energy with an equation of state, massive
neutrinos, a second dark species, another theory of gravity. In a
monolithic solver each extension edits the right-hand side, the initial
conditions and the parameter handling. In DISCO-EB a species is defined by a class
that declares its states, parameters and equations, and a model combined species by 
inheriting from these classes. A new species doesn't require changing the rest of the code,
just adding a new class. The
[Design](design.md) page explains how the components become one GPU kernel.
