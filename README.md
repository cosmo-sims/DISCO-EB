# DISCO-EB - The DISCO-DJ Einstein-Boltzmann module
Implementation of a differentiable linear Einstein-Boltzmann solver for cosmology in JAX -- a module of the DISCO-DJ framework (DIfferentiable Simulations for COsmology, Done with Jax).

Note that this program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY.

Currently supported features (the list is growing, so check back):

- Autodifferentiable via JAX
- Standard LCDM model with Quintessence DE fluid (w0,wa) and massive neutrinos (one species)
- Thermal history from a RECFAST implementation in the module, integrated with the perturbations of every wave mode in one GPU solve (a block of a modax hierarchy that steps on its own and publishes the scale factor, opacity, sound speed and neutrino density to the perturbations as interpolants; no tabulated thermodynamics)
- Numerous example jupyter notebooks, e.g. for Euclid-like Fisher forecasts
  
New modules/plugins can be easily added (see how to contribute in CONTRIBUTING.md file). We are enthusiastic if extensions/improvements that are of broader interest are re-integrated into the master branch and DISCO-EB grows as a community effort.

Currently work in progress:

- performance improvements

## Installation

Use `uv` to install the GPU compiler and solver dependencies with the
constraints this project pins:

```bash
git clone https://github.com/cosmo-sims/DISCO-EB.git
cd DISCO-EB
uv sync
```

### Current CUDA solver environment

`uv sync` installs modax and numba-enzyme-cuda from PyPI, together with the
numba-cuda-mlir and CUDA toolkit versions pinned in `[tool.uv]`. Plain
`pip install` does not apply those constraints, nor the override that lets
numba-enzyme-cuda run with current numba, and is not sufficient for this
solver stack.

```bash
uv run python examples/solve_eb_comoving_curvature.py
```

On Blackwell GPUs (compute capability 10.0 and up) modax compiles Rodas5P and
its Enzyme derivatives, including the tangents of the quantities `solve_eb`
keeps, for `sm_90` and links the result for the actual GPU,
which keeps both on the LLVM 7 dialect Enzyme's LLVM 15 toolchain reads.
`NUMBA_ENZYME_CUDA_CC`, set before the first solve, chooses that target for
both instead.

### For Development

`uv sync` installs this checkout in editable mode. Add the development
dependencies and run the tests with:

```bash
uv sync --extra dev
uv run pytest
```

The solver tests should be run on GPU platforms.


## Getting started

Start with the sample notebook [nb_minimal_example.ipynb](https://github.com/ohahn/DISCO-EB/blob/master/notebooks/nb_minimal_example.ipynb) in the [notebooks subdirectory](https://github.com/ohahn/DISCO-EB/tree/master/notebooks). It explains how to compute a matter power spectrum and take a derivative w.r.t. a cosmological parameter.

## Benchmarks 

See [BENCHMARKS.md](BENCHMARKS.md) for benchmark results on different hardware.

## Contributing and Licensing

See file CONTRIBUTING.md on how to contribute to the development. 

The software is licensed under GPL v3 (see file LICENSE).

## Citing in scientific publications or presentations

If you use DISCO-EB in your scientific work, you are required to acknowledge this by linking to this repository and citing the relevant papers:

- Hahn et al. 2024 [arXiv:2311.03291](https://arxiv.org/abs/2311.03291) [JCAP OA](https://doi.org/10.1088/1475-7516/2024/06/063)


## Required Python Packages
- JAX
- diffrax
- equinox
- jaxtyping
- jax_cosmo

