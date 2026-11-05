# DISCO-EB - The DISCO-DJ Einstein-Boltzmann module
Implementation of a differentiable linear Einstein-Boltzmann solver for cosmology in JAX -- a module of the DISCO-DJ framework (DIfferentiable Simulations for COsmology, Done with Jax).

Note that this program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY.

Currently supported features (the list is growing, so check back):

- Autodifferentiable via JAX
- Standard LCDM model with Quintessence DE fluid (w0,wa) and massive neutrinos (one species)
- Thermal history solver based on a simplified Recfast implementation in the module
- Numerous example jupyter notebooks, e.g. for Euclid-like Fisher forecasts
  
New modules/plugins can be easily added (see how to contribute in CONTRIBUTING.md file). We are enthusiastic if extensions/improvements that are of broader interest are re-integrated into the master branch and DISCO-EB grows as a community effort.

Currently work in progress:

- performance improvements

## Installation

Use `uv` to install the pinned GPU compiler and solver dependencies:

```bash
git clone https://github.com/ohahn/DISCO-EB.git
cd DISCO-EB
uv sync
```

### Current CUDA solver environment

Use `uv sync` for the current GPU solver stack. The project pins source
revisions of Enzyme and modax: Enzyme needs optimizer passes to compile the
generated perturbation equations, and modax needs a sparse diagonal fix to
evolve them correctly. Their PyPI releases lack these fixes. The Enzyme
source build downloads the matching LLVM/Enzyme tools automatically.
Plain `pip install` does not apply the source pins or compiler constraints
in `[tool.uv]` and is not sufficient for this solver stack.

```bash
uv run python examples/solve_eb_comoving_curvature.py
```

With modax 0.0.4, numba-enzyme-cuda 0.2.2 and numba-cuda-mlir 0.5.1,
DISCO-EB automatically compiles Rodas5P and its derivatives for `sm_90` on
Blackwell GPUs, then links for the actual GPU. This keeps both inputs on the
LLVM path supported by Enzyme. Setting `NUMBA_ENZYME_CUDA_CC` before the first
solve is respected for both compiler inputs; setting it alone without the
matching Rodas5P target causes a linker data-layout error. The compatibility
adapter uses modax's internal compilation hook until modax exposes a target
option.

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

