<h1 align="center">DISCO-EB</h1>

<!--
The coverage badge reads a gist that the `tests` workflow rewrites on every
push to main. The id below is the repository variable COVERAGE_GIST_ID.
These badges sit above the `overview` snippet marker, so the documentation
site does not repeat them.
-->
<p align="center">
  <a href="https://github.com/cosmo-sims/DISCO-EB/actions/workflows/tests.yml"><img alt="tests" src="https://github.com/cosmo-sims/DISCO-EB/actions/workflows/tests.yml/badge.svg?branch=main"></a>
  <a href="https://github.com/cosmo-sims/DISCO-EB/actions/workflows/tests.yml"><img alt="coverage" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Flawrenceberry%2Fb451e14a9620fabb0d49c345f13b5414%2Fraw%2Fdiscoeb-coverage.json"></a>
  <a href="https://github.com/cosmo-sims/DISCO-EB/actions/workflows/lint.yml"><img alt="lint" src="https://github.com/cosmo-sims/DISCO-EB/actions/workflows/lint.yml/badge.svg?branch=main"></a>
  <a href="https://github.com/cosmo-sims/DISCO-EB/actions/workflows/types.yml"><img alt="types" src="https://github.com/cosmo-sims/DISCO-EB/actions/workflows/types.yml/badge.svg?branch=main"></a>
  <a href="https://pypi.org/project/discoeb/"><img alt="PyPI" src="https://img.shields.io/pypi/v/discoeb"></a>
  <a href="https://pypi.org/project/discoeb/"><img alt="Downloads" src="https://img.shields.io/pypi/dm/discoeb"></a>
  <a href="https://pypi.org/project/discoeb/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/discoeb"></a>
  <a href="https://docs.jax.dev/"><img alt="JAX" src="https://img.shields.io/badge/JAX-compatible-3f51b5"></a>
  <a href="https://arxiv.org/abs/2311.03291"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2311.03291-b31b1b.svg"></a>
  <a href="https://www.conventionalcommits.org/en/v1.0.0/"><img alt="Conventional Commits" src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196?logo=conventionalcommits&logoColor=white"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="https://cosmo-sims.github.io/DISCO-EB/">Documentation</a> •
  <a href="https://arxiv.org/abs/2311.03291">Paper</a> •
  <a href="#citation">Citation</a>
</p>

<!-- start:overview -->

DISCO-EB is a **differentiable linear Einstein-Boltzmann solver** for cosmology, and the Einstein-Boltzmann module of the DISCO-DJ framework (DIfferentiable Simulations for COsmology, Done with Jax). It evolves the background, the recombination history and the linear perturbations of every wave mode of a **batch of cosmologies in one GPU launch**.

The solve runs as a [modax](https://github.com/lawrenceberry/modax) Rodas5P kernel. RECFAST and the optical depth step alongside the perturbation hierarchy in the same kernel, so the solve needs no tables. You choose which quantities to keep at which times, and the kernel evaluates them as it goes. The results are JAX arrays, differentiable in the cosmological parameters by forward sensitivities. DISCO-EB supports flat or curved ΛCDM, a CPL dark-energy fluid $(w_0, w_a)$, massive neutrinos and reionization.

<!-- end:overview -->

The [documentation](https://cosmo-sims.github.io/DISCO-EB/) covers the API and how to use it. The [paper](https://arxiv.org/abs/2311.03291) describes the method and its accuracy against CLASS.

## Installation

<!-- start:install -->

```bash
pip install "discoeb[cuda13]"   # or [cuda12], which Python 3.10 needs
```

You need a GPU to run a solve, but not to import the package. modax brings the CUDA compiler stack with it. The CMB spectra live in a separate package, [DISCO-CMB](https://github.com/lawrenceberry/DISCO-CMB), which builds on this one.

<!-- end:install -->

## Quickstart

<!-- start:quickstart -->

This example keeps the comoving curvature perturbation $\mathcal{R}$ and the CDM density contrast of 64 wave modes at 100 conformal times:

```python
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
from discoeb import solve_eb

h = 0.6732
cosmology = {
    "Omegam": 0.1424 / h**2, "Omegab": 0.02238 / h**2, "H0": 100 * h,
    "w_DE_0": -1.0, "w_DE_a": 0.0, "cs2_DE": 1.0, "Omegak": 0.0,
    "A_s": 2.1e-9, "n_s": 0.966, "k_p": 0.05,
    "Tcmb": 2.7255, "YHe": 0.2454, "Neff": 3.046, "Nmnu": 0.0, "mnu": 0.0,
}
k = np.geomspace(1e-4, 1.0, 64)      # 1/Mpc
tau = np.geomspace(1.0, 2e4, 100)    # Mpc, the last times past today

solution = solve_eb(cosmology, k, tau, keep=("R", "clxc"))
R = solution["R"]  # (1, 64, 100): cosmologies, wave modes, times
```

<!-- end:quickstart -->

## Development

<!-- start:development -->

The repository uses [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/cosmo-sims/DISCO-EB.git && cd DISCO-EB
uv sync --extra cuda13  # or --extra cuda12

uv run pytest tests -rs   # needs a GPU, and -rs shows skipped CLASS comparisons
uv run ruff format && uv run ruff check --fix
uv run ty check
uv run pre-commit install   # the checks above, and a commit-message check
```

The documentation (Sphinx with MyST and autodoc) builds in an environment of
its own, the package's CPU install plus the docs group:

```bash
export UV_PROJECT_ENVIRONMENT=.venv-docs
uv sync --no-default-groups --group docs
uv run --no-sync sphinx-autobuild docs docs/_build/html   # live preview
uv run --no-sync sphinx-build -W docs docs/_build/html    # what CI runs
```

<!-- end:development -->

## Citation

<!-- start:citation -->

If you use DISCO-EB in your research, please cite the
[JCAP article](https://doi.org/10.1088/1475-7516/2024/06/063):

```bibtex
@article{DISCO-EB:2024,
    author = "Hahn, Oliver and List, Florian and Porqueres, Natalia",
    title = "{DISCO-DJ I: a differentiable Einstein-Boltzmann solver for cosmology}",
    eprint = "2311.03291",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    doi = "10.1088/1475-7516/2024/06/063",
    journal = "JCAP",
    volume = "06",
    pages = "063",
    year = "2024"
}
```

<!-- end:citation -->

## License

<!-- start:license -->

DISCO-EB uses the
[MIT License](https://github.com/cosmo-sims/DISCO-EB/blob/main/LICENSE). The
recombination module adapts code from
[Recfast](https://www.astro.ubc.ca/people/scott/recfast.html), whose UBC
license its file header reproduces.

<!-- end:license -->
