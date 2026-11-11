# AGENTS.md

This file provides guidance to coding agents working in this repository.

## Project Overview

DISCO-EB is a differentiable linear Einstein-Boltzmann solver for cosmology implemented in JAX. It is part of the DISCO-DJ framework (DIfferentiable Simulations for COsmology, Done with Jax). The code solves the coupled Einstein-Boltzmann equations to compute the linear matter power spectrum, its derivatives, and the unlensed CMB temperature spectrum, with the perturbation and recombination solves running as modax numba-CUDA kernels beside JAX.

**Key Features:**
- Autodifferentiable via JAX for efficient Fisher forecasts and parameter estimation
- Standard ΛCDM model with Quintessence dark energy (w0, wa) and massive neutrinos
- Thermal history solver based on simplified Recfast implementation
- GPU-accelerated for performance (tests should be run on GPU platforms)

**Citation:** Hahn et al. 2024 [arXiv:2311.03291](https://arxiv.org/abs/2311.03291)

## Commands

```bash
# Install the project environment (uses uv). The CUDA extra gives JAX's GPU
# build; a plain `uv sync` removes it again.
uv sync --extra cuda13  # or --extra cuda12

# Run the tests (they need a GPU). -rs lists skips: without `classy` (dev
# group) the CLASS comparisons skip after their benchmark round, which reads
# like a pass in -q output.
uv run pytest tests -rs
uv run pytest "tests/test_perturbations.py::test_matter_power_spectrum_matches_class[N128]" -v

# Run with coverage, as CI does
uv run pytest tests --cov --cov-report=term-missing

# Format and lint
uv run ruff format
uv run ruff check --fix

# Type check (scoped to src/discoeb by [tool.ty.src] in pyproject.toml)
uv run ty check

# Install the pre-commit hooks once; they then run the three checks above on
# every commit, and conventional-pre-commit checks every commit message
# against Conventional Commits (`feat: ...`, `fix(scope): ...`, `perf: ...`).
uv run pre-commit install
uv run pre-commit run --all-files

# Docs (Sphinx + MyST + autodoc, sphinx-book-theme), in their OWN environment:
# the project with its CPU dependencies plus the docs group.
export UV_PROJECT_ENVIRONMENT=.venv-docs
uv sync --no-default-groups --group docs
uv run --no-sync sphinx-autobuild docs docs/_build/html  # live preview
uv run --no-sync sphinx-build -W docs docs/_build/html   # as CI builds it
```

Requires Python >= 3.10 (`.python-version` pins 3.13). modax-solvers and
numba-enzyme-cuda come from PyPI; modax is capped below 0.2, whose `solve`
no longer takes the `lu_precision`, `pcoeff` and `icoeff` keywords
`perturbations._launch` passes. numba-enzyme-cuda's CPU-only cap on numba
and llvmlite is lifted by `override-dependencies`, since only the CUDA
backend is used. To develop against a modax checkout, `uv pip install
--no-deps -e ../modax`; never replace `site-packages/numba_enzyme` by a
symlink into a checkout, since a later sync uninstalls through it.

Tests use `pytest-benchmark`. The spectrum benchmarks time a *fresh* batch
after warming up on a distinct one, so the thermal history is included, as in
an inference iteration. `--benchmark-disable` makes `benchmark.pedantic` run
only the warm-up batch, which breaks those tests.

## CI

Five workflows, all in `.github/workflows/`, structured as in modax:

- **`lint.yml`**: `ruff format --check` and `ruff check` on every push to
  `main` and every pull request. The ruff version is pinned in the
  workflow's `env` and as the `rev` in `.pre-commit-config.yaml`; bump the
  two together.
- **`types.yml`**: `ty check` with the project installed without the dev
  group, ty from `uvx`, pinned like ruff.
- **`tests.yml`**: `pytest --cov` on the runner named by the repository
  variable `GPU_RUNNER`. Unset, the job falls back to `ubuntu-latest`, where
  `conftest.py` skips every test that launches a kernel. On a push to `main`
  it writes the coverage percentage to the gist named by `COVERAGE_GIST_ID`
  (modax's gist, file `discoeb-coverage.json`) with the secret `GIST_TOKEN`.
- **`docs.yml`**: builds `docs/` with `sphinx-build -W` and publishes it to
  GitHub Pages (https://cosmo-sims.github.io/DISCO-EB/).
- **`publish.yml`**: python-semantic-release reads the Conventional Commits
  since the last `v*` tag; a `feat`, `fix`, `perf` or breaking change bumps
  `version`, updates `CHANGELOG.md` and `uv.lock`, tags, builds and uploads
  to PyPI (`discoeb`) by trusted publishing. Do not bump `version` by hand.

The docs pages pull the overview, installation, quickstart, development,
citation and license from `README.md` between `<!-- start:... -->` markers,
so edit those there. Keep the prose short, active and plain.

## Code Architecture

### Module Structure

**The interface.** The package itself (`discoeb/__init__.py`) is the API the
CMB code uses. That code lives in
DISCO-CMB (`~/git/DISCO-CMB`, package `discocmb`, split out 2026-11-11 with
the history of `cmb.py` and `lensing.py`), which imports from `discoeb` and the
generic `spline_interpolation` and nothing else; this package never imports
it (`tests/test_boundary.py`). The contract is `solve_eb(cosmologies, k, tau,
keep=names, extra=components)`: named quantities evaluated in the kernel at the
save times, the CMB's sources being a `LineOfSightSources` component passed as
`extra`; error weights by state name; differentiable in
`DIFFERENTIABLE_PARAMETERS` or in the nine `DIRECTED_NAMES` inputs passed as
`inputs=` (`directed_inputs` gives a cosmology's). Grids read
`background_history`, `density_coefficients` and `conformal_ages`. A CMB
change that seems to need a private name should add to `__init__.py` instead;
changing `__init__.py` means running DISCO-CMB's tests too.

The codebase is organized into modular components in `src/discoeb/`:

1. **The background** lives in `eb_model.py` (`background.py` folded in on
   2026-10-21): the component model's own Friedmann sum evaluated on JAX
   arrays. `density_parameters(param, xp)` (the `grho*` coefficients, the
   budget closed at `a = 1` with the model's `grho`), `expansion_rate` /
   `dtauda` (the model's `adotoa` via `Model.evaluate(..., math=jnp)`), the
   massive-neutrino momentum bins (`neutrino_momentum_bins`, cached) and the
   shared `rho_nu(y)` table (`neutrino_density_ratio`). The conformal-time
   quadrature is `perturbations._background_tau`; `background_endpoints`
   vmaps it over a batch for `tau0`, `ln a` at `TAU_START` and `z_re`

2. **recombination.py** - Recombination history, inside the joint kernel
   - RECFAST's `(x_H, x_He, T_m)`, `ln a` and the optical depth `kappa` are
     five states of every trajectory, the `Recombination` component, a block
     of a modax `Hierarchy` stepping ahead of the perturbations
     (`perturbations.joint_kernel`) and publishing `a`, `opacity`, `cs2`,
     `rhonu` (and `pnu`) to them as quintic interpolants. Nothing is tabulated
   - The stand-alone RECFAST solve in `ln a` (`thermodynamics_recfast.py`,
     `thermal_history_batch`, `evolve_background`) was removed on 2026-11-11:
     it served only tests. `tests/test_thermo.py` reads the kernel's block
     through `solve_eb(..., keep=("a", "xe", "Tmat", "cs2", "opacity",
     "kappa"))` at one wave mode (`kernel_thermal_history`): `x_e` within
     0.36 percent of CLASS recfast, `T_m` 0.04, `c_b^2` 0.4 (9.7 in the
     tight-coupling switch window near z = 850)
   - The rates' transcendentals run in fp32 by default (`FP32_MATH`,
     `DISCO_THERMO_FP32_MATH`); `a` and the reionization opacity stay fp64
   - Reionization: a cosmology's `tau_reio` (default 0, none) adds CAMB's tanh
     step to `x_e` (H and HeII together, width 0.5 in `(1+z)^1.5`; HeIII at
     z = 3.5), with `z_re` bisected on the optical depth on the background
     (`_reionization_redshift`, residual recombination ionization neglected:
     2e-4 of `tau_reio`). Same parametrization as CLASS `reio_camb`
     (`tests/test_reionization.py`)
   - Hydrogen leaves Saha equilibrium on *redshift*, not on `x_H`: Saha slope
     above `Z_H_SAHA_END` = 1600, the rate equation below, the two slopes
     blended over 2 x `DZ_H_SAHA_BLEND` = 50 in z (CLASS 3's
     `recfast_z_full_H_recombination` and its smoothing). RECFAST's switch at
     `x_H < 0.99` is a jump in the Jacobian, and the Rosenbrock step straddling
     it overshot `x_e` by up to 5 percent
   - The RECFAST constants live here too. **Recfast license** (UBC, not MIT):
     see the file header

3. **perturbations.py** - Perturbation evolution (core module)
   - Solves synchronous gauge Einstein-Boltzmann equations
   - Handles CDM, baryon, photon, and neutrino perturbations
   - Implements multipole hierarchies for radiation (photons, neutrinos)
   - Massive neutrino perturbations integrated over momentum bins
   - Solved by modax's Rodas5P numba-CUDA kernel, one wave mode per thread, for
     a batch of cosmologies at once (`solve_matter_power_spectrum_batch`,
     `matter_power_spectrum_jax` for derivatives)
   - The kernel is compiled once per layout (`joint_kernel` is an
     `lru_cache`, modax's `cache=True` keeps it across processes); nothing
     per batch is compiled or copied, since the thermal history is part of
     the state. `_prepare_solve` caches a batch's endpoints
     (`BATCH_CACHE_SIZE` batches)
   - The GPU right-hand side is the component model of `eb_model.py` (plus
     `Recombination`) built by `component.Model.build`: a driver with the
     tuple-return signature modax and Enzyme need, calling one device function
     per provider. Loops over multipoles and runtime indices are fine because
     numba-enzyme optimises the primal with `O3` before Enzyme (unrolling them
     and folding the bounds-check atomics Enzyme used to refuse).
     `PARAM_COLUMNS` names the parameter row's columns
   - `solve_eb(cosmologies, k_values, tau, keep=...)` solves a batch and
     records only the named quantities at the given times through modax's
     save hook (`build_keep_hook`: state elements copied from the state,
     provided quantities such as the comoving curvature perturbation `R`
     compiled by `Model.build_function`), so the device holds
     `n_k * n_cosmo * n_tau * len(keep)` numbers and never the history. Returns
     an `EBSolution` (`solution["R"]` is `(n_cosmo, n_k, n_tau)` in the
     caller's order); a time past a cosmology's `tau0` holds its state today.
     The hook is cached per layout and set of names (`_keep_hook`). `values` is differentiable
     in the dictionary's `DIFFERENTIABLE_PARAMETERS` (`Omegam`, `Omegab`,
     `H0`, `Tcmb`, `Neff`, given as JAX values; the other keys must be plain
     numbers): a `custom_jvp` in those parameters whose rule, with concrete
     primals, prepares the batch and chains `density_parameters` (JAX)
     into the inputs-level rule `_kept_values`, which runs the
     one-column-per-replica sensitivity launch with the
     *tangent* hook (`build_keep_hook(..., tangent=True)`: Enzyme's forward
     sweep of the compiled quantities seeded with `(S, 0, e_eps)`, state
     elements' tangents read from `S`), through modax's `solve(...,
     sensitivity=SensitivitySpec)` (joint `[y, S]` launch as the primal, hook
     sees the augmented state; modax be0a298, 2026-11-05). fp64 LU,
     sensitivities out of step control; the joint path's primal agrees with
     the plain fp32-LU one to ~1e-6 of scale. At *fixed tau0* (unless
     `tau_end_of` is given, as `matter_power_spectrum_jax` does: its value
     stops at the inputs' own `tau0` and an explicit end-time term carries
     `d tau0 / d densities`; it shares this launch path since 2026-11-11). `jax.jvp`, `jacfwd`, `grad` all
     cost one joint launch (0.33 s against 0.12 s plain for 8 modes); not
     under `jit`. `test_solve_eb_derivatives` gates the densities-level rule
     against central differences and the parameter-level one as its chain rule.
     Example: `examples/solve_eb_comoving_curvature.py`
   - Hierarchy truncations are the components' settings (`eb_model.py`):
     `DISCO_LMAX_G`, `DISCO_LMAX_POL`, `DISCO_LMAX_NR`, `DISCO_LMAX_NU`
     (default 15, minimum 3) and `DISCO_NQMAX` (default 3), read when
     `eb_model` is imported, or rebind `Photons.LMAX_G` etc. before the first
     solve

7. **component.py** - Right-hand sides from components, by dependency injection
   - A `Component` declares its settings (`LMAX_G = Setting(15)`, loaded from
     `DISCO_LMAX_G` when the class is created, overridable as a constructor
     argument) and its state as class attributes (`etak = State()`,
     `theta = State(lambda self: self.LMAX_G + 1, tail_from=3)`), and provides
     quantities through ordinary methods whose *parameter names after `self`
     are the quantities they depend on* (a method or lambda that reads nothing
     of its component may leave `self` out; `@staticmethod` is accepted).
     `d_<state>` provides a derivative contribution: a scalar state's is
     returned; a vector state's is written into `output`, which the framework
     hands the provider filled with zeros and then sums into the derivative,
     so several components may write one state. A method named in the model's
     `sums` contributes to that sum. Wherever a function fills an array
     instead of returning, the array is its parameter `output`
   - A `State` also carries its initial condition (`State(..., initial=fn)`,
     a lambda for a scalar or a `_`-prefixed function defined just above for
     a vector, resolved by name like a provider from `k`,
     `tau`, the background densities, other states' initial values and
     host-evaluable providers; `Model.initial_state`). The Jacobian's
     sparsity pattern is not declared: `Model.sparsity()` traces it through
     the providers (2026-10-27; until then the states declared a dense core
     and `tail_from` tails). It resolves the same dependency graph as a build
     through `_HostValues`, on `_Traced` values that carry the set of state
     elements they depend on -- a state element its own index, the time and
     the parameters nothing, every operator and `math` function the union --
     element by element through a vector derivative's `output`, so `stream`
     comes out tridiagonal with no one saying so. A provider none of whose
     inputs depends on the state is not called (a table lookup, a `max`
     floor, `int()` of a tag never see a traced value); a branch on a traced
     value raises, with the advice to move the branching quantity into a
     provider of inputs alone. The layout, the initial state and the
     Jacobian pattern are all read off the components
   - A provider reads its settings and the other public numeric attributes
     of its instance (a tuple of quadrature weights) as `self.NQMAX`,
     `self.weight[i]`. On the device `self` is `Component.device_self()`, a
     named tuple of those constants passed by the generated driver as a
     global constant (`Photons_self`): numba types each field (int, float,
     tuple of floats) and O3 folds them, so `range(self.NQMAX)` unrolls like a
     literal. A `types.ModuleType` would also work for scalars but
     numba-cuda-mlir cannot lower a tuple attribute of a module; a local
     array's size must still be a literal (`cuda.local.array(16, float64)`),
     since numba types it before anything folds. `cuda` and `float64` from
     `component.py` are placeholders for scratch space, replaced at compile
     time or by host stand-ins in tests; the physics needs neither
   - `Model.build(params)` reads dependencies off `inspect.signature`, orders
     providers with `graphlib.TopologicalSorter`, compiles each as its own
     `cuda.jit(device=True)` function with its globals rebound (device names,
     `math`, plain helpers such as `stream` compiled for it) and
     generates a thin driver: `y` into per-state local arrays, zeroed
     contribution buffers, the provider calls in order, the sums, the
     returned tuple. `Model.build_function(outputs, inputs=, params=)`
     generates a device function for any quantities from any base inputs
     (an input shadows a provider of the same name), used for the
     thermodynamics kernel's Hubble rate and the CMB sources;
     `Model.evaluate(..., math=jnp)` is the same resolution on the host over
     floats or JAX arrays and `Model.state_values(y)` names a flat state
     array's states. No `ast`, tracing or templates: loops over multipoles
     are ordinary loops, which numba-enzyme's pre-Enzyme `O3` pipeline unrolls
     (see modax/numba-enzyme). No external DI library: python-dependency-injector
     and its kin wire runtime object graphs from explicit declarations, not
     build-time code from parameter names
   - Host-only tests: `tests/test_component.py`

8. **eb_model.py** - The Einstein-Boltzmann system as components
   - `Metric` (etak and the Einstein constraints `z`, `sigma`), `Background`
     (`adotoa` from the summed `grho`), `AdiabaticMode` (the growing-mode
     quantities `x`, `Rv`, `om` the initial conditions read), `Curvature`,
     `ColdDarkMatter`, `Baryons`, `Photons` (settings `LMAX_G`, `LMAX_POL`;
     states `theta` = Theta_0..LMAX_G and `e` = E_2..LMAX_POL, collisionless
     part), `ThomsonScattering` (drag, polarization source, damping of every
     higher multipole), `MasslessNeutrinos` (`LMAX_NR`), `CosmologicalConstant`
     or `DarkEnergy` (CPL fluid), `MassiveNeutrinos` (`LMAX_NU`, `NQMAX`;
     matrix state `psi[bin, l]`, quadrature nodes as tuple constants). The
     free-streaming tail is one helper `stream(kv, x, inv_tau, output)`
     shared by the three hierarchies. The sums `grho`, `gpres` (CAMB's
     pressure sum, no curvature term), `dgrho`, `dgq`, `dgpi` run over the
     species a layout enables; `Metric.R` is the comoving curvature
     perturbation `-etak/k - aH dgq / (k (grho - grhok + gpres))`, CAMB's
     sign (unity for the adiabatic mode), read by `solve_eb`'s hook and not
     by the right-hand side. `perturbation_model(layout,
     thermal_history=None, extra=())` passes the layout's truncations to the
     components as settings and appends `extra` components (the CMB sources);
     `perturbations.py` takes its `LMAX_*`/`NQMAX` module
     constants, `_layout_for`, `make_initial_states` and the Jacobian pattern
     (`perturbation_sparsity`) from the model, so rebinding `Photons.LMAX_G`
     before the first solve changes the layout. `layout_state_names` is the
     state order by name. Host-only tests: `tests/test_eb_model.py`

9. **Jacobian sparsity** (`eb_model.perturbation_sparsity`; `eb_sparsity.py`
   was folded into it on 2026-10-21, its test kept as `tests/test_eb_sparsity.py`)
   - `perturbation_sparsity(layout)` is `perturbation_model(layout).sparsity()`:
     the exact pattern, traced through the providers (see `component.py`
     above). `test_the_pattern_is_the_jacobians` checks it against a
     finite-difference Jacobian of the host-built right-hand side, with the
     thermal channels of order one: at a realistic `cs2 ~ 1e-8` or `a ~ 1e-3`
     the baryon pressure and dark-energy couplings fall below rounding
   - The pattern has no diagonal on the 14 undamped rows (`n_1`, `n_3..n_14`)
     and needs modax at or after commit 9f54d12 (2026-11-01): before it, the
     packed `CompressedJacobian.slot(i, i)` went through the `(row, colour)`
     grid, and where another column of colour `colour[i]` had an entry in
     row `i` the `1/(h gamma)` landed on that entry (12 of the 14 rows) and
     the solve returned the initial state (P(k) ~ 1e-20 of CLASS). The
     diagonal now always comes from the factorisation's own slot table
     (`test_undeclared_diagonals_get_the_factorisations_slots`)
   - What the trace finds: a core (metric + fluids + the lowest photon /
     polarization / neutrino multipoles, plus the dark-energy fluid and the
     lowest massive-neutrino multipoles when enabled) that is far from dense --
     the metric reads the densities and momenta through `dgrho` and `dgq`, the
     quadrupoles read `sigma`, a free-streaming multipole reads its two
     neighbours and itself only where Thomson scattering damps it (the neutrino
     tail has no diagonal) -- bordered by tridiagonal free-streaming
     hierarchies, each meeting the core at its lowest multipole only. 165
     entries at the default layout against the 238 of the dense core
   - That pattern is all modax needs: it colours it (11 colours, so 12 Enzyme
     sweeps instead of 51 columns; the dense core took 12 and 13) and compiles a
     sparse direct solver for it. There is no hand-written Jacobian and no
     hand-written linear solver any more
   - AMD still peels each hierarchy from its truncated end with no fill; the
     core, no longer a clique, fills in (198 slots of `L + U` at the default
     layout for the 165 entries, 14 undeclared diagonals and 19 of fill,
     against the dense core's 238, which had none). The fill and the
     diagonals have slots of their own in modax, written by no column of `J`

10. **spline_interpolation.py** - Cubic splines, construct and `evaluate` only
   - `spline_interpolation(x, y)`: natural cubic spline on any grid, batched
     over `y`'s columns; DISCO-CMB's transfer and lensing interpolation
   - `FastUniformCubicSpline1D`: natural cubic spline on a uniform grid stored
     as Horner coefficients in the local coordinate; `evaluate` takes a
     scalar (`jax.vmap` over arrays). The `rho_nu(y)` table
   - The integral and derivative methods went with the stand-alone RECFAST
     path on 2026-11-11 (the non-uniform `integral` had a bug: `y` in place of
     the second derivative in its linear coefficient)

11. **util.py** - the generalised Gauss-Laguerre rule the neutrino momentum
    bins use for `nqmax > 5`

### Computation Pipeline

The typical computation flow is:

1. **Define cosmological parameters** (Ωm, Ωb, h, As, ns, mν, w0, wa, etc.)

2. **Background endpoints** (`perturbations.background_endpoints`)
   - The conformal-time quadrature of the model's Friedmann sum gives each
     cosmology's `tau0`, `ln a` at `TAU_START` and the reionization redshift

3. **Thermal history** (`recombination.py`)
   - Not a separate step: RECFAST and the optical depth are the recombination
     block of every trajectory, stepping ahead of the hierarchy

4. **Perturbation evolution** (`perturbations.py`, `eb_model.py`)
   - Set up initial conditions for perturbations at early times
   - Build the right-hand side from the layout's components and solve the
     coupled ODEs for each k-mode (or batched)
   - Output multipole hierarchies at specified output times/scale factors

5. **CMB spectrum**: DISCO-CMB, through `eb.solve_eb`

### Key Design Patterns

**JAX-First Design:**
- All code is JAX-compatible (no NumPy unless wrapped)
- Extensive use of `jax.numpy` instead of `numpy`
- Compatible with `jax.jit`, `jax.grad`, `jax.vmap` for performance and autodiff
- Enable 64-bit precision: `jax.config.update("jax_enable_x64", True)` (see conftest.py)

**Cosmologies:**
- A cosmology is a parameter dictionary (`Omegam`, `Omegab`, `H0`, `Tcmb`,
  `YHe`, `Neff`, `A_s`, `n_s`, ...) or the hashable object `as_cosmology`
  returns; `_as_batch` takes one or a sequence

**Neutrino Treatment:**
- Massive neutrinos handled via momentum binning (nqmax bins)
- Gauss-Laguerre quadrature for momentum integration
- Perturbations tracked separately for each momentum bin (psi0, psi1, psi2, ...)

**Stiff ODE Integration:**
- Einstein-Boltzmann equations are stiff; modax's Rodas5P kernel integrates them
  with an Enzyme-derived Jacobian and a sparse direct solver (see
  `eb_model.perturbation_sparsity`)
- The recombination system is not stiff but has piecewise switches; Rodas5P
  spends fewer steps on them than Tsit5

**Testing Philosophy:**
- Tests compare against CLASS (standard cosmology code)
- Regression testing with `pytest-regressions` to track accuracy changes
- Benchmark testing with `pytest-benchmark` to track performance
- CLASS reference spectra are computed on the fly by the tests (classy)

## Example Notebooks

See `notebooks/` directory:
- `nb_minimal_example.ipynb` - Start here: compute matter power spectrum and derivatives

The notebooks still call the removed diffrax `evolve_perturbations*` functions;
they need porting to `solve_matter_power_spectrum_batch`. The CMB notebook moved
to DISCO-CMB.

## Contributing

- Fork from `main`
- MIT license; `recombination.py` also carries the Recfast (UBC) license
- When adding tests, use `pytest-regressions` for accuracy tracking
- Update documentation if changing APIs or config options
- Discuss major changes via issue or email first (see CONTRIBUTING.md)

## Important Notes

- **GPU Required:** Tests expect GPU platform for reasonable performance
- **JAX 64-bit:** Tests enable 64-bit precision via `jax.config.update("jax_enable_x64", True)`
- **Recfast License:** `recombination.py` contains code adapted from Recfast (UBC license, not MIT)
- **Benchmarks:** See BENCHMARKS.md for reference timings on various hardware.
  The spectrum benchmarks (`test_power_spectra_match_class`,
  `test_matter_power_spectrum_matches_class` likewise if changed) time a *fresh*
  batch after warming up on a distinct one: the thermodynamics is included, as
  in an inference iteration
- **Parameter Units:** Neutrino masses in eV, wave numbers in 1/Mpc, conformal time in Mpc, scale factor dimensionless
