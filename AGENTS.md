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

# The entry point: P(k) of Planck 2018 flat LCDM, one jitted function
uv run python examples/flat_lcdm_power_spectrum.py

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
numba-enzyme-cuda come from PyPI. The code needs modax 0.4.0 or later:
`modax.solve` is its only entry point and takes the Rodas5P numerics on the
method object (`kernel._launch` passes `method=Rodas5P(lu_precision=...,
gains=PIDGains(pcoeff=0.3, icoeff=0.4))`), a save hook is a plain `(y, t, p)
-> tuple` function whose values come back as `(N, n_save, n_out)`, and
`params` may be a pytree whose leaves `jax.jvp` differentiates one at a
time. There is no `sensitivity=`, `hook_size` or `quantity_hook` any more. numba-enzyme-cuda's CPU-only cap on numba
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
  to PyPI (`disco-eb`, imported as `discoeb`) by trusted publishing. Do not bump `version` by hand.

The docs pages pull the overview, installation, quickstart, development,
citation and license from `README.md` between `<!-- start:... -->` markers,
so edit those there. Keep the prose short, active and plain.

## Code Architecture

### Module Structure

**The interface.** The package itself (`discoeb/__init__.py`) is the API the
CMB code uses. That code lives in
DISCO-CMB (`~/git/DISCO-CMB`, package `discocmb`, split out 2026-11-11 with
the history of `cmb.py` and `lensing.py`), which imports from `discoeb` and nothing
else (it took `spline_interpolation` with it on 2026-11-12); this package never imports
it (`tests/test_boundary.py`). The contract is `solve_eb(LCDM(cosmologies), k,
tau, keep=names, extra=components)`: named quantities evaluated in the kernel at the
save times, the CMB's sources being a `LineOfSightSources` component passed as
`extra`; error weights by state name; differentiable in
`DIFFERENTIABLE_PARAMETERS` (those `differentiable_parameters(model)` keeps)
or in the model's directed inputs passed as `inputs=`: all of them, the
15-wide `DIRECTED_NAMES` superset `directed_inputs` gives a dictionary, or
the first nine every built-in model shares (`N_COMMON_DIRECTED`), which
DISCO-CMB's `compute_cl` passes. Grids read
`background_history`, `density_coefficients` and `conformal_ages`. A CMB
change that seems to need a private name should add to `__init__.py` instead;
changing `__init__.py` means running DISCO-CMB's tests too.

The codebase is organized into modular components in `src/discoeb/`:

1. **component.py** - Right-hand sides from components, by dependency injection
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
   - Cosmological parameters are declared by components too (2026-11-13):
     `Omegam = Parameter()` a cosmology-dictionary key (with a default it may
     be left out), `w_DE_0 = Parameter(-1.0, column=True)` a key the kernel
     reads as a row column, `grhoc = Parameter(value=lambda grhom, Omegam,
     Omegab: ..., directed=True)` a kernel input derived on the host by name
     from other parameters. `directed=True` makes columns `x_nominal` and
     `dir_x` and generates the provider `x = x_nominal + eps * dir_x`.
     `grhov` has `closes_budget=True`, the closure itself staying in
     `background.density_parameters`. Two components of one model may share
     a name only through the same `Parameter` object (`Recombination.amnu =
     MassiveNeutrinos.amnu`). A component's `check` classmethod names the
     keys it validates (`DarkEnergy.check(cls, w_DE_0, w_DE_a)`).
     `declared_parameters(classes)`, `derive_parameters(...)`
   - Host-only tests: `tests/test_component.py`; a species with a parameter
     of its own, added from the test file alone: `tests/test_new_species.py`

2. **species.py, interactions.py, gravity.py** - The Einstein-Boltzmann system
   as components (`eb_model.py` was split into these on 2026-11-12)
   - `gravity.py`: `Background` (`adotoa` from the summed `grho`), `Metric`
     (etak and the Einstein constraints `z`, `sigma`, and `R`, the comoving
     curvature perturbation `-etak/k - aH dgq / (k (grho - grhok + gpres))`,
     CAMB's sign, read by `solve_eb`'s hook and not by the right-hand side),
     `Curvature`, `AdiabaticMode` (the growing-mode quantities `x`, `Rv`, `om`
     the initial conditions read) and `SUMS` (`grho`, `gpres`, `dgrho`, `dgq`,
     `dgpi`, summed over the species a model includes). The density constants
     (`GRHOM_PER_H0SQ`, ...) live here. Parameters: `H0`, `grhom` on
     `Background`, `A_s`, `n_s`, `k_p` on `AdiabaticMode`, `Omegak`, `grhok`
     on `Curvature`; in `species.py` `Omegam`, `grhoc` on CDM, `Omegab`,
     `YHe`, `grhob` on `Baryons`, `Tcmb`, `grhog` on `Photons`, `Neff`,
     `grhornomass` on `MasslessNeutrinos`, `grhov` on both dark energies, the
     `w_DE_*`, `cs2_DE` on `DarkEnergy`, `Nmnu`, `mnu`, `grhor_nu`, `n_mnu`,
     `amnu` on `MassiveNeutrinos`; `tau_reio` on `Recombination`
   - `species.py`: `ColdDarkMatter`, `Baryons`, `Photons` (settings `LMAX_G`,
     `LMAX_POL`; states `theta` = Theta_0..LMAX_G and `e` = E_2..LMAX_POL,
     collisionless part), `MasslessNeutrinos` (`LMAX_NR`),
     `CosmologicalConstant` or `DarkEnergy` (CPL fluid), `MassiveNeutrinos`
     (`LMAX_NU`, `NQMAX`; matrix state `psi[bin, l]`, quadrature nodes as tuple
     constants) and `RadiationStreaming` (CLASS's RSA). The free-streaming tail
     is one helper `stream(...)` shared by the three hierarchies
   - `interactions.py`: `ThomsonScattering` (drag, polarization source,
     damping of every higher multipole). An `Interaction` owns no state
   - Truncations are the components' settings: `DISCO_LMAX_G`,
     `DISCO_LMAX_POL`, `DISCO_LMAX_NR`, `DISCO_LMAX_NU` (default 15, minimum 3)
     and `DISCO_NQMAX` (default 3), read when `species` is imported, or a
     model's `with_settings(LMAX_G=...)`, or rebind `Photons.LMAX_G`
   - Host-only tests: `tests/test_eb_model.py`

3. **models.py** - Cosmological models built from those components
   - `CosmologyModel(name, components, settings)`: the
     component classes in state order and setting overrides; frozen and
     hashable. Models derive from each other with `replace(old, new, name=)`,
     `add(*components, name=)`, `remove(...)` and `with_settings(**kw)`.
     `build(thermal_history=None, extra=())` instantiates the component
     `Model`; `state_names()`, `sparsity()`, `setting(name)`,
     `parameters(extra=())` (the components' and `Recombination`'s
     declarations), `parameter_keys()` and `check(cosmology)`: a key another
     component declares but none of this model's must keep its default
     (`w_DE_0 != -1` needs `DarkEnergy`, `Nmnu > 0` `MassiveNeutrinos`), then
     each component's `check` hook runs (`DarkEnergy` rejects `w = -1`),
     traced values skipped. `resolved()` makes every
     setting explicit at the classes' current values: the kernel caches on
     the resolved model, so rebinding a setting recompiles
   - The models: `LCDM` (flat unless `Omegak`), `LCDM_MNU =
     LCDM.add(MassiveNeutrinos)`, `W0WACDM =
     LCDM.replace(CosmologicalConstant, DarkEnergy)`, `W0WACDM_MNU`.
     `add` appends after `ThomsonScattering`; no sum or state order moves,
     since an interaction contributes to neither
   - `LCDM(cosmologies)` (`CosmologyModel.__call__`) checks each cosmology
     and returns a `CosmologyBatch(model, cosmologies)`, the one argument a
     solve takes for both (2026-11-13; before, `model=` was a separate
     keyword). The dicts are kept as given, so JAX values in them stay
     differentiable; `check` skips traced values.
     Indexing gives a batch of the same model (`batch[0]`, `batch[1:]`).
     The kernel still caches on the model alone. There is no
     `PerturbationLayout` any more (removed 2026-11-12): indices come from
     `model.state_names()` or the joint model's `index`

4. **background.py** - The expansion history
   - `density_parameters(param, xp, model)` (every parameter the model
     declares, derived ones by `derive_parameters`; the `closes_budget`
     parameter is the root of `3 adotoa(a=1)^2 = grhom`, one secant step
     where the rate is linear in it, `CLOSURE_STEPS` = 6 more through
     `_background_solution` for a model with background states, so a scalar
     field's `V0` may close the budget: `tests/test_quintessence.py`), `expansion_rate` / `dtauda` (the
     model's `adotoa` via `Model.evaluate(..., math=jnp)` over
     `background_model(model)`, the model's own species). Without a model
     these use `W0WACDM_MNU`, whose sum equals every built-in model's on the
     cosmologies it accepts (`density_coefficients`, `background_history`,
     `conformal_ages` of bare dicts). The massive-neutrino density ratio
     is `MassiveNeutrinos.rhonu`, the same provider on the host and the
     device (branch-free, `NQ_NU_BACKGROUND` = 8 nodes). `_background_solution`
     gives `(a, tau, states)` on a geometric grid from `A_START` = 1e-9: the
     trapezoid quadrature of `d tau / d a` when the model has no background
     state of its own, else fixed RK4 steps in `ln a` of those states
     (`background_model(model).stages()[0]`, scalars only) beside the time.
     `background_endpoints` vmaps it over a batch for `tau0`, `ln a` at
     `TAU_START`, `z_re` and `start`, the model's own background states at
     `TAU_START`, which `_pack` writes into the initial state
     (`conformal_ages` reads it; the model is a static argument). Nothing is
     cached per batch: the whole preparation is JAX and traces under `jit`
   - The momentum bins (`neutrino_momentum_bins`, cached) live in `util.py`
     beside the Gauss-Laguerre rule

5. **cosmology.py** - The parameter dictionary, its hashable form
   `_ParamCosmology` (`as_cosmology`), batches, and
   `DIFFERENTIABLE_PARAMETERS`. `_columns(cosmologies, keys=model.parameter_keys())`
   stacks a batch into JAX columns `(n_cosmo,)` by key, values traced or
   not, plus any key another component declares that a cosmology gives
   (`w_DE_0` in an LCDM dict); a key the solve reads but the model does not
   differentiate (outside `differentiable` = `differentiable_parameters(model)`:
   `Nmnu`, `YHe`, `cs2_DE`, `tau_reio`, `w_DE_*` without `DarkEnergy`,
   `mnu` without `MassiveNeutrinos`) passes through `_not_differentiable`, a `custom_jvp`
   whose rule raises, so a derivative is never a silent zero.
   A dictionary may leave out a key whose declaration has a default (`w_DE_0`,
   `Omegak`, `Nmnu`, `k_p`, ...). `_ParamCosmology.extra` holds declared
   keys beyond its fixed fields. `CosmologyModel.check` skips traced values

6. **recombination.py** - Recombination history, inside the joint kernel
   - RECFAST's `(x_H, x_He, T_m)`, `ln a` and the optical depth `kappa` are
     five states of every trajectory, the `Recombination` component, a block
     of a modax `Hierarchy` stepping ahead of the perturbations
     (`kernel.joint_kernel`) and publishing `a`, `opacity` and `cs2` to them
     as quintic interpolants. Nothing is tabulated, and nothing names these
     states as the background (see `Model.stages` below)
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

7. **kernel.py** - The joint kernel and the launch
   - Solved by modax's Rodas5P numba-CUDA kernel, one wave mode per thread, for
     a batch of cosmologies at once. `joint_kernel(model, extra)` builds the
     modax `Hierarchy` (recombination block ahead of the perturbations) once
     per resolved model (`lru_cache`, modax's `cache=True` keeps it across
     processes); nothing per batch is compiled or copied, since the thermal
     history is part of the state. `_prepare_solve(model, cosmologies, k)`
     checks the cosmologies against the model (when concrete) and packs the
     batch with `_pack`, one `jax.jit` per (model, k grid, batch size), about
     1.2 s to compile and 3 to 11 ms a call: the background endpoints, the
     parameter rows (`_rows`) and the initial state
     (`Model.initial_state(xp=jnp)`, whose fillers write into a recording
     `_Filled` buffer, nothing in place). k must be concrete: it sets the
     launch's shape and order
   - The two blocks are read off the equations (2026-11-14; before,
     `THERMAL_STATES` and `THERMAL_QUANTITIES` named them).
     `Model.stages(mode=("k",))` runs the sparsity trace with the wave number
     followed like a state: a state whose rate reads neither `k` nor a state
     that does is background. `Model.published()` is the minimum vertex cut
     (by numbers handed over, nearest the perturbations among equals)
     between the background states and what the perturbation rates read:
     `("a", "opacity", "cs2")` for every built-in model, plus a background
     state the perturbations read directly (a scalar field). So a component
     with a state such as `d_t(a) = a` lands in the background block, and
     `thermal_indices(model)` lists every background state
     (`tests/test_background_block.py`)
   - The GPU right-hand side is the model's components plus `Recombination`,
     built by `component.Model.build`: a driver with the tuple-return
     signature modax and Enzyme need, calling one device function per
     provider. Loops over multipoles and runtime indices are fine because
     numba-enzyme optimises the primal with `O3` before Enzyme.
   - The parameter row is generated: `parameter_layout(model, extra)` (cached
     per resolved model) is `(x_nominal of each directed parameter,
     SOLVER_COLUMNS = tau_start, tau_end, k, cosmology_tag, save_index, z_re,
     the other column parameters, eps, dir_x of each directed)`, the directed
     ones in `DIRECTED_NAMES` order, then component order. `grhok`, `w_DE_*`, `cs2_DE`,
     `grhor_nu` and `amnu` are directed (2026-11-13); `amnu` is one
     declaration `MassiveNeutrinos` and `Recombination` share, and
     `Model._collect` lets two components bring the same generated provider. `_rows` and the
     initial state iterate over the declarations. `PARAM_COLUMNS`, `N_PARAM`,
     `DIRECTED_COLUMNS` are `W0WACDM_MNU`'s row (`DEFAULT_LAYOUT`), a superset
     of every built-in model's, for host tests and `make_params`
   - `build_keep_hook(model, names)` is the save hook: a `(y, tau, p) ->
     tuple` function of state elements copied from the state and provided
     quantities such as `R` compiled by `Model.build_function`, cached per
     model and set of names (`_keep_hook`)

8. **sensitivities.py** - `_kept_values`: the hooked launch as a
   `jax.custom_jvp` in a pytree of variables that `to_inputs` maps to the
   layout's directed inputs (`solve_eb` passes the cosmology columns and
   `z_re`). Its rule (`symbolic_zeros=True`) replicates every trajectory once
   per *differentiated* variable, with direction `d inputs / d variable`
   from `jax.jvp` of `to_inputs`, so the cost is the number of parameters
   differentiated (5 for the densities' five), not of directed inputs, and
   an undifferentiated input costs nothing. It runs `jax.jvp` of the launch
   in the parameter row's `eps` column, split into a leaf of its own (`(head, eps,
   tail)`), with `sens_error_control=False`: modax integrates one sensitivity
   column per replica and keeps each quantity's derivative beside it. fp64
   LU; the joint path's primal agrees with the plain fp32-LU one to ~1e-6 of
   scale. At *fixed tau0*: `matter_power_spectrum` adds the end-time
   term `d tau0 / dp` itself. `z_re` stays fixed too: a directed `z_re`
   with the implicit function theorem on the bisection was tried
   (2026-11-13) and missed central differences of `P(k)` in `tau_reio` by
   1.8 percent, so `tau_reio` is held. The rows, initial state and save times are
   explicit arguments of the `custom_jvp`, never closed over, and the rule
   closes over no JAX array: JAX traces it lazily, after the enclosing
   `jit` trace, so a captured tracer breaks `jit(grad(jit(f)))`

9. **solver.py** - `solve_eb(LCDM(cosmologies), k, tau, *, keep=...)` solves
   a batch and records only the named quantities at the given times, so the
   device holds `n_k * n_cosmo * n_tau * len(keep)` numbers and never the
   history. Returns an `EBSolution` (`solution["R"]` is `(n_cosmo, n_k,
   n_tau)` in the caller's order, `solution.model` the model); a time past a
   cosmology's `tau0` holds its state today. Traces under `jit` with every
   cosmology key traced (only k and the names static); `EBSolution` is a
   pytree. `values` is differentiable in `differentiable_parameters(model)`
   of `DIFFERENTIABLE_PARAMETERS` (`Omegam`, `Omegab`, `H0`, `Tcmb`, `Neff`,
   `Omegak`, and `mnu`, `w_DE_0`, `w_DE_a` where the model has the
   component): plain JAX from them to the directed inputs
   (`_directed_inputs`), then the `custom_jvp` of `_kept_values`. `jax.jvp`, `jacfwd`, `grad` all cost one joint launch,
   with or without `jit` (`test_matter_power_spectrum_under_jit`). `test_solve_eb_derivatives` gates the densities-level rule
   against central differences, `test_more_parameters_*` each added
   parameter's `solve_eb` values and `P(k)`. The solve is not smooth in the
   parameters below ~1e-6 of a value even at rtol 1e-7, so central
   differences carry noise falling as 1 / step: for `YHe` the apparent
   bias in `solve_eb` values went 0.2, 0.01, 0.002 at steps 2.5e-4, 2.5e-3,
   1e-2 (fp32 or fp64 rate math alike). The 0.5 to 2 percent thermal bias
   recorded before for `YHe`, `Tcmb`, `Omegab` was measured at small steps
   and may be mostly that noise (`Tcmb` and `Omegab` not re-measured).
   `dP/dYHe`, `dP/dcs2_DE` and `dP/dtau_reio` stay 0.8, 1 and 1.8 percent
   off at every step tried, so those three are held. Also `solve_perturbation_history_batch`,
   `directed_inputs`, `conformal_ages`, `state_names(model)`

10. **power_spectrum.py** - `matter_power_spectrum(k, *, clxc, clxb,
    z, vb, tau0, Omegam, Omegab, A_s, n_s, k_p)`, JAX only, takes arrays and
    no solution or batch: the four quantities today `(..., n_k)` from a
    `solve_eb(..., keep=("clxc", "clxb", "z", "vb"),
    thermo_weight=0.1)` at a last time at or past every
    `tau0`, and the parameters `(...)`. Differentiable like `solve_eb`; it
    adds the end-time term (`clxc' = -k z`, `clxb' = -k (z + vb)` times
    `tau0 - stop_gradient(tau0)`), so `tau0` must come from
    `conformal_ages(batch)`: `solution.tau0` is under `stop_gradient` and
    drops the term silently. `examples/flat_lcdm_power_spectrum.py` is the
    entry point

11. **Jacobian sparsity** (`CosmologyModel.sparsity`; its test is
    `tests/test_eb_sparsity.py`)
   - `model.sparsity()` is `model.build().sparsity()`:
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
     entries at the default truncations against the 238 of the dense core
   - That pattern is all modax needs: it colours it (11 colours, so 12 Enzyme
     sweeps instead of 51 columns; the dense core took 12 and 13) and compiles a
     sparse direct solver for it. There is no hand-written Jacobian and no
     hand-written linear solver any more
   - AMD still peels each hierarchy from its truncated end with no fill; the
     core, no longer a clique, fills in (198 slots of `L + U` at the default
     layout for the 165 entries, 14 undeclared diagonals and 19 of fill,
     against the dense core's 238, which had none). The fill and the
     diagonals have slots of their own in modax, written by no column of `J`

12. **util.py** - the neutrino momentum bins and the generalised
    Gauss-Laguerre rule they use for `nqmax > 5`

### Computation Pipeline

The typical computation flow is:

1. **Pick a model and apply it to cosmological parameters** (`LCDM(params)`,
   `W0WACDM(params)`, ...; Ωm, Ωb, h, As, ns, mν, w0, wa, etc.)

2. **Background endpoints** (`background.background_endpoints`)
   - The conformal-time quadrature of the Friedmann sum gives each
     cosmology's `tau0`, `ln a` at `TAU_START` and the reionization redshift

3. **Thermal history** (`recombination.py`)
   - Not a separate step: RECFAST and the optical depth are the recombination
     block of every trajectory, stepping ahead of the hierarchy

4. **Perturbation evolution** (`kernel.py`, the model's components)
   - Initial conditions from the components' `State(initial=...)`
   - One modax launch over every (wave mode, cosmology) pair, keeping named
     quantities through the save hook (`solver.solve_eb`), which
     `power_spectrum.matter_power_spectrum` turns into `P(k)`

5. **CMB spectrum**: DISCO-CMB, through `eb.solve_eb(LCDM(cosmologies), ...)`

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
  `CosmologyModel.sparsity`)
- The recombination system is not stiff but has piecewise switches; Rodas5P
  spends fewer steps on them than Tsit5

**Testing Philosophy:**
- Tests compare against CLASS (standard cosmology code)
- Regression testing with `pytest-regressions` to track accuracy changes
- Benchmark testing with `pytest-benchmark` to track performance
- CLASS reference spectra are computed on the fly by the tests (classy)
- `tests/test_sweep.py` sweeps a fixed-seed Latin hypercube of 32 cosmologies
  per model (LCDM, W0WACDM, LCDM_MNU) in one launch each: finite P(k), steps
  well under `max_steps`, P(k) against CLASS at gates set from measured errors,
  and a self-convergence check against a 1000x tighter fp64-LU solve. A CPL
  `w(a)` crossing -1 hits `max_steps` and is silently wrong (strict `xfail`).
  The self-convergence tests are marked `slow`, which `pyproject.toml`
  deselects by default: run them with `uv run pytest tests -m slow -rs`

## Example Notebooks

See `notebooks/` directory:
- `nb_minimal_example.ipynb` - Start here: compute matter power spectrum and derivatives

The notebooks still call the removed diffrax `evolve_perturbations*` functions;
they need porting to `solve_eb` and `matter_power_spectrum`. The CMB notebook moved
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
