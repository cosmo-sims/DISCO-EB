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
uv run pytest "tests/observables/test_power_spectrum.py::test_matter_power_spectrum_matches_class[N128]" -v

# The entry point: P(k) of 128 flat LCDM cosmologies about Planck 2018, built,
# solved and timed in one jitted function
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
  `conftest.py` skips every test marked `gpu`. On a push to `main`
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
it (`tests/test_init.py`). The contract is `solve_eb(cosmologies, k,
tau, keep=names, extra=components)`, the cosmologies instances of one model
class (`[LCDM(H0=..., ...), ...]`): named quantities evaluated in the kernel at the
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

The codebase is organized in four folders under `src/discoeb/` (2026-11-16),
which import each other in one direction, `components` <- `models` <-
`solver`, with `observables` importing none: `components/` (the framework,
species, gravity, interactions, recombination), `models/` (the model classes
and cosmologies), `solver/` (background, kernel, sensitivities, `solve_eb`)
and `observables/` (the matter power spectrum). Each folder's `__init__.py`
re-exports its public names, and a module with a leading underscore is
private: import its public names from the folder (`from discoeb.components
import Component, State`, `from discoeb.solver import solve_eb`). The modules:

1. **components/_component.py** - Right-hand sides from components, by dependency injection
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
     `sums` contributes to that sum. Since 2026-11-15 a contribution to a sum,
     or to another component's state, has a name of its own and says what it
     adds to: `@contributes("dgrho") def dgrho_b(...)` (`_contributed`; the
     suffix names the species: `_c`, `_b`, `_g`, `_r`, `_nu`, `_de`,
     `_lambda`, `_fluid`, `_thomson`). A model class inherits every
     component, and shared method names with different signatures are what
     ty and pyright report as incompatible bases. A plain-named contribution
     still works at run time. Wherever a function fills an array
     instead of returning, the array is its parameter `output`
   - A `State` also carries its initial condition (`State(..., initial=fn)`,
     a lambda for a scalar or a `_`-prefixed function defined just above for
     a vector, resolved by name like a provider from `k`,
     `tau`, the background densities, other states' initial values and
     host-evaluable providers; `Equations.initial_state`). The Jacobian's
     sparsity pattern is not declared: `Equations.sparsity()` traces it through
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
   - `Equations.build(params)` reads dependencies off `inspect.signature`, orders
     providers with `graphlib.TopologicalSorter`, compiles each as its own
     `cuda.jit(device=True)` function with its globals rebound (device names,
     `math`, plain helpers such as `stream` compiled for it) and
     generates a thin driver: `y` into per-state local arrays, zeroed
     contribution buffers, the provider calls in order, the sums, the
     returned tuple. `Equations.build_function(outputs, inputs=, params=)`
     generates a device function for any quantities from any base inputs
     (an input shadows a provider of the same name), used for the
     thermodynamics kernel's Hubble rate and the CMB sources;
     `Equations.evaluate(..., math=jnp)` is the same resolution on the host over
     floats or JAX arrays and `Equations.state_values(y)` names a flat state
     array's states. No `ast`, tracing or templates: loops over multipoles
     are ordinary loops, which numba-enzyme's pre-Enzyme `O3` pipeline unrolls
     (see modax/numba-enzyme). No external DI library: python-dependency-injector
     and its kin wire runtime object graphs from explicit declarations, not
     build-time code from parameter names
   - Cosmological parameters are declared by components too (2026-11-13),
     annotated since 2026-11-15: `Omegam: ArrayLike = Parameter()` a key, a
     constructor argument of every model with the component (with a default
     it may be left out), `w_DE_0: ArrayLike = Parameter(-1.0, column=True)`
     a key the kernel reads as a row column, `grhoc: ArrayLike =
     Derived(lambda grhom, Omegam, Omegab: ..., directed=True)` a kernel
     input derived on the host by name from other parameters. `Parameter`
     and `Derived` are `dataclass_transform` field specifiers returning a
     `ParameterDeclaration`: `Derived`'s `init: Literal[False]` default keeps
     it out of the signature a type checker sees, and `__set_name__` raises
     on an unannotated declaration, which no type checker would see.
     `directed=True` makes columns `x_nominal` and
     `dir_x` and generates the provider `x = x_nominal + eps * dir_x`.
     `grhov` is `Derived(closes_budget=True)`, the closure itself staying in
     `background.density_parameters`. Two components of one model may share
     a name only through the same declaration (`grhov` on the
     `_DarkEnergyDensity` base of both dark energies). A component's `check` classmethod names the
     keys it validates (`DarkEnergy.check(cls, w_DE_0, w_DE_a)`).
     `declared_parameters(classes)`, `derive_parameters(...)`
   - `Component.named("x")` (2026-11-14) is a cached subclass whose states,
     parameters and quantities carry the prefix `x_` (`_renames`), so a
     model may list one component several times: `LCDM.add(Fluid.named("dr"),
     Fluid.named("idm"))`. The functions are the class's own. `_Provider`,
     a state's `initial`, a `Parameter`'s `reads` and the `check` hook
     translate each signature name the component itself declares
     (`_model_name`); other names and sum contributions stay as written, so
     a named component cannot read a foreign quantity that shares one of its
     own names. Settings are shared. `species.Fluid` (constant `w`, `cs2`)
     is meant to be used this way: `Fluid.named` in `tests/helpers.py`
   - Host-only tests: `tests/components/test_component.py`; a species with a
     parameter of its own, added from the tests alone: `SecondDarkMatter` in
     `tests/helpers.py`

2. **components/species.py, interactions.py, gravity.py** - The Einstein-Boltzmann system
   as components (`eb_model.py` was split into these on 2026-11-12)
   - `gravity.py`: `Kinematics` (`a2`, `inv_a`, `inv_a2`, `inv_tau`,
     `inv_k2`), `Background` (the Friedmann equation: `adotoa = sqrt(grho/3 -
     Kcurv)` and `adotoa_prime = -(grho + 3 gpres)/6`), `Metric`
     (etak and the Einstein constraints `z`, `sigma`, and `R`, the comoving
     curvature perturbation `-etak/k - aH dgq / (k (grho + gpres))`,
     CAMB's sign, read by `solve_eb`'s hook and not by the right-hand side),
     `Curvature` (geometry: `Kcurv`, `s2`; in no sum since 2026-11-14),
     `AdiabaticMode` (the growing-mode quantities `x`, `Rv`, `om`
     the initial conditions read) and `SUMS` (`grho`, `gpres`, `dgrho`, `dgq`,
     `dgpi`, summed over the species a model includes, and `grho_matter`,
     `grho_radiation`, `grho_free_streaming`, from which `Rv` and `om` are
     ratios with `a` cancelling, so a new species enters the initial
     conditions by contributing its density; `MassiveNeutrinos` and `Fluid`
     contribute to none, as before 2026-11-14 when the four densities were
     named). The density constants
     (`GRHOM_PER_H0SQ`, ...) live here. Parameters: `H0`, `grhom` on
     `Background`, `A_s`, `n_s`, `k_p` on `AdiabaticMode`, `Omegak`, `grhok`
     on `Curvature`; in `species.py` `Omegam`, `grhoc` on CDM, `Omegab`,
     `YHe`, `grhob` on `Baryons`, `Tcmb`, `grhog` on `Photons`, `Neff`,
     `grhornomass` on `MasslessNeutrinos`, `grhov` on both dark energies, the
     `w_DE_*`, `cs2_DE` on `DarkEnergy`, `Nmnu`, `mnu`, `grhor_nu`, `n_mnu`,
     `amnu` on `MassiveNeutrinos`; `tau_reio` on `Recombination`
   - The gravity interface (2026-11-14): species contribute to `SUMS`
     (fluids only) and read `GRAVITY_INTERFACE` = `adotoa`, `z`, `sigma`,
     which `component.Gravity` subclasses (`Background`, `Metric`,
     `Curvature`) provide. `ModelSpec` raises if a model provides none
     of one. Another theory is `LCDM.swap(Background, Other, name=)` or
     `swap(Metric, Other, name=)`; the closure follows since it solves
     `adotoa(a=1)` (`tests/components/test_gravity.py`). A scalar state's rate is a
     quantity `d_<state>` any provider or hook may read (`_collect` adds it
     to the contributions; the driver then computes it once)
   - `species.py`: `ColdDarkMatter`, `Baryons`, `Photons` (settings `LMAX_G`,
     `LMAX_POL`; states `theta` = Theta_0..LMAX_G and `e` = E_2..LMAX_POL,
     collisionless part), `MasslessNeutrinos` (`LMAX_NR`),
     `CosmologicalConstant` or `DarkEnergy` (CPL fluid), `Fluid` (constant
     `w` and `cs2`, no shear, listed under a name), `MassiveNeutrinos`
     (`LMAX_NU`, `NQMAX`; matrix state `psi[bin, l]`, quadrature nodes as tuple
     constants) and `RadiationStreaming` (CLASS's RSA). The free-streaming tail
     is one helper `stream(...)` shared by the three hierarchies
   - Rules a model keeps (2026-11-14): `ColdDarkMatter.fixes_gauge` states
     the synchronous gauge's condition and `ModelSpec` raises without
     such a species. `RadiationStreaming` names no species: each streaming
     one contributes to the sum `dgrho_streaming`, and
     `NoRadiationStreaming` (`rsa = 0`) replaces it for a theory whose
     constraints the streaming solution does not know
   - `interactions.py`: `ThomsonScattering` (drag, polarization source,
     damping of every higher multipole). An `Interaction` owns no state
   - Truncations are the components' settings: `DISCO_LMAX_G`,
     `DISCO_LMAX_POL`, `DISCO_LMAX_NR`, `DISCO_LMAX_NU` (default 15, minimum 3)
     and `DISCO_NQMAX` (default 3), read when `species` is imported, or a
     model's `with_settings(LMAX_G=...)`, or rebind `Photons.LMAX_G`
   - Host-only tests: `tests/components/test_species.py`,
     `tests/models/test_model.py`

3. **models/_model.py, lcdm.py, w0wacdm.py** - Cosmological models built from those components
   - Split on 2026-11-16: `_model.py` has the base classes (`Model`,
     `ModelSpec`, `CosmologyBatch`, `_batch_of`, `_spec_of`), `lcdm.py`
     `LCDM` and `LCDM_MNU`, `w0wacdm.py` `W0WACDM` and `W0WACDM_MNU`, and
     `models/__init__.py` exports them and defines `DEFAULT_MODEL` and
     `DEFAULT_SPEC`
   - Since 2026-11-15 a model is a class and a cosmology an instance of it:
     `class LCDM(Model, Kinematics, Background, ..., ThomsonScattering)`,
     `class LCDM_MNU(LCDM, MassiveNeutrinos, name="LCDM+mnu")`, then
     `LCDM(H0=67.3, Omegam=..., ...)`. The bases are the components in state
     order, a model among them standing for its own (`Model.__init_subclass__`
     builds `cls.spec`); `Model` subclasses `Recombination`, so every model
     has `tau_reio`. `Component` carries `dataclass_transform`, so ty and
     pyright read the constructor's keyword arguments off the annotated
     declarations of every base (`tests/models/test_model.py` runs ty on
     user code). At run time the same signature is `cls.__signature__`, and
     `__init__` raises `TypeError` for an unknown or missing key, then runs
     `spec.check`. A cosmology is immutable (`replace(**changes)`,
     `__replace__`), a JAX pytree of its keys (unflattened without a check),
     and `to_dict()`. `from_dict(params)` is the lenient path a dictionary
     for another model takes (`spec.check_keys`). Class keywords are settings
     (`class Coarse(LCDM, LMAX_G=10)`) and `name=`. `swap`, `add`, `remove`
     and `with_settings` derive classes at run time (memoised in `_DERIVED`,
     `with_settings` a subclass); a type checker sees only class statements.
     The class docstring gets a generated numpy "Parameters" section
     (`_documented`). Classmethods `parameters`, `parameter_keys`,
     `state_names`, `sparsity`, `setting`, `has`, `build` forward to the spec
   - `ModelSpec(name, components, settings)` (the old `CosmologyModel`): the
     component classes in state order and setting overrides; frozen and
     hashable, the solver's cache key. `replace(old, new, name=)`,
     `add(*components, name=)`, `remove(...)` and `with_settings(**kw)`
     derive specs. `build(thermal_history=None, extra=())` instantiates the component
     `Equations` (the old `component.Model`); `state_names()`, `sparsity()`, `setting(name)`,
     `parameters(extra=())` (the components' and `Recombination`'s
     declarations), `parameter_keys()` and `check(cosmology)`: a key another
     component declares but none of this model's must keep its default
     (`w_DE_0 != -1` needs `DarkEnergy`, `Nmnu > 0` `MassiveNeutrinos`), then
     each component's `check` hook runs (`DarkEnergy` rejects `w = -1`),
     traced values skipped. `resolved()` makes every
     setting explicit at the classes' current values: the kernel caches on
     the resolved spec, so rebinding a setting recompiles. Internal code
     takes specs; public functions taking a model accept either (`_spec_of`)
   - The models: `LCDM` (flat unless `Omegak`), `LCDM_MNU`, `W0WACDM` (its
     bases written out with `DarkEnergy` for `CosmologicalConstant`),
     `W0WACDM_MNU`. Massive neutrinos come after `ThomsonScattering`; no sum
     or state order moves, since an interaction contributes to neither
   - `DEFAULT_MODEL` (= `W0WACDM_MNU`, `DEFAULT_SPEC` its spec) is the one
     place a bare parameter dictionary gets a model (`density_coefficients`,
     `background_history`, `conformal_ages`, `directed_inputs`, which also
     take cosmologies). Those run its `check_keys` (the first half of
     `check`, without the components' hooks), so a dictionary that sets a
     parameter of a component outside it raises instead of being ignored
   - `_batch_of(cosmologies)` turns one cosmology or a sequence of one
     model's into the internal `CosmologyBatch(spec, dicts, model_class)`
     the solver reads; mixed models raise `TypeError`. The dicts' values are
     as given, so JAX values stay differentiable. There is no
     `PerturbationLayout` any more (removed 2026-11-12): indices come from
     `model.state_names()` or the joint model's `index`

4. **solver/_background.py** - The expansion history
   - `density_parameters(param, xp, model)` (every parameter the model
     declares, derived ones by `derive_parameters`; the `closes_budget`
     parameter is the root of `3 adotoa(a=1)^2 = grhom`, one secant step
     where the rate is linear in it, `CLOSURE_STEPS` = 6 more through
     `_background_solution` for a model with background states, so a scalar
     field's `V0` may close the budget: `Quintessence` in `tests/helpers.py`,
     `tests/solver/test_background.py`), `expansion_rate` / `dtauda` (the
     model's `adotoa` via `Equations.evaluate(..., math=jnp)` over
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

5. **models/_cosmology.py** - Cosmologies as the solver reads them: parameter
   dictionaries (`_cosmology_dictionary` of an instance is its `to_dict()`),
   their hashable form `_ParamCosmology` (`as_cosmology`, which takes an
   instance too and keeps no model), and
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
   keys beyond its fixed fields. `ModelSpec.check` skips traced values

6. **components/recombination.py** - Recombination history, inside the joint kernel
   - RECFAST's `(x_H, x_He, T_m)`, `ln a` and the optical depth `kappa` are
     five states of every trajectory, the `Recombination` component, a block
     of a modax `Hierarchy` stepping ahead of the perturbations
     (`kernel.joint_kernel`) and publishing `a`, `opacity` and `cs2` to them
     as quintic interpolants. Nothing is tabulated, and nothing names these
     states as the background (see `Equations.stages` below)
   - The stand-alone RECFAST solve in `ln a` (`thermodynamics_recfast.py`,
     `thermal_history_batch`, `evolve_background`) was removed on 2026-11-11:
     it served only tests. `tests/components/test_recombination.py` reads the kernel's block
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
     (`tests/components/test_recombination.py`)
   - Hydrogen leaves Saha equilibrium on *redshift*, not on `x_H`: Saha slope
     above `Z_H_SAHA_END` = 1600, the rate equation below, the two slopes
     blended over 2 x `DZ_H_SAHA_BLEND` = 50 in z (CLASS 3's
     `recfast_z_full_H_recombination` and its smoothing). RECFAST's switch at
     `x_H < 0.99` is a jump in the Jacobian, and the Rosenbrock step straddling
     it overshot `x_e` by up to 5 percent
   - The RECFAST constants live here too. **Recfast license** (UBC, not MIT):
     see the file header

7. **solver/_kernel.py** - The joint kernel and the launch
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
     (`Equations.initial_state(xp=jnp)`, whose fillers write into a recording
     `_Filled` buffer, nothing in place). k must be concrete: it sets the
     launch's shape and order
   - The two blocks are read off the equations (2026-11-14; before,
     `THERMAL_STATES` and `THERMAL_QUANTITIES` named them).
     `Equations.stages(mode=("k",))` runs the sparsity trace with the wave number
     followed like a state: a state whose rate reads neither `k` nor a state
     that does is background. `Equations.published()` is the minimum vertex cut
     (by numbers handed over, nearest the perturbations among equals)
     between the background states and what the perturbation rates read:
     `("a", "opacity", "cs2")` for every built-in model, plus a background
     state the perturbations read directly (a scalar field). So a component
     with a state such as `d_t(a) = a` lands in the background block, and
     `thermal_indices(model)` lists every background state
     (`tests/solver/test_kernel.py`)
   - The GPU right-hand side is the model's components plus `Recombination`,
     built by `component.Equations.build`: a driver with the tuple-return
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
     `Equations._collect` lets two components bring the same generated provider. `_rows` and the
     initial state iterate over the declarations. `PARAM_COLUMNS`, `N_PARAM`,
     `DIRECTED_COLUMNS` are `W0WACDM_MNU`'s row (`DEFAULT_LAYOUT`), a superset
     of every built-in model's, for host tests and `make_params`
   - `build_keep_hook(model, names)` is the save hook: a `(y, tau, p) ->
     tuple` function of state elements copied from the state and provided
     quantities such as `R` compiled by `Equations.build_function`, cached per
     model and set of names (`_keep_hook`)

8. **solver/_sensitivities.py** - `_kept_values`: the hooked launch as a
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

9. **solver/_solver.py** - `solve_eb(cosmologies, k, tau, *, keep=...)` solves
   one cosmology or a list of one model's and records only the named quantities at the given times, so the
   device holds `n_k * n_cosmo * n_tau * len(keep)` numbers and never the
   history. Returns the values themselves, `Annotated[jax.Array,
   ("cosmology", "name", "k", "tau")]`, names in `keep` order and the rest in
   the caller's order (there is no `Solution` class since 2026-11-16;
   `jnp.unstack(values, axis=1)` splits it by name, `helpers.by_name` in the
   tests); a time past a cosmology's `tau0` holds its state today. Traces
   under `jit` with every cosmology key traced (only k and the names
   static). The values are differentiable in `differentiable_parameters(model)`
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

10. **observables/power_spectrum.py** - `matter_power_spectrum(k, *, clxc, clxb,
    z, vb, tau0, Omegam, Omegab, A_s, n_s, k_p)`, JAX only, takes arrays and
    no solution or batch: the four quantities today `(..., n_k)` from a
    `solve_eb(..., keep=("clxc", "clxb", "z", "vb"),
    thermo_weight=0.1)` at a last time at or past every
    `tau0`, and the parameters `(...)`. Differentiable like `solve_eb`; it
    adds the end-time term (`clxc' = -k z`, `clxb' = -k (z + vb)` times
    `tau0 - stop_gradient(tau0)`), so `tau0` must come from
    `conformal_ages(batch)`: a `tau0` under `stop_gradient` drops the term
    silently. `examples/flat_lcdm_power_spectrum.py` is the
    entry point

11. **Jacobian sparsity** (`ModelSpec.sparsity`; its test is
    `tests/models/test_model.py`, modax's side of it
    `tests/solver/test_kernel.py`)
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

12. **components/_util.py** - the neutrino momentum bins and the generalised
    Gauss-Laguerre rule they use for `nqmax > 5`

### Computation Pipeline

The typical computation flow is:

1. **Make cosmologies of a model** (`LCDM(H0=..., Omegam=..., ...)`,
   `W0WACDM(..., w_DE_0=...)`; Ωm, Ωb, h, As, ns, mν, w0, wa, etc.)

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

5. **CMB spectrum**: DISCO-CMB, through `eb.solve_eb(cosmologies, ...)`

### Key Design Patterns

**JAX-First Design:**
- All code is JAX-compatible (no NumPy unless wrapped)
- Extensive use of `jax.numpy` instead of `numpy`
- Compatible with `jax.jit`, `jax.grad`, `jax.vmap` for performance and autodiff
- 64-bit precision: importing `discoeb` enables `jax_enable_x64` and turns
  off JAX's GPU preallocation (`_jax_config.py`, imported first)

**Cosmologies:**
- A cosmology is an instance of a model class, its parameters keyword
  arguments (`Omegam`, `Omegab`, `H0`, `Tcmb`, `YHe`, `Neff`, `A_s`, `n_s`,
  ...). Tests holding dictionaries go through `helpers.cosmologies_of(model,
  params)`, which is `from_dict` over one or a list

**Neutrino Treatment:**
- Massive neutrinos handled via momentum binning (nqmax bins)
- Gauss-Laguerre quadrature for momentum integration
- Perturbations tracked separately for each momentum bin (psi0, psi1, psi2, ...)

**Stiff ODE Integration:**
- Einstein-Boltzmann equations are stiff; modax's Rodas5P kernel integrates them
  with an Enzyme-derived Jacobian and a sparse direct solver (see
  `ModelSpec.sparsity`)
- The recombination system is not stiff but has piecewise switches; Rodas5P
  spends fewer steps on them than Tsit5

**Testing Philosophy:**
- `tests/` mirrors `src/discoeb/` (2026-11-16): one test file per source
  file, in the same folder, named after it without the underscore
  (`tests/solver/test_kernel.py` for `solver/_kernel.py`, `tests/test_init.py`
  for `__init__.py`). A new test goes in the file of the module it tests; no
  test file stands for a feature. What several files share lives in
  `tests/helpers.py` (host stand-ins, `MODELS`, the components added from
  outside: `SecondDarkMatter`, `Quintessence`, `CosmicTime`, the named
  fluids) and `tests/sweep.py`, imported by name (`pythonpath`). A test that
  launches a kernel carries `@pytest.mark.gpu`
- Tests compare against CLASS (standard cosmology code)
- Regression testing with `pytest-regressions` to track accuracy changes
- Benchmark testing with `pytest-benchmark` to track performance
- CLASS reference spectra are computed on the fly by the tests (classy)
- `tests/sweep.py` (called from `tests/models/test_lcdm.py` and
  `test_w0wacdm.py`) sweeps a fixed-seed Latin hypercube of 32 cosmologies
  per model (LCDM, W0WACDM, LCDM_MNU) in one launch each: finite P(k), steps
  well under `max_steps`, P(k) against CLASS at gates set from measured errors,
  and a self-convergence check against a 1000x tighter fp64-LU solve. A CPL
  `w(a)` crossing -1 hits `max_steps` and is silently wrong (strict `xfail`).
  The sweeps and the self-convergence tests are marked `slow`
- The default run is the fast suite (2026-11-16): 120 tests, 4.5 minutes
  from an empty kernel cache, about 30 s with it warm. A kernel costs 30 to
  200 s to compile per (model, settings, keep names, plain or joint), so the
  fast suite launches LCDM only, with five kernels: `P(k)` against CLASS,
  its derivatives, under `jit`, RECFAST and reionization against CLASS, and
  `solve_eb`'s kept quantities against the history. Marked `slow`, which
  `pyproject.toml` deselects: every launch of another model (the sweeps,
  `test_more_parameters_*`, the components added in the tests), the
  `solve_eb`-level derivative gate (200 s), the comoving curvature, the block
  step statistics and the scalar-field shooting. A new GPU test on a kernel
  outside those five is `slow`. Run everything with `uv run pytest tests -m
  "" -rs` (12 minutes warm)
- Two gates are noise-limited (measured 2026-11-14 by scaling H0 by 1 +
  O(1e-11), on the code before and after that day's refactor alike), so a
  change to any kernel's bits can flip them: `test_recfast_matches_class`'s
  `c_b^2` in the T_m switch window (over its 0.2 gate in 15 to 23 of 40
  draws; 0.5 percent of cosmologies, a few off by a factor 62, since
  `matter_temperature_rate` switches hard and `cs2` reads the rate), and
  `test_more_parameters_solve_eb[w_DE_a]`'s `dclxb` (3e-4 to 5e-3 against
  1e-3). DISCO-CMB's EE derivative gate (1.2e-3 to 2.2e-3 against 2e-3) and
  its reionization TE gate at `l = 30` (0.0195 to 0.0200 against 0.02) sit
  as close. Rerun with a nudged H0 before bisecting one of them

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
- **JAX 64-bit:** `import discoeb` enables `jax_enable_x64`; `conftest.py`
  also sets it before anything else is imported
- **Recfast License:** `recombination.py` contains code adapted from Recfast (UBC license, not MIT)
- **Benchmarks:** See BENCHMARKS.md for reference timings on various hardware.
  The spectrum benchmarks (`test_power_spectra_match_class`,
  `test_matter_power_spectrum_matches_class` likewise if changed) time a *fresh*
  batch after warming up on a distinct one: the thermodynamics is included, as
  in an inference iteration
- **Parameter Units:** Neutrino masses in eV, wave numbers in 1/Mpc, conformal time in Mpc, scale factor dimensionless
