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

# Host-only tests, in parallel over the cores (JAX stays on the CPU)
uv run pytest tests -m "not gpu" -n auto

# Run the tests (they need a GPU). -rs lists skips: without `classy` (dev
# group) the CLASS comparisons skip after their benchmark round, which reads
# like a pass in -q output.
uv run pytest tests -rs
uv run pytest "tests/test_observables.py::test_matter_power_spectrum_matches_class[N128]" -v

# The entry point, a notebook: P(k) of 128 flat LCDM cosmologies about Planck
# 2018, built, solved and timed in one jitted function. This reruns it and
# stores its outputs, which the docs show
uv run jupyter execute --inplace examples/flat_lcdm_power_spectrum.ipynb

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
`params` may be a pytree. It also needs modax's directional forward mode
(2026-11-17, the release after 0.4.0): `jax.jvp` of a launch integrates
one sensitivity column along the row's tangent, and a `vmap` over tangents
integrates one column per direction in the trajectory's own thread. There is no `sensitivity=`, `hook_size` or `quantity_hook` any more. numba-enzyme-cuda's CPU-only cap on numba
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
a, keep=names, extra=components)`, the cosmologies instances of one model
class (`[LCDM(omega_b=..., ...), ...]`): named quantities evaluated in the kernel at the
saved scale factors, the CMB's sources being a `LineOfSightSources` component passed as
`extra`; error weights by state name; differentiable in
`differentiable_parameters(model)` or in the model's directed inputs passed as `inputs=`: all of them, as
`directed_inputs` gives them, or the seven `DIRECTED_NAMES` every built-in
model leads with (`N_COMMON_DIRECTED`: the five densities, `YHe`, `Tcmb`),
which DISCO-CMB's `cl_power_spectrum_jax` passes. There is no default model
(removed 2026-11-16): every function takes cosmologies of a model or names
the model, and a bare dictionary raises `TypeError`. **Nothing is integrated before the solve** (2026-11-17; before,
`solver/_background.py` ran a quadrature for `tau0`, `ln a` at the start
and the reionization redshift, and DISCO-CMB read `background_history`,
`density_coefficients` and `conformal_ages`). The solve integrates in the
scale factor from `A_START` to 1, the conformal time is its state `tau`
(keep it: its value at `a = 1` is the conformal age), and a cosmology's
parameters are physical densities whose sum is `H0`. A CMB
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
     so several components may write one state. A sum is declared by the
     component that reads it, `grho = Sum()` (2026-11-16; before, a global
     `SUMS` tuple passed to `Equations`), and a model's sums are those its
     components declare. A method named after a sum contributes to it.
     Since 2026-11-15 a contribution to a sum,
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
     annotated since 2026-11-15: `omega_c: ArrayLike = Parameter()` a key, a
     constructor argument of every model with the component (with a default
     it may be left out), `w_DE_0: ArrayLike = Parameter(-1.0, column=True)`
     a key the kernel reads as a row column, `Parameter(differentiable=True)`
     a key `solve_eb` may differentiate (2026-11-17; before, a global
     `DIFFERENTIABLE_PARAMETERS` tuple listed them, so a component from
     outside could never add one), `A_s: ArrayLike =
     Parameter(solved=False)` a key the solve never reads, whose tangent
     `solve_eb` drops instead of raising (the primordial spectrum's;
     before, a global `UNUSED_BY_THE_SOLVE` tuple), `grhoc: ArrayLike =
     Derived(lambda omega_c: ..., directed=True)` a kernel
     input derived on the host by name from other parameters. `Parameter`
     and `Derived` are `dataclass_transform` field specifiers returning a
     `ParameterDeclaration`: `Derived`'s `init: Literal[False]` default keeps
     it out of the signature a type checker sees, and `__set_name__` raises
     on an unannotated declaration, which no type checker would see.
     `directed=True` makes it one of the leading row columns a
     derivative may move (until 2026-11-17 it made columns `x_nominal`
     and `dir_x` and a generated provider `x = x_nominal + eps * dir_x`;
     modax's directional mode does that itself for every column now).
     No parameter closes a budget any more (2026-11-17, `closes_budget`
     removed): every density is given, physically (`omega = Omega h^2`),
     and `Model.from_density_parameters` does the closure's algebra for
     whoever wants `H0` as an input. Two components of one model may share
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
   - Host-only tests: `tests/test_components.py`; a species with a
     parameter of its own, added from the tests alone: `SecondDarkMatter` in
     `tests/helpers.py`

2. **components/species.py, interactions.py, gravity.py** - The Einstein-Boltzmann system
   as components (`eb_model.py` was split into these on 2026-11-12)
   - There is no `Kinematics` component (removed 2026-11-17): a provider
     computes `1.0 / a`, `1.0 / tau` or `1.0 / (k * k)` itself. LLVM merges
     the repeated divisions, so the kernel compiles and runs as before
     (35.5 s cold either way, P(k) 186 ms and 270 ms)
   - `gravity.py`: `Background` (the Friedmann equation: `adotoa = sqrt(grho/3 -
     Kcurv)` and `adotoa_prime = -(grho + 3 gpres)/6`), `Metric`
     (etak and the Einstein constraints `z`, `sigma`, and `R`, the comoving
     curvature perturbation `-etak/k - aH dgq / (k (grho + gpres))`,
     CAMB's sign, read by `solve_eb`'s hook and not by the right-hand side),
     `Curvature` (geometry: `Kcurv`, `s2`; in no sum since 2026-11-14),
     `AdiabaticMode` (the growing-mode quantities `x`, `Rv`, `om`
     the initial conditions read). The sums live on their readers:
     `grho`, `gpres` on `Background`, `dgrho`, `dgq`, `dgpi` and
     `dgrho_streaming` on `Metric` (the last two read only by DISCO-CMB's
     Weyl potential), `grho_matter`, `grho_radiation`,
     `grho_free_streaming` on `AdiabaticMode`, from which `Rv` and `om` are
     ratios with `a` cancelling, so a new species enters the initial
     conditions by contributing its density; `MassiveNeutrinos` and `Fluid`
     contribute to none, as before 2026-11-14 when the four densities were
     named). The density constants
     (`GRHOM_PER_H0SQ`, `GRHO_PER_OMEGA` = 1e4 times it, ...) live here.
     Parameters: none on `Background` (`H0` is no parameter: it is
     `adotoa(a = 1)`, `Model.H0`; `Background.adot = a adotoa` is the
     solve's clock), `A_s`, `n_s`, `k_p` on `AdiabaticMode`, `omega_k`,
     `grhok` on `Curvature`; in `species.py` `omega_c`, `grhoc` on CDM,
     `omega_b`, `YHe`, `grhob` on `Baryons`, `Tcmb`, `grhog` on `Photons`,
     `Neff`, `grhornomass` on `MasslessNeutrinos`, `omega_de`, `grhov` on
     both dark energies, `omega`, `grho0` on `Fluid`, the
     `w_DE_*`, `cs2_DE` on `DarkEnergy`, `Nmnu`, `mnu`, `grhor_nu`, `n_mnu`,
     `amnu` on `MassiveNeutrinos`; `z_re` on `Recombination`
   - The gravity interface (2026-11-14): species contribute to the sums
     (fluids only) and read `adotoa`, `z`, `sigma`, which
     `component.Gravity` subclasses (`Background`, `Metric`, `Curvature`)
     provide. A model missing one fails at build, like any unprovided
     quantity (no special check since 2026-11-16). Another theory is `LCDM.swap(Background, Other, name=)` or
     `swap(Metric, Other, name=)`; `from_density_parameters` follows since
     it solves the model's own `adotoa(a=1)` (`tests/test_components.py`). A scalar state's rate is a
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
     the synchronous gauge's condition and a model class raises without
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
   - Host-only tests: `tests/test_components.py`,
     `tests/test_models.py`

3. **models/_model.py, lcdm.py, w0wacdm.py** - Cosmological models built from those components
   - Split on 2026-11-16: `_model.py` has the base class `Model`,
     `_batch_of` and `_model_of`, `lcdm.py`
     `LCDM` and `LCDM_MNU`, `w0wacdm.py` `W0WACDM` and `W0WACDM_MNU`, and
     `models/__init__.py` exports them
   - Since 2026-11-15 a model is a class and a cosmology an instance of it:
     `class LCDM(Model, Background, ..., ThomsonScattering)`,
     `class LCDM_MNU(LCDM, MassiveNeutrinos, name="LCDM+mnu")`, then
     `LCDM(omega_b=0.0224, omega_c=0.120, omega_de=0.311, ...)`. The bases are the components in state
     order, a model among them standing for its own (`Model.__init_subclass__`
     sets `cls.name`, `cls.components` and `cls.settings`); `Model` subclasses `Recombination`, so every model
     has `z_re`. `Component` carries `dataclass_transform`, so ty and
     pyright read the constructor's keyword arguments off the annotated
     declarations of every base (`tests/test_models.py` runs ty on
     user code). At run time the same signature is `cls.__signature__`, and
     `__init__` raises `TypeError` for an unknown or missing key, then runs
     `cls.validate`. A cosmology is immutable (`replace(**changes)`,
     `__replace__`), a JAX pytree of its keys (unflattened without a check),
     and `to_dict()`. `from_dict(params)` is the lenient path a dictionary
     for another model takes (`validate_keys`). Class keywords are settings
     (`class Coarse(LCDM, LMAX_G=10)`) and `name=`. `swap`, `add`, `remove`
     and `with_settings` derive classes at run time (memoised in `_DERIVED`,
     `with_settings` a subclass); a type checker sees only class statements.
     The class docstring gets a generated numpy "Parameters" section
     (`_documented`)
   - The model class is also what the solver reads (2026-11-17; until then
     a frozen `ModelSpec(name, components, settings)` on `cls.spec`, the
     old `CosmologyModel`). Classmethods: `build(thermal_history=None,
     extra=(), ...)` instantiates the component `Equations` (the old
     `component.Model`); `state_names()`, `sparsity()`, `setting(name)`,
     `has`, `instances`, `parameters(extra=())` (the components' and
     `Recombination`'s declarations), `parameter_keys()`,
     `differentiable_parameters()` and `validate(cosmology)`: a key another
     component declares but none of this model's must keep its default
     (`w_DE_0 != -1` needs `DarkEnergy`, `Nmnu > 0` `MassiveNeutrinos`), then
     each component's `check` hook runs (`DarkEnergy` rejects `w = -1`),
     traced values skipped. It is not `check`, nor `check_keys` but
     `validate_keys`, since a model inherits its components' `check` hooks;
     the spec's `replace(old, new)` is `swap`, `replace` being the
     cosmology's. `resolved()` makes every setting explicit at the classes'
     current values and returns one class per (name, components, settings),
     memoised in `_RESOLVED`: the model itself if its settings are already
     explicit, else a subclass of the first model asked. The kernel caches
     on it, so rebinding a setting recompiles and equal models share a
     kernel. Public functions taking a model check it with `_model_of`
   - The models: `LCDM` (flat unless `Omegak`), `LCDM_MNU`, `W0WACDM` (its
     bases written out with `DarkEnergy` for `CosmologicalConstant`),
     `W0WACDM_MNU`. Massive neutrinos come after `ThomsonScattering`; no sum
     or state order moves, since an interaction contributes to neither
   - Physical densities (2026-11-17). A model's own parameters are
     `omega_b`, `omega_c`, `omega_de`, `omega_k` (and a fluid's `omega`),
     so the kernel reads no `H0`: recombination takes `grhob` and the
     model's `Hz` (RECFAST's Hubble-time estimate `2 / (3 H0 (1+z)^1.5)` in
     the matter-temperature switch became `2 / (3 H)`).
     `Model.from_density_parameters(Omegam=, Omegab=, H0=, Omegak=0, ...)`
     converts and finds `omega_de` (`Model.physical_parameters`, also
     the dictionary-level form `helpers.cosmologies_of` uses: one secant
     step on `hubble_constant()^2`, exact since it is linear in a density;
     any other component's `x_Omega` becomes `x_omega`). Plain JAX
     arithmetic, so a derivative in `H0` or `Omegam` goes through it.
     With floats it uses `math`, with JAX values `jnp`, and the two round
     `omega_de` differently in the last bit, which moves a step-acceptance
     and `P(k)` by 1e-5: compare a jitted solve against an eager one of the
     same converted parameters (`test_matter_power_spectrum_under_jit`).
     `Model.H0` (`Model.hubble_constant`) is `adotoa(a=1)`; a model
     whose rate reads a state (`Equations.reads_state("adotoa")`: a scalar
     field) raises, has no `from_density_parameters`, and gets its Hubble
     constant from a solve (`adotoa` kept at `a = 1`). So a scalar field's
     `V0` is a parameter and nothing shoots (`Quintessence`,
     `qcdm_parameters` in `tests/helpers.py`)
   - No model is a default (`DEFAULT_MODEL`, `W0WACDM_MNU`, removed
     2026-11-16): `directed_inputs` takes cosmologies, the model coming
     from their class, and `build_keep_hook` and `_columns` take the model
     or its layout as a required argument
   - `_batch_of(cosmologies)` turns one cosmology or a sequence of one
     model's into the pair `(model, dicts)` the solver reads; mixed models raise `TypeError`. The dicts' values are
     as given, so JAX values stay differentiable. There is no
     `PerturbationLayout` any more (removed 2026-11-12): indices come from
     `model.state_names()` or the joint model's `index`

4. **solver/_background.py** is gone (2026-11-17). What it did, and where
   each piece went:
   - the budget closure: `Model.from_density_parameters` (above);
     `density_parameters(param, model, xp)`, now in `_kernel.py`, is
     `derive_parameters` alone
   - `tau0`, the end of every trajectory: the solve ends at `a = 1`
   - `ln a` at `TAU_START`: the start is `A_START` = 2e-7 for everyone, and
     the conformal time there is `Recombination`'s `_tau_initial`, `(2
     sqrt(1+e) / (sqrt(1+e) + 1)) / adotoa` with `e` the matter-to-radiation
     ratio, exact for radiation plus matter (1e-6 against the integral,
     massive neutrinos included: `tests/test_components.py`)
   - a model's own background states at the start: their `State(initial=)`
   - the scalar-field shooting: gone with the closure
   - `background_history`, `density_coefficients`, `conformal_ages`,
     `expansion_rate`: removed; keep `tau`, `adotoa` in a solve, or
     evaluate on the host (`helpers.on_the_host`)
   - The momentum bins (`neutrino_momentum_bins`, cached) live in `util.py`
     beside the Gauss-Laguerre rule

5. **models/_cosmology.py** is gone (2026-11-17). `_columns(cosmologies,
   model, held=True)` lives in `solver/_kernel.py`: it stacks a batch's
   dictionaries (`_batch_of`'s, every key present) into JAX columns
   `(n_cosmo,)` by key, values traced or not. With `held` a key declared
   `solved=False` gets `stop_gradient`, and a key the solve reads but the
   model does not differentiate (outside
   `Model.differentiable_parameters()`: `Nmnu`, `YHe`, `cs2_DE`) passes
   through `_not_differentiable`, a `custom_jvp` whose rule raises, so a
   derivative is never a silent zero. `_is_concrete` lives in
   `models/_model.py`. A cosmology still holds the traced keys of other
   models `from_dict` was given (`_held`, in the batch's dictionaries
   beside its own): `_columns` gives each a column the kernel never reads,
   so that differentiating one raises (dropping them was tried on
   2026-11-17 and returned a silent zero for `mnu` on LCDM)

6. **components/recombination.py** - Recombination history, inside the joint kernel
   - The conformal time `tau`, RECFAST's `(x_H, x_He, T_m)` and the optical
     depth `kappa` are five states of every trajectory, the `Recombination` component, a block
     of a modax `Hierarchy` stepping ahead of the perturbations
     (`kernel.joint_kernel`) and publishing `tau`, `opacity` and `cs2` to them
     as quintic interpolants. The scale factor is the time itself, and
     `tau`'s rate is one (per unit of itself: the clock divides). Nothing is tabulated, and nothing names these
     states as the background (see `Equations.stages` below)
   - The stand-alone RECFAST solve in `ln a` (`thermodynamics_recfast.py`,
     `thermal_history_batch`, `evolve_background`) was removed on 2026-11-11:
     it served only tests. `tests/test_components.py` reads the kernel's block
     through `solve_eb(..., keep=("tau", "xe", "Tmat", "cs2", "opacity",
     "kappa"))` at one wave mode (`kernel_thermal_history`): `x_e` within
     0.36 percent of CLASS recfast, `T_m` 0.04, `c_b^2` 0.4 (9.7 in the
     tight-coupling switch window near z = 850)
   - The rates' transcendentals run in fp32 by default (`FP32_MATH`,
     `DISCO_THERMO_FP32_MATH`); `a` and the reionization opacity stay fp64
   - Reionization: a cosmology's `z_re` (default 0, none), a directed
     parameter, centres CAMB's tanh step in `x_e` (H and HeII together, width
     0.5 in `(1+z)^1.5`; HeIII at z = 3.5). `tau_reio` is an output, the state
     `kappa_reio` today (2026-11-16; before, `tau_reio` was the input
     and `z_re` a solver column bisected on the background). That state is
     the component `ReionizationDepth`, which a model adds
     (`LCDM.add(ReionizationDepth, name=...)`, `helpers.LCDM_REIO`): in
     every model's block it cost a solve 4 to 7 percent (2026-11-17), so
     only a model with it reads `tau_reio` or differentiates `z_re`
     (`Model.differentiable_parameters` drops `z_re` otherwise, and the
     derivative raises). Same parametrization as CLASS `reio_camb`:
     `kappa_reio` today against its `tau_reio` at the same `z_reio`
     (`tests/test_components.py`). A sensitivity launch
     weights `kappa_reio` by `REIO_ERROR_WEIGHT` = 1e4: at the block's
     weight the steps across the 0.5-wide step resolve `tau_reio` but not
     its `z_re` derivative (6 percent off; `dP/dz_re` 3), and the weight
     costs a primal P(k) batch with reionization 14 percent, so primal
     launches go without it
   - `helium_rate_equation` floors `1 - x_He` at 1e-6: a solver stage that
     probes `x_He >= 1` made the escape probabilities overflow, and the NaN
     stalled one sweep cosmology at `max_steps` (2026-11-16)
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
     batch with `_pack`, one `jax.jit` per (model, k grid, batch size): the
     parameter rows (`_rows`) and the initial state at `A_START`
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
     `("tau", "opacity", "cs2")` for every built-in model, plus a background
     state the perturbations read directly (a scalar field). So a component
     with a state such as `d_t(a) = a` lands in the background block, and
     `thermal_indices(model)` lists every background state
     (`tests/test_solver.py`)
   - The GPU right-hand side is the model's components plus `Recombination`,
     built by `component.Equations.build`: a driver with the tuple-return
     signature modax and Enzyme need, calling one device function per
     provider. Loops over multipoles and runtime indices are fine because
     numba-enzyme optimises the primal with `O3` before Enzyme.
   - The clock (2026-11-17). `Equations(components, time=, clock=)`: the
     components write their rates in conformal time, and the built driver
     returns each divided by `clock`, the rate of `time` in it. The joint
     model has `time="a"`, `clock="adot"` (`a' = a adotoa`): every
     cosmology runs from `A_START` to 1 and none needs its `tau0` first.
     The trace of `sparsity`/`stages`/`published` adds the clock's
     dependencies to every rate (none for the built-in models, whose rate
     reads the time and parameters; a scalar field's reads the field).
     `ln a` (`clock="adotoa"`) was tried first and lost: at the same
     tolerance it was 13 times less accurate at every k (9e-3 against 4e-4
     of a converged solve), needing `atol` = 1e-6 and 40 percent more time,
     and its large early steps jumped over the HeIII Saha bump in one
     cosmology in ten. Early on every solution is a low-order polynomial in
     `tau`, which Rodas5P integrates nearly exactly, and an exponential in
     `ln a`; `a` grows as `tau` in the radiation era and keeps the
     polynomials. Measured on the 128-cosmology LCDM batch against
     `refactor/models` in `tau`: P(k) vs CLASS 3.54e-3 (3.64e-3), rms
     self-convergence 5.7e-5 (5.7e-5), 267 ms (270), one cosmology 186 ms
     (188); `PERTURB_FIRST_STEP` = `A_START`
   - The parameter row is generated: `parameter_layout(model, extra)` (cached
     per resolved model) is `(the directed parameters,
     SOLVER_COLUMNS = k,
     the other column parameters)`, the directed
     ones in `DIRECTED_NAMES` order, then component order. `grhok`, `w_DE_*`, `cs2_DE`,
     `grhor_nu` and `amnu` are directed (2026-11-13), `z_re` too
     (2026-11-16, last: `Recombination` comes last); `H0` and `Omegab`
     left with the physical densities (2026-11-17); `amnu` is one
     declaration `MassiveNeutrinos` and `Recombination` share. `_rows` and the
     initial state iterate over the declarations. Host tests that build any
     built-in model on one row use `helpers.PARAM_COLUMNS` and `N_PARAM`,
     `W0WACDM_MNU`'s row, a superset of every built-in model's
   - `build_keep_hook(model, names)` is the save hook: a `(y, a, p) ->
     tuple` function of state elements copied from the state and provided
     quantities such as `R` compiled by `Equations.build_function`, cached per
     model and set of names (`_keep_hook`)

8. **solver/_sensitivities.py** - `_kept_values`: the hooked launch as a
   `jax.custom_jvp` in a pytree of variables that `to_inputs` maps to the
   layout's directed inputs (`solve_eb` passes the cosmology columns). Its rule (`symbolic_zeros=True`) is
   `jax.vmap` of `jax.jvp` of the launch over one unit direction per
   *differentiated* variable, contracted with the variables' tangent
   (linear, so `grad` works off the same launch). modax's directional mode
   integrates one sensitivity column along each direction's row tangent
   (`d inputs / d variable`, plain JAX through `to_inputs`) and its vmap
   rule packs the directions as columns of the trajectory's own thread
   (one trajectory, one thread: no replicas), so the cost is
   the number of parameters differentiated (5 for the densities' five),
   not of directed inputs, and an undifferentiated input costs nothing.
   The joint kernel is compiled per number of directions, and a thread's
   local memory, which the driver reserves for every resident thread,
   grows with them: seven columns of the 55-state system exhaust a 12 GB
   card. `MAX_SENSITIVITY_STATE` (330 components, `DISCO_MAX_SENSITIVITY_STATE`)
   caps a thread's joint state, and a derivative in more variables runs as
   several launches of one size, the last padded with zero directions.
   Measured (RTX 4070 SUPER, five parameters): one cosmology 2.4 s where
   replicas took 0.39 s, 128 cosmologies 3.6 s where they took 4.5 s.
   `solve_eb(..., split_derivatives=True)` (default `False`) brings the
   replicas back: the rule copies each trajectory once per differentiated
   variable (row `(k_idx * n_cosmo + ci) * R + r`) and takes one `jax.jvp`
   of the launch with copy `r`'s row tangent `d inputs / d variable_r`, so
   each thread holds the state and one column, with no cap or chunking.
   One cosmology, five parameters: 0.38 s against 2.4 s. The two modes
   agree to 1e-7 at rtol 1e-6 and to 1e-3 at the default 1e-4
   (`test_split_derivatives_match_one_thread`).
   The launch runs with `sens_error_control=False` and keeps each
   quantity's derivative beside it. Until 2026-11-17 the rule built
   replicas of every trajectory and the `eps` and `dir` columns by hand. fp64
   LU; the joint path's primal agrees with the plain fp32-LU one to ~1e-6 of
   scale. At *fixed scale factor*, and at fixed initial state. (A `tau_reio` input with the implicit
   function theorem on a host bisection for `z_re` missed central
   differences of `P(k)` by 1.8 percent on 2026-11-13; `z_re` is now the
   input and differentiates directly.) The rows, initial state and save times are
   explicit arguments of the `custom_jvp`, never closed over, and the rule
   closes over no JAX array: JAX traces it lazily, after the enclosing
   `jit` trace, so a captured tracer breaks `jit(grad(jit(f)))`

9. **solver/_solver.py** - `solve_eb(cosmologies, k, a, *, keep=...)` solves
   one cosmology or a list of one model's and records only the named quantities at the given scale factors, so the
   device holds `n_k * n_cosmo * n_a * len(keep)` numbers and never the
   history. Returns the values themselves, `Annotated[jax.Array,
   ("cosmology", "name", "k", "a")]`, names in `keep` order and the rest in
   the caller's order (there is no `Solution` class since 2026-11-16;
   `jnp.unstack(values, axis=1)` splits it by name, `helpers.by_name` in the
   tests). `a` runs from `A_START` on, strictly increasing, and 1 is today
   for every cosmology. Traces
   under `jit` with every cosmology key traced (only k and the names
   static). The values are differentiable in `differentiable_parameters(model)`,
   the keys declared `Parameter(differentiable=True)` (`omega_b`, `omega_c`,
   `omega_de`, `Tcmb`, `Neff`, `omega_k`, and `mnu`, `w_DE_0`, `w_DE_a`,
   `z_re` where the model has the component): plain JAX from them to the directed inputs
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
   `dP/dYHe` and `dP/dcs2_DE` stay 0.8 and 1 percent off at every step
   tried, so those two are held. `dP/dz_re` needs a step of 1 in `z_re`:
   it moves sum P by 1.4e-5 of itself per unit, so a step of 0.05 is noise. Also
   `directed_inputs`, `state_names(model)`. The history of every state is
   `_kernel._solve_batch` (tests only; `solve_perturbation_history_batch`,
   `make_params` and `make_initial_states` were removed 2026-11-17: the
   tests read rows and initial states off `_pack`, `helpers.packed`)

10. **observables/power_spectrum.py** - `matter_power_spectrum(k, *, clxc, clxb,
    omega_c, omega_b, A_s, n_s, k_p)`, JAX only, is one cosmology's
    (2026-11-16): the two contrasts `(n_k,)` from a `solve_eb(..., [1.0],
    keep=("clxc", "clxb"), thermo_weight=0.1)`, and the parameters as
    scalars. A batch is
    `jax.vmap(matter_power_spectrum, in_axes=(None,))(k, **today, ...)`,
    the keyword arguments mapped over their leading axis. Differentiable
    like `solve_eb`, at fixed scale factor, so there is no end-time term
    any more (2026-11-17; it took `z`, `vb` and `tau0` for it).
    `examples/flat_lcdm_power_spectrum.ipynb` is the entry point

11. **Jacobian sparsity** (`Model.sparsity`; its test is
    `tests/test_models.py`, modax's side of it
    `tests/test_solver.py`)
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

1. **Make cosmologies of a model** (`LCDM(omega_b=..., omega_c=...,
   omega_de=..., ...)` or `LCDM.from_density_parameters(Omegam=...,
   Omegab=..., H0=..., ...)`, `W0WACDM(..., w_DE_0=...)`)

2. **Nothing before the solve** but algebra: the parameter rows and the
   initial state at `A_START` (`kernel._pack`)

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
  arguments (`omega_b`, `omega_c`, `omega_de`, `Tcmb`, `YHe`, `Neff`, `A_s`,
  `n_s`, ...). Tests hold dictionaries in `Omegam`, `Omegab` and `H0` and go
  through `helpers.cosmologies_of(model, params)`, which converts them
  (`Model.physical_parameters`) and is `from_dict` over one or a list;
  `helpers.physical(model, params)` gives the converted dictionary

**Neutrino Treatment:**
- Massive neutrinos handled via momentum binning (nqmax bins)
- Gauss-Laguerre quadrature for momentum integration
- Perturbations tracked separately for each momentum bin (psi0, psi1, psi2, ...)

**Stiff ODE Integration:**
- Einstein-Boltzmann equations are stiff; modax's Rodas5P kernel integrates them
  with an Enzyme-derived Jacobian and a sparse direct solver (see
  `Model.sparsity`)
- The recombination system is not stiff but has piecewise switches; Rodas5P
  spends fewer steps on them than Tsit5

**Testing Philosophy:**
- `tests/` has one file per source folder (2026-11-17; one per source
  file from 2026-11-16 until then): `test_components.py`, `test_models.py`,
  `test_observables.py`, `test_solver.py`, and `test_init.py` for the
  package's `__init__.py`. Each has a section per module, under a banner
  comment, and a new test goes in the section of the module it tests.
  `test_components.py` defines toy components named like the package's
  (`Background`, `ColdDarkMatter`, `Photons`), so its later sections reach
  the real ones as `gravity.Background`, `species.Photons`.
  Host tests run in parallel: `uv run pytest tests -m "not gpu" -n auto`
  (pytest-xdist). A run that excludes the `gpu` tests keeps JAX on the CPU
  (`conftest.pytest_configure`), so it holds no device memory and may run
  beside a GPU run. With `-n` and the `gpu` tests included, pass `--dist
  loadgroup` too: it keeps every `gpu` test on one worker, since two
  processes on one GPU return garbage. It is not in `addopts`:
  pytest-benchmark takes any `--dist` for an xdist run and disables the
  benchmarks, which then run their warm-up batch alone. What several files share lives in
  `tests/helpers.py` (host stand-ins, `MODELS`, the components added from
  outside: `SecondDarkMatter`, `Quintessence`, `CosmicTime`, the named
  fluids) and `tests/sweep.py`, imported by name (`pythonpath`). A test that
  launches a kernel carries `@pytest.mark.gpu`
- Tests compare against CLASS (standard cosmology code)
- Regression testing with `pytest-regressions` to track accuracy changes
- Benchmark testing with `pytest-benchmark` to track performance
- CLASS reference spectra are computed on the fly by the tests (classy)
- `tests/sweep.py` (called from `tests/test_models.py`) sweeps a fixed-seed Latin hypercube of 32 cosmologies
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

## Examples

`examples/<name>.ipynb` is one minimal case of one use each, a notebook of
code cells between explanation, committed with its outputs. The docs show
them through MyST-NB with `nb_execution_mode = "off"`: nothing runs at build
time (they need a GPU), so the pages are the notebooks as last run.
`docs/examples` is a symlink to `examples/`, and `examples/index.md` is the
card grid and toctree. After changing a notebook, rerun it with `uv run
jupyter execute --inplace examples/<name>.ipynb` (nbclient, dev group) and
commit the outputs. ruff lints and formats the notebooks too. They replaced
the old notebooks on 2026-11-16 (the CMB one moved to DISCO-CMB):
- `flat_lcdm_power_spectrum` - Start here: P(k) of a batch, timed
- `matter_power_derivative` - `dP/dOmega_m` by `jacfwd` against a finite
  difference
- `power_spectrum_jacobian` - the Jacobian of `P(k, a)` in ten parameters
- `compare_with_class` - CDM and baryon spectra against CLASS
- `correlation_function_multipoles` - Kaiser `xi_0`, `xi_2`, `xi_4` and
  their derivatives
- `redshift_space_power_response` - `P(k, mu)` with Kaiser and
  Alcock-Paczynski, response maps
- `fisher_forecast` - a Euclid-like galaxy clustering forecast
- `setup_n_body_ics` - initial conditions for an N-body run

`solve_eb` saves at scale factors and differentiates at fixed scale factor,
so a derivative at fixed redshift is what it returns: the notebooks save at
their redshifts (ported 2026-11-17; before, they kept `a` at a few
conformal times and interpolated). They build cosmologies with
`from_density_parameters`, and those that need a distance keep `tau`.

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
