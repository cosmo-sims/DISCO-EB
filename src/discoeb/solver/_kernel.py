"""The joint kernel: the parameter row, the packed batch and the modax launch.

One modax launch integrates every (wave mode, cosmology) pair of a batch. The
joint model is the perturbation model with the recombination block
(:class:`~discoeb.components.recombination.Recombination`) stepping ahead of it as a modax
:class:`~modax.Hierarchy`. This module builds that hierarchy, packs a batch
into parameter rows and initial states, launches it, and compiles the save
hooks that keep named quantities.
"""

import functools
import os
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..components._component import Equations, _compile
from ..components.recombination import THERMO_ERROR_WEIGHT, Recombination
from ..models import DEFAULT_SPEC
from ..models._cosmology import _as_batch, _columns
from ..models._model import Model, ModelSpec, _spec_of
from ._background import TAU_START, background_endpoints, density_parameters

PERTURB_RTOL = 1.0e-4
PERTURB_ATOL = 1.0e-4
PERTURB_FIRST_STEP = 1.0e-2
PERTURB_MAX_STEPS = 20000

SOLVER_COLUMNS = ("tau_start", "tau_end", "k", "cosmology_tag", "save_index", "z_re")
"""The row's columns the solver sets rather than a component: the
integration span, the wave number, the cosmology's index in the batch, the
save counter and the reionization redshift the background finds for
``tau_reio``."""

_BUILT_IN_DIRECTED = (
    "grhog",
    "grhornomass",
    "grhoc",
    "grhob",
    "grhov",
    "YHe",
    "Tcmb",
    "H0",
    "Omegab",
)
"""The public order of the built-in directed parameters (:data:`DIRECTED_NAMES`).
A model's other directed parameters follow them in component order."""


@dataclass(frozen=True)
class ParameterLayout:
    """The columns of a model's parameter row, generated from its parameters.

    The row is ``(x_nominal for each directed x, SOLVER_COLUMNS, the other
    column parameters, eps, dir_x for each directed x)``. The directed
    parameters come in :data:`DIRECTED_NAMES` order, then in component
    order, and the other columns in component and declaration order, so the
    layout is a function of the resolved model and the kernels cached on it
    stay valid. ``eps`` is zero in every primal solve, and the derivative in
    it is the directional derivative along ``dir``
    (:func:`~discoeb.solver._sensitivities._kept_values`).
    """

    columns: dict
    """Column by the name the providers read it under: ``grhoc_nominal``, ``k``, ``eps``, ``dir_grhoc``."""
    directed: tuple[str, ...]
    """The directed parameters: the inputs a sensitivity launch moves."""
    values: tuple[tuple[str, int], ...]
    """``(parameter, column)`` of the parameters the row holds."""

    @property
    def n(self) -> int:
        return len(self.columns)

    @property
    def ix_eps(self) -> int:
        return self.columns["eps"]

    @property
    def ix_dir(self) -> int:
        return self.ix_eps + 1

    @property
    def ix_tau_end(self) -> int:
        return self.columns["tau_end"]

    @property
    def directed_columns(self) -> tuple[int, ...]:
        """The nominal columns of :attr:`directed`."""
        return tuple(self.columns[f"{name}_nominal"] for name in self.directed)


def parameter_layout(model: type[Model] | ModelSpec, extra=()) -> ParameterLayout:
    """Return the row layout of ``model`` with the component classes ``extra`` appended (cached)."""

    return _parameter_layout(_spec_of(model).resolved(), tuple(extra))


@functools.lru_cache(maxsize=None)
def _parameter_layout(model: ModelSpec, extra) -> ParameterLayout:
    parameters = model.parameters(extra)
    directed = [n for n, p in parameters.items() if p.directed]
    directed.sort(
        key=lambda n: (
            _BUILT_IN_DIRECTED.index(n)
            if n in _BUILT_IN_DIRECTED
            else len(_BUILT_IN_DIRECTED)
        )
    )
    others = [n for n, p in parameters.items() if p.column and not p.directed]
    clash = sorted(set(SOLVER_COLUMNS) & set(parameters))
    if clash:
        raise ValueError(f"{clash} name solver columns, not parameters")
    names = [
        *(f"{n}_nominal" for n in directed),
        *SOLVER_COLUMNS,
        *others,
        "eps",
        *(f"dir_{n}" for n in directed),
    ]
    columns = {name: i for i, name in enumerate(names)}
    values = tuple(
        (n, columns[parameters[n].column_name]) for n in (*directed, *others)
    )
    return ParameterLayout(columns, tuple(directed), values)


DEFAULT_LAYOUT = parameter_layout(DEFAULT_SPEC)
"""The row of :data:`~discoeb.models.DEFAULT_MODEL`, whose columns are a superset of
every built-in model's. Each model's kernel reads its own
(:func:`parameter_layout`); a host test may build any built-in model on this one."""

PARAM_COLUMNS = DEFAULT_LAYOUT.columns
"""The default row's columns by the names the providers take them under."""
N_PARAM = DEFAULT_LAYOUT.n
DIRECTED_NAMES = DEFAULT_LAYOUT.directed
"""The inputs a trajectory's sensitivity direction may move in
:data:`~discoeb.models.DEFAULT_MODEL`, a superset of every built-in model's.
The first nine, every model's, are the five background densities and the
recombination rates' ``YHe``, ``Tcmb``, ``H0`` and ``Omegab``
(:data:`N_COMMON_DIRECTED`). The others are the curvature density, the
dark-energy parameters and the massive neutrinos' density and mass; each
model has those of its components (:func:`parameter_layout`)."""
N_COMMON_DIRECTED = len(_BUILT_IN_DIRECTED)
"""The directed inputs every built-in model leads with, which ``solve_eb``
also takes alone as ``inputs``."""
DIRECTED_COLUMNS = DEFAULT_LAYOUT.directed_columns
"""The nominal columns of :data:`DIRECTED_NAMES` in the default row."""

TRAJECTORIES_PER_BLOCK = 32
"""Wave modes packed into one CUDA block by modax's thread-local Rodas5P kernel."""

CUDA_MAX_REGISTERS = int(os.environ.get("DISCO_CUDA_MAX_REGISTERS", "168"))
"""Per-thread register cap for that kernel: fewer registers, more resident warps."""


def joint_model(model: type[Model] | ModelSpec, extra=()) -> Equations:
    """Return the components of ``model`` with :class:`~discoeb.components.recombination.Recombination` appended.

    The recombination states come after the perturbations', so the
    perturbation state keeps the indices of :meth:`ModelSpec.state_names
    <discoeb.models.ModelSpec.state_names>`. The five thermal states
    (``ln a``, ``x_H``, ``x_He``, ``T_m`` and the optical depth) are the last
    columns of every history, at ``model.index["lna"]`` and so on. ``extra``
    lists component classes to append after them (the line-of-sight sources).
    """

    return _spec_of(model).build(extra=[Recombination(), *(cls() for cls in extra)])


def thermal_indices(model) -> tuple[int, ...]:
    """Return the flat indices of the background states in the joint ``model``.

    They are the states :meth:`Equations.stages <discoeb.components._component.Equations.stages>`
    finds to evolve without the wave number: the five of the recombination
    block and any a component of the model adds.
    """

    return tuple(model.state_indices(model.stages()[0]))


def _restrict(pattern: np.ndarray, indices) -> np.ndarray:
    """Return the ``(nnz, 2)`` pattern's rows and columns ``indices``, renumbered."""

    position = {idx: k for k, idx in enumerate(indices)}
    kept = [
        (position[r], position[c])
        for r, c in pattern.tolist()
        if r in position and c in position
    ]
    return np.array(sorted(kept), dtype=np.int64).reshape(-1, 2)


def joint_kernel(model: type[Model] | ModelSpec, extra=()):
    """Return ``(joint model, hierarchy)`` of ``model``, cached per resolved model and ``extra``.

    The hierarchy is a modax :class:`~modax.Hierarchy` of two blocks, which
    the equations themselves split (:meth:`Equations.stages
    <discoeb.components._component.Equations.stages>`). The background block holds every
    state whose rate reads no wave number: ``ln a``, the recombination
    history, the optical depth, and whatever a model adds, such as a scalar
    field. It publishes the fewest quantities that separate it from the
    perturbations (:meth:`Equations.published
    <discoeb.components._component.Equations.published>`: the scale factor, the opacity and
    the baryon sound speed for the built-in models), and the perturbation
    block's right-hand side takes them as given. modax steps the background
    block ahead of the perturbations and hands the quantities down as a
    quintic Hermite interpolant. So the perturbation Jacobian has no
    background columns, and the thermal providers run once per background
    step. The state keeps the joint model's order.
    """

    return _joint_kernel(_spec_of(model).resolved(), tuple(extra))


@functools.lru_cache(maxsize=None)
def _joint_kernel(cosmology_model: ModelSpec, extra):
    from modax import Block, Hierarchy

    model = joint_model(cosmology_model, extra)
    columns = _parameter_layout(cosmology_model, extra).columns
    pattern = model.sparsity()
    background, others = model.stages()
    published = model.published()
    thermal = model.state_indices(background)
    hierarchy_idx = model.state_indices(others)
    recombination = Block(
        model.build(columns, states=background, name="recombination_rhs"),
        len(thermal),
        outputs=model.build_function(
            published,
            inputs=("y", "tau"),
            params=columns,
            states=background,
            name="thermal_quantities",
        ),
        n_outputs=len(published),
        sparsity=_restrict(pattern, thermal),
        name="recombination",
    )
    hierarchy = Block(
        model.build(
            columns,
            states=others,
            quantities=published,
            name="hierarchy_rhs",
        ),
        len(hierarchy_idx),
        reads=(recombination,),
        sparsity=_restrict(pattern, hierarchy_idx),
        name="hierarchy",
    )
    return model, Hierarchy(hierarchy, layout=(hierarchy, recombination))


class _PreparedSolve(NamedTuple):
    """A batch ready to launch: the compiled joint model, the endpoints, the rows and the initial state.

    The arrays are JAX arrays, traced under ``jit``. Rows go as ``row = k_idx
    * n_cosmo + cosmology_idx``, so consecutive rows share a wave mode, over
    the ascending ``k_sorted``.
    """

    cosmologies: tuple
    cosmology_model: ModelSpec
    model: Equations
    layout: ParameterLayout
    ode_fn: object  # the modax Hierarchy
    tau0: jax.Array  # (n_cosmologies,)
    lna_start: jax.Array
    z_re: jax.Array
    params: jax.Array  # (n_k * n_cosmologies, layout.n)
    y0: jax.Array  # (n_k * n_cosmologies, nvar)
    order: np.ndarray  # the caller's wave modes, sorted
    k_sorted: np.ndarray


def _prepare_solve(
    model: ModelSpec, cosmologies, k_values, extra=(), columns=None
) -> _PreparedSolve:
    """Return a batch packed for launch at ``k_values``: endpoints, parameter rows and initial states.

    The cosmologies' values may be traced under ``jit``. ``k_values`` must be
    numbers: they set the launch's shape and order. ``columns`` are the
    cosmologies' :func:`~discoeb.models._cosmology._columns` if already at hand. Every
    cosmology must suit
    the model (:meth:`ModelSpec.check
    <discoeb.models.ModelSpec.check>`), checked when its values are
    numbers. The kernel compiles once per model, and :func:`_pack` once per
    model, wave-mode grid and batch size.

    The derivative holds the endpoints, the parameter rows and the initial
    state fixed: :func:`~discoeb.solver._sensitivities._kept_values` sets the directed
    columns from its own inputs.
    """

    resolved = _spec_of(model).resolved()
    cosmologies = tuple(cosmologies)
    for cosmology in cosmologies:
        resolved.check(cosmology)
    joint, hierarchy = joint_kernel(resolved, extra)
    k = np.asarray(k_values, dtype=np.float64).reshape(-1)
    order = np.argsort(k, kind="stable")
    k_sorted = k[order]
    if columns is None:
        columns = _columns(cosmologies, keys=resolved.parameter_keys())
    columns = jax.lax.stop_gradient(columns)
    tau0, lna_start, z_re, params, y0 = _pack(
        resolved, tuple(extra), tuple(k_sorted.tolist()), columns
    )
    return _PreparedSolve(
        cosmologies,
        resolved,
        joint,
        parameter_layout(resolved, extra),
        hierarchy,
        tau0,
        lna_start,
        z_re,
        params,
        y0,
        order,
        k_sorted,
    )


@functools.partial(jax.jit, static_argnums=(0, 1, 2))
def _pack(model: ModelSpec, extra, k_sorted, columns):
    """Return ``(tau0, lna_start, z_re, params, y0)`` of a batch's columns, on the device.

    The background endpoints come from the vmapped quadrature
    (:func:`~discoeb.solver._background.background_endpoints`), the rows from
    :func:`_rows`, and the initial state from the joint model's initial
    conditions in ``jax.numpy``.
    """

    joint, _ = joint_kernel(model, extra)
    layout = parameter_layout(model, extra)
    tau0, lna_start, z_re, start = background_endpoints(columns, model)
    n_cosmo, n_k = tau0.shape[0], len(k_sorted)
    values = density_parameters(columns, model=model)
    params = jnp.tile(_rows(layout, values, tau0, z_re), (n_k, 1))
    k = jnp.repeat(jnp.asarray(k_sorted, dtype=jnp.float64), n_cosmo)
    params = params.at[:, layout.columns["k"]].set(k)
    params = params.at[:, layout.columns["cosmology_tag"]].set(
        jnp.tile(jnp.arange(n_cosmo, dtype=jnp.float64), n_k)
    )
    y0 = joint.initial_state(
        k,
        xp=jnp,
        tau=TAU_START,
        lna_start=jnp.tile(lna_start, n_k),
        **{name: jnp.tile(value, n_k) for name, value in values.items()},
    )
    # A background state of the model's own starts where the background's
    # integration from A_START has brought it.
    for name, value in start.items():
        y0 = y0.at[:, joint.index[name]].set(jnp.tile(value, n_k))
    return tau0, lna_start, z_re, params, y0


def _rows(layout: ParameterLayout, values, tau_end, z_re):
    """Return the cosmologies' parameter rows ``(n_cosmologies, layout.n)``, without the wave number.

    ``values`` are the model's parameters
    (:func:`~discoeb.solver._background.density_parameters`), each ``(n_cosmologies,)``.
    ``eps`` and the direction block are zero, as in every primal solve.
    """

    by_column = {column: values[name] for name, column in layout.values}
    by_column[layout.columns["tau_start"]] = TAU_START
    by_column[layout.columns["tau_end"]] = tau_end
    by_column[layout.columns["save_index"]] = 1.0
    by_column[layout.columns["z_re"]] = z_re
    n_cosmo = jnp.shape(tau_end)
    return jnp.stack(
        [
            jnp.broadcast_to(
                jnp.asarray(by_column.get(i, 0.0), jnp.float64),
                n_cosmo,
            )
            for i in range(layout.n)
        ],
        axis=1,
    )


def make_params(
    cosmology,
    k_values: np.ndarray,
    tau_end: float,
    z_re: float = 0.0,
    model: ModelSpec = DEFAULT_SPEC,
) -> np.ndarray:
    """Return packed parameters for the device right-hand side, one row per wave mode.

    The row is ``model``'s (:func:`parameter_layout`), by default
    :data:`PARAM_COLUMNS`.
    """

    columns = _columns((cosmology,), held=False, keys=model.parameter_keys())
    row = _rows(
        parameter_layout(model),
        density_parameters(columns, model=model),
        jnp.asarray([tau_end]),
        z_re,
    )
    params = np.tile(np.asarray(row), (len(k_values), 1))
    params[:, parameter_layout(model).columns["k"]] = k_values
    return params


def make_initial_states(
    cosmology,
    k_values: np.ndarray,
    model,
    lna_start,
    cosmology_model: ModelSpec = DEFAULT_SPEC,
) -> np.ndarray:
    """Return adiabatic initial states plus the fully ionized thermal state, ``(n_k, nvar)``.

    ``cosmology`` is one cosmology or a sequence of ``len(k_values)``, one per
    trajectory. ``model`` is the joint component model and
    ``cosmology_model`` the model whose parameters the initial conditions
    read. ``lna_start`` is a number or one per trajectory. The initial
    conditions broadcast over the trajectories.
    """

    columns = _columns(
        _as_batch(cosmology), held=False, keys=cosmology_model.parameter_keys()
    )
    inputs = density_parameters(columns, model=cosmology_model)

    def per(values):
        values = np.asarray(values)
        return values[0] if values.shape == (1,) else values

    return model.initial_state(
        k_values,
        tau=TAU_START,
        lna_start=np.asarray(lna_start, dtype=np.float64),
        **{name: per(value) for name, value in inputs.items()},
    )


def error_weights_for(
    prepared, n_traj: int, base=None, thermo_weight: float = THERMO_ERROR_WEIGHT
):
    """Return the solver's error weights ``(n_traj, nvar)``.

    ``base`` (or ones) weights the hierarchy and ``thermo_weight`` the
    recombination block. The optical depth gets the block's weight too, so its
    error counts against ``rtol |kappa|``. The block's steps then resolve its
    increments where ``kappa`` is of order one, through recombination and after,
    which is where ``exp(-kappa)`` reads them. Unweighted, it left the lensed TT
    0.5 percent off CLASS at ``l = 1500``.
    """

    model = prepared.model
    weights = np.ones((n_traj, model.nvar), dtype=np.float64)
    if base is not None:
        base = np.asarray(base, dtype=np.float64)
        weights[:, : base.shape[-1]] = base
    weights[:, list(thermal_indices(model))] = thermo_weight
    return weights


def _named_error_weights(weights, model: ModelSpec, k_launch):
    """Turn ``{state name: weight}`` into solver error weights ``(n_traj, nvar)`` in launch order.

    Each weight is a number or a function of the trajectories' wave numbers
    (``k_launch``, launch order) that returns one per trajectory. Unnamed states
    keep weight one. The names are the model's
    (:meth:`~discoeb.models.ModelSpec.state_names`), so a caller never needs the
    state's index order or the launch's trajectory order.
    """

    names = model.state_names()
    unknown = [n for n in weights if n not in names]
    if unknown:
        raise KeyError(f"{unknown} are not states of {model.name}: {names[:6]} ...")
    k_launch = np.asarray(k_launch, dtype=np.float64)
    out = np.ones((len(k_launch), len(names)))
    for name, weight in weights.items():
        out[:, names.index(name)] = weight(k_launch) if callable(weight) else weight
    return out


def _launch(
    prepared,
    params,
    y0,
    t_span,
    *,
    rtol,
    atol,
    first_step,
    max_steps,
    error_weights=None,
    thermo_weight=THERMO_ERROR_WEIGHT,
    lu_precision="fp32",
    max_registers=CUDA_MAX_REGISTERS,
    **extra,
):
    """Run one modax Rodas5P launch of the prepared hierarchy over the packed trajectories.

    The method object carries the numerics: the hierarchy's LU precision and the
    PI controller's gains. The recombination block, which the hierarchy reads,
    always factorises in double. Each trajectory gets its own row of save
    times, the shared ``t_span`` clipped to its ``tau0`` in ``tau_end``, so
    it stops there and its later saves hold its final state.
    """

    from modax import PIDGains, Rodas5P
    from modax import solve as modax_solve

    method = Rodas5P(lu_precision=lu_precision, gains=PIDGains(pcoeff=0.3, icoeff=0.4))
    # A tuple is the row split into leaves, the first holding tau_end.
    head = params[0] if isinstance(params, tuple) else params
    ix_tau_end = prepared.layout.ix_tau_end
    if isinstance(head, np.ndarray):
        t_rows = np.minimum(np.asarray(t_span)[None, :], head[:, ix_tau_end, None])
    else:
        tau_end = jax.lax.stop_gradient(head[:, ix_tau_end, None])
        t_rows = jnp.minimum(jnp.asarray(t_span)[None, :], tau_end)
    return modax_solve(
        prepared.ode_fn,
        y0,
        t_rows,
        params,
        method=method,
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
        trajectories_per_block=TRAJECTORIES_PER_BLOCK,
        max_registers=max_registers,
        error_weights=error_weights_for(
            prepared, np.shape(head)[0], error_weights, thermo_weight
        ),
        cache=True,
        **extra,
    )


def _solve_batch(
    model: ModelSpec,
    cosmologies,
    k_values,
    tau_save=None,
    *,
    rtol: float = PERTURB_RTOL,
    atol: float = PERTURB_ATOL,
    first_step: float = PERTURB_FIRST_STEP,
    max_steps: int = PERTURB_MAX_STEPS,
    **options,
):
    """Solve a batch of cosmologies at ``k_values`` in one launch.

    Prepares the batch (:func:`_prepare_solve`) and launches it
    (:func:`_launch`, which takes ``options``). ``tau_save`` defaults to the
    two endpoints. Returns ``(solution, prepared, k_sorted)``. ``solution`` is
    whatever modax returns, with rows in ``k_idx * n_cosmo + cosmology_idx``
    order over the ascending ``k_sorted``.
    """

    prepared = _prepare_solve(model, cosmologies, k_values)
    if tau_save is None:
        t_span = jnp.stack([jnp.asarray(TAU_START), jnp.max(prepared.tau0)])
    else:
        t_span = jnp.asarray(tau_save, dtype=jnp.float64)
    solution = _launch(
        prepared,
        prepared.params,
        prepared.y0,
        t_span,
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
        **options,
    )
    return solution, prepared, prepared.k_sorted


def build_keep_hook(model, names, layout: ParameterLayout = DEFAULT_LAYOUT):
    """Return a modax save hook ``(y, tau, p) -> tuple`` of the quantities ``names``.

    A name is a state element or a quantity the joint model provides. State
    elements (``clxc``, ``theta_2``, ``psi_0_1``: the
    :meth:`~discoeb.models.ModelSpec.state_names`, or ``lna``, ``thermo_0`` ...
    ``kappa`` of the recombination block) come straight from the state. Provided
    quantities (``R``, ``sigma``, ``phi`` with the line-of-sight sources
    appended, ``opacity`` or ``cs2`` from the thermal states) come from
    :meth:`~discoeb.components._component.Equations.build_function`, with the state, the time and
    the parameter row of ``layout`` as inputs. The hook clamps the time to the trajectory's
    own ``tau_end``: a save past a cosmology's ``tau0`` holds its final state,
    and the background at ``tau0`` goes with it. ``hook._generated_source``
    holds its generated source.

    modax keeps what it returns at every save, and differentiates it like the
    history: a forward-sensitivity launch stores each quantity's derivative
    along every sensitivity column beside it, one Enzyme sweep of the hook per
    column.
    """

    names = tuple(names)
    if not names:
        raise ValueError("keep at least one quantity")
    vectors = [n for n in names if n in model.states and n not in model.index]
    if vectors:
        raise ValueError(
            f"{vectors} are vector states: name their elements ({', '.join(model.states[vectors[0]].elements[:3])}, ...)"
        )
    provided = [n for n in names if n not in model.index]
    namespace: dict = {}
    lines = [
        "def quantities(y, tau, p):",
        f"    t = p[{layout.ix_tau_end}]",
        "    if tau < t:",
        "        t = tau",
    ]
    if provided:
        # build_function returns a lone output bare, so pair it with tau.
        outputs = provided if len(provided) > 1 else [*provided, "tau"]
        namespace["kept"] = model.build_function(
            outputs, inputs=("y", "tau"), params=layout.columns, name="kept"
        )
        targets = ", ".join(f"q_{i}" for i in range(len(outputs)))
        lines.append(f"    {targets} = kept(y, t, p)")
    values = [
        f"y[{model.index[name]}]"
        if name in model.index
        else f"q_{provided.index(name)}"
        for name in names
    ]
    lines.append(f"    return ({', '.join(values)},)")
    return _compile("quantities", "\n".join(lines) + "\n", namespace)


@functools.lru_cache(maxsize=None)
def _keep_hook(model: ModelSpec, extra, names):
    """Return the compiled hook for a resolved model, extra components and names (cached)."""

    return build_keep_hook(
        joint_model(model, extra), names, _parameter_layout(model, extra)
    )
