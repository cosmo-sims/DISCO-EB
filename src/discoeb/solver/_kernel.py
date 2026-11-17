"""
Constructs a modax GPU kernel.

One launch integrates every (wave mode, cosmology) pair of a batch. The
recombination block steps ahead of the perturbations as a modax
:class:`~modax.Hierarchy`. This module builds that hierarchy, packs a batch
into parameter rows and initial states, launches it and compiles the save
hooks that keep named quantities.
"""

import functools
import math
import os
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..components._component import Equations, _compile, derive_parameters
from ..components.recombination import THERMO_ERROR_WEIGHT, Recombination
from ..models._model import Model, ModelSpec, _batch_of, _spec_of

A_START = 2.0e-7
"""The scale factor where every trajectory starts, deep in radiation
domination (``tau`` near 0.1 Mpc). The solve integrates in ``a`` from here
to 1, today."""

PERTURB_RTOL = 1.0e-4
PERTURB_ATOL = 1.0e-4
PERTURB_FIRST_STEP = A_START  # in a: 0.1 Mpc of conformal time
PERTURB_MAX_STEPS = 20000

SOLVER_COLUMNS = ("k",)
"""The row's column the solver sets rather than a component: the wave number."""

DIRECTED_NAMES = (
    "grhog",
    "grhornomass",
    "grhoc",
    "grhob",
    "grhov",
    "YHe",
    "Tcmb",
)
"""The directed inputs every built-in model leads with: the five background
densities and the recombination rates' ``YHe`` and ``Tcmb``. A model's others (the curvature density, the dark-energy
parameters, the massive neutrinos' density and mass) follow in component
order (:func:`parameter_layout`)."""
N_COMMON_DIRECTED = len(DIRECTED_NAMES)
"""The number of :data:`DIRECTED_NAMES`, which ``solve_eb`` also takes alone
as ``inputs``."""


@dataclass(frozen=True)
class ParameterLayout:
    """The columns of a model's parameter row, generated from its parameters.

    The row is ``(the directed parameters, SOLVER_COLUMNS, the other column
    parameters)``. The directed parameters come in :data:`DIRECTED_NAMES`
    order, then in component order, and the other columns in component and
    declaration order, so the layout is a function of the resolved model and
    the kernels cached on it stay valid. A derivative moves the directed
    columns alone: modax integrates the one sensitivity column along their
    tangent (:func:`~discoeb.solver._sensitivities._kept_values`).
    """

    columns: dict
    """Column by the name the providers read it under: ``grhoc``, ``k``."""
    directed: tuple[str, ...]
    """The directed parameters: the inputs a derivative moves."""
    values: tuple[tuple[str, int], ...]
    """``(parameter, column)`` of the parameters the row holds."""

    @property
    def n(self) -> int:
        return len(self.columns)

    @property
    def directed_columns(self) -> tuple[int, ...]:
        """The columns of :attr:`directed`."""
        return tuple(self.columns[name] for name in self.directed)


def parameter_layout(model: type[Model] | ModelSpec, extra=()) -> ParameterLayout:
    """Return the row layout of ``model`` with the component classes ``extra`` appended (cached)."""

    return _parameter_layout(_spec_of(model).resolved(), tuple(extra))


@functools.lru_cache(maxsize=None)
def _parameter_layout(model: ModelSpec, extra) -> ParameterLayout:
    parameters = model.parameters(extra)
    directed = [n for n, p in parameters.items() if p.directed]
    directed.sort(
        key=lambda n: (
            DIRECTED_NAMES.index(n) if n in DIRECTED_NAMES else len(DIRECTED_NAMES)
        )
    )
    others = [n for n, p in parameters.items() if p.column and not p.directed]
    clash = sorted(set(SOLVER_COLUMNS) & set(parameters))
    if clash:
        raise ValueError(f"{clash} name solver columns, not parameters")
    names = [*directed, *SOLVER_COLUMNS, *others]
    columns = {name: i for i, name in enumerate(names)}
    values = tuple((n, columns[n]) for n in (*directed, *others))
    return ParameterLayout(columns, tuple(directed), values)


TRAJECTORIES_PER_BLOCK = 32
"""Wave modes packed into one CUDA block by modax's thread-local Rodas5P kernel."""

CUDA_MAX_REGISTERS = int(os.environ.get("DISCO_CUDA_MAX_REGISTERS", "168"))
"""Per-thread register cap for that kernel: fewer registers, more resident warps."""


def joint_model(model: type[Model] | ModelSpec, extra=()) -> Equations:
    """Return the components of ``model`` with :class:`~discoeb.components.recombination.Recombination` appended.

    The recombination states come after the perturbations', so the
    perturbation state keeps the indices of :meth:`ModelSpec.state_names
    <discoeb.models.ModelSpec.state_names>`. The five thermal states
    (the conformal time, ``x_H``, ``x_He``, ``T_m`` and the optical depths)
    are the last columns of every history, at ``model.index["tau"]`` and so
    on. ``extra`` lists component classes to append after them (the
    line-of-sight sources). The joint model integrates in the scale factor:
    its time is ``a`` and its clock ``adot = a'``, the rate of ``a`` in the
    conformal time the components write their rates in. Every cosmology
    then runs from :data:`A_START` to 1, and since ``a`` grows as ``tau``
    in the radiation era, the early solutions are the low-order polynomials
    they are in ``tau``. In ``ln a`` they are exponentials, which the same
    tolerance integrated 13 times less exactly (2026-11-17).
    """

    return _joint_model(_spec_of(model), tuple(extra))


@functools.lru_cache(maxsize=None)
def _joint_model(spec: ModelSpec, extra) -> Equations:
    def build(last=()):
        return spec.build(
            extra=[Recombination(), *(cls() for cls in extra)],
            time="a",
            clock="adot",
            last=last,
        )

    # modax lays the two blocks out one after the other, the perturbations
    # first. The recombination block's states are the last declared, but a
    # model's own background states (a scalar field's) are not: put every
    # background state last. Until 2026-11-17 they kept their place, so the
    # kernel evolved the field in the slots the hook read as its
    # perturbation, and the other way round.
    model = build()
    background = model.stages()[0]
    indices = model.state_indices(background)
    if indices != list(range(model.nvar - len(indices), model.nvar)):
        model = build(last=background)
    return model


def density_parameters(param, model: ModelSpec, xp=jnp) -> dict:
    """The parameters of ``model`` for a parameter dictionary, keyed as the components name them.

    The keys pass through, and every derived parameter comes from its
    declaration (:class:`~discoeb.components._component.Parameter`): ``grhog``,
    ``grhornomass``, ``grhoc``, ``grhob``, ``grhov`` and ``grhok`` the
    photons, massless neutrinos, cold dark matter, baryons, dark energy and
    curvature, ``grhor_nu``, ``n_mnu`` and ``amnu`` the massive neutrinos.
    Pass ``xp=jnp`` for a traced cosmology and ``xp=np`` for one given as floats.
    """

    parameters = model.parameters()
    return derive_parameters(parameters, param, math=math if xp is np else jnp)


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
    state whose rate reads no wave number: the conformal time, the recombination
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
            inputs=("y", "a"),
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
    """A batch ready to launch: the joint model, its hierarchy, the rows and the initial state.

    The arrays are JAX arrays, traced under ``jit``. Rows go as ``row = k_idx
    * n_cosmo + cosmology_idx``, so consecutive rows share a wave mode, over
    the ascending ``k_sorted``.
    """

    cosmologies: tuple
    cosmology_model: ModelSpec
    model: Equations
    layout: ParameterLayout
    ode_fn: object  # the modax Hierarchy
    params: jax.Array  # (n_k * n_cosmologies, layout.n)
    y0: jax.Array  # (n_k * n_cosmologies, nvar)
    order: np.ndarray  # the caller's wave modes, sorted
    k_sorted: np.ndarray


def _prepare_solve(
    model: ModelSpec, cosmologies, k_values, extra=(), columns=None
) -> _PreparedSolve:
    """Return a batch packed for launch at ``k_values``: parameter rows and initial states.

    ``cosmologies`` are the batch's parameter dictionaries
    (:func:`~discoeb.models._model._batch_of`); their values may be traced under ``jit``. ``k_values`` must be
    numbers: they set the launch's shape and order. ``columns`` are the
    cosmologies' :func:`_columns` if already at hand. Every
    cosmology must suit
    the model (:meth:`ModelSpec.check
    <discoeb.models.ModelSpec.check>`), checked when its values are
    numbers. The kernel compiles once per model, and :func:`_pack` once per
    model, wave-mode grid and batch size.

    The derivative holds the parameter rows and the initial state fixed: :func:`~discoeb.solver._sensitivities._kept_values` sets the directed
    columns from its own inputs.
    """

    resolved = _spec_of(model).resolved()
    for cosmology in cosmologies:
        resolved.check(cosmology)
    joint, hierarchy = joint_kernel(resolved, extra)
    k = np.asarray(k_values, dtype=np.float64).reshape(-1)
    order = np.argsort(k, kind="stable")
    k_sorted = k[order]
    if columns is None:
        columns = _columns(cosmologies, resolved)
    columns = jax.lax.stop_gradient(columns)
    params, y0 = _pack(resolved, tuple(extra), tuple(k_sorted.tolist()), columns)
    return _PreparedSolve(
        cosmologies,
        resolved,
        joint,
        parameter_layout(resolved, extra),
        hierarchy,
        params,
        y0,
        order,
        k_sorted,
    )


@functools.partial(jax.jit, static_argnums=(0, 1, 2))
def _pack(model: ModelSpec, extra, k_sorted, columns):
    """Return ``(params, y0)`` of a batch's columns, on the device.

    The rows come from :func:`_rows` and the initial state from the joint
    model's initial conditions at :data:`A_START`, in ``jax.numpy``.
    Nothing is integrated: the conformal time there is a closed form, and
    a model's own background states start from their initial conditions.
    """

    joint = joint_model(model, extra)
    layout = parameter_layout(model, extra)
    values = density_parameters(columns, model=model)
    n_cosmo = next(iter(columns.values())).shape[0]
    n_k = len(k_sorted)
    params = jnp.tile(_rows(layout, values, n_cosmo), (n_k, 1))
    k = jnp.repeat(jnp.asarray(k_sorted, dtype=jnp.float64), n_cosmo)
    params = params.at[:, layout.columns["k"]].set(k)
    y0 = joint.initial_state(
        k,
        xp=jnp,
        a=A_START,
        **{name: jnp.tile(value, n_k) for name, value in values.items()},
    )
    return params, y0


@functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
def _not_differentiable(value, name):
    """Return ``value``, and raise if anything differentiates with respect to it."""

    return value


@_not_differentiable.defjvp
def _not_differentiable_jvp(name, primals, tangents):
    raise ValueError(
        f"{name!r} cannot be differentiated through solve_eb of this model. The "
        "differentiable parameters are those its components declare "
        "differentiable (differentiable_parameters(model))"
    )


def _columns(cosmologies, model: type[Model] | ModelSpec, held: bool = True) -> dict:
    """Return a batch's parameters as JAX columns ``(n_cosmologies,)`` by key.

    ``cosmologies`` are the batch's parameter dictionaries
    (:func:`~discoeb.models._model._batch_of`), their values numbers or JAX
    values, traced or not. With ``held``, the columns are the variables of
    a derivative. A key the solve never reads (``solved=False``) then
    carries no tangent, and a key it reads but the model does not
    differentiate (:meth:`ModelSpec.differentiable_parameters
    <discoeb.models.ModelSpec.differentiable_parameters>`) raises if
    differentiated, where its derivative would come back as a silent zero.
    So does a traced key of another model that a cosmology holds
    (:meth:`Model.from_dict <discoeb.models.Model.from_dict>`), which the
    kernel never reads.
    """

    spec = _spec_of(model)
    declared = spec.parameters()
    differentiable = spec.differentiable_parameters()
    keys = spec.parameter_keys()
    foreign = sorted({key for c in cosmologies for key in c} - set(keys))
    columns = {}
    for key in (*keys, *foreign):
        values = [c.get(key, 0.0) if key in foreign else c[key] for c in cosmologies]
        if any(isinstance(v, jax.Array) for v in values):
            column = jnp.stack([jnp.asarray(v, dtype=jnp.float64) for v in values])
        else:  # one device array, not one per cosmology
            column = jnp.asarray(np.asarray(values, dtype=np.float64))
        if held and key in keys and not declared[key].solved:
            column = jax.lax.stop_gradient(column)
        elif held and key not in differentiable:
            column = _not_differentiable(column, key)
        columns[key] = column
    return columns


def _rows(layout: ParameterLayout, values, n_cosmo: int):
    """Return the cosmologies' parameter rows ``(n_cosmologies, layout.n)``, without the wave number.

    ``values`` are the model's parameters (:func:`density_parameters`), each
    ``(n_cosmologies,)``.
    """

    by_column = {column: values[name] for name, column in layout.values}
    return jnp.stack(
        [
            jnp.broadcast_to(
                jnp.asarray(by_column.get(i, 0.0), jnp.float64),
                (n_cosmo,),
            )
            for i in range(layout.n)
        ],
        axis=1,
    )


def error_weights_for(
    prepared,
    n_traj: int,
    base=None,
    thermo_weight: float = THERMO_ERROR_WEIGHT,
    reio_weight: float | None = None,
):
    """Return the solver's error weights ``(n_traj, nvar)``.

    ``base`` (or ones) weights the hierarchy and ``thermo_weight`` the
    recombination block, and ``reio_weight`` (if given) ``kappa_reio``. The optical
    depth gets the block's weight too, so its
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
    if reio_weight is not None and "kappa_reio" in model.index:
        weights[:, model.index["kappa_reio"]] = reio_weight
    return weights


def _named_error_weights(weights, joint: Equations, k_launch):
    """Turn ``{state name: weight}`` into solver error weights ``(n_traj, nvar)`` of the joint state, in launch order.

    Each weight is a number or a function of the trajectories' wave numbers
    (``k_launch``, launch order) that returns one per trajectory. Unnamed states
    keep weight one. The names are the state elements of the joint model
    (:meth:`~discoeb.models.ModelSpec.state_names`, and the background's), so
    a caller never needs the state's index order or the launch's trajectory
    order.
    """

    index = joint.index
    unknown = [n for n in weights if n not in index]
    if unknown:
        raise KeyError(f"{unknown} are not states of the model: {joint.state[:6]} ...")
    k_launch = np.asarray(k_launch, dtype=np.float64)
    out = np.ones((len(k_launch), len(index)))
    for name, weight in weights.items():
        out[:, index[name]] = weight(k_launch) if callable(weight) else weight
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
    reio_weight=None,
    lu_precision="fp32",
    max_registers=CUDA_MAX_REGISTERS,
    **extra,
):
    """Run one modax Rodas5P launch of the prepared hierarchy over the packed trajectories.

    The method object carries the numerics: the hierarchy's LU precision and the
    PI controller's gains. The recombination block, which the hierarchy reads,
    always factorises in double. Every trajectory runs over the same span
    of the scale factor, ``t_span``.
    """

    from modax import PIDGains, Rodas5P
    from modax import solve as modax_solve

    method = Rodas5P(lu_precision=lu_precision, gains=PIDGains(pcoeff=0.3, icoeff=0.4))
    return modax_solve(
        prepared.ode_fn,
        y0,
        t_span,
        params,
        method=method,
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
        trajectories_per_block=TRAJECTORIES_PER_BLOCK,
        max_registers=max_registers,
        error_weights=error_weights_for(
            prepared, np.shape(params)[0], error_weights, thermo_weight, reio_weight
        ),
        cache=True,
        **extra,
    )


def _solve_batch(
    model: ModelSpec,
    cosmologies,
    k_values,
    a_save=None,
    *,
    rtol: float = PERTURB_RTOL,
    atol: float = PERTURB_ATOL,
    first_step: float = PERTURB_FIRST_STEP,
    max_steps: int = PERTURB_MAX_STEPS,
    **options,
):
    """Solve a batch of cosmologies at ``k_values`` in one launch.

    Prepares the batch (:func:`_prepare_solve`) and launches it
    (:func:`_launch`, which takes ``options``). ``a_save`` defaults to the
    two endpoints, :data:`A_START` and 1. Returns ``(solution, prepared, k_sorted)``. ``solution`` is
    whatever modax returns, with rows in ``k_idx * n_cosmo + cosmology_idx``
    order over the ascending ``k_sorted``.
    """

    prepared = _prepare_solve(model, _batch_of(cosmologies)[1], k_values)
    if a_save is None:
        t_span = jnp.asarray([A_START, 1.0])
    else:
        t_span = jnp.asarray(a_save, dtype=jnp.float64)
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


def build_keep_hook(model, names, layout: ParameterLayout):
    """Return a modax save hook ``(y, a, p) -> tuple`` of the quantities ``names``.

    A name is a state element or a quantity the joint model provides. State
    elements (``clxc``, ``theta_2``, ``psi_0_1``: the
    :meth:`~discoeb.models.ModelSpec.state_names`, or ``tau``, ``thermo_0`` ...
    ``kappa`` of the recombination block) come straight from the state. Provided
    quantities (``R``, ``sigma``, ``phi`` with the line-of-sight sources
    appended, ``opacity`` or ``cs2`` from the thermal states) come from
    :meth:`~discoeb.components._component.Equations.build_function`, with the state, the time and
    the parameter row of ``layout`` as inputs. ``hook._generated_source``
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
    lines = ["def quantities(y, a, p):"]
    if provided:
        # build_function returns a lone output bare, so pair it with the time.
        outputs = provided if len(provided) > 1 else [*provided, "a"]
        namespace["kept"] = model.build_function(
            outputs, inputs=("y", "a"), params=layout.columns, name="kept"
        )
        targets = ", ".join(f"q_{i}" for i in range(len(outputs)))
        lines.append(f"    {targets} = kept(y, a, p)")
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
