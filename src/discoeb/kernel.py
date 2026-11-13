"""The joint kernel: the parameter row, the packed batch and the modax launch.

One modax launch integrates every (wave mode, cosmology) pair of a batch. The
joint model is the perturbation model with the recombination block
(:class:`~discoeb.recombination.Recombination`) stepping ahead of it as a modax
:class:`~modax.Hierarchy`. This module builds that hierarchy, packs a batch
into parameter rows and initial states, launches it, and compiles the save
hooks that keep named quantities.
"""

import functools
import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .background import TAU_START, background_endpoints, density_parameters
from .component import Model, _compile
from .cosmology import _as_batch, _columns
from .models import CosmologyModel
from .recombination import THERMAL_PARAMETERS, THERMO_ERROR_WEIGHT, Recombination

PERTURB_RTOL = 1.0e-4
PERTURB_ATOL = 1.0e-4
PERTURB_FIRST_STEP = 1.0e-2
PERTURB_MAX_STEPS = 20000

# Columns of the packed parameter row the device right-hand side reads:
#   p = (grhog, grhornomass, grhoc, grhob, grhov,
#        tau_start, tau_end, k, cosmology_idx, save_index,
#        w_DE_0, w_DE_a, cs2_DE, grhok, grhor_nu, n_mnu, amnu,
#        YHe, Tcmb, H0, Omegab, z_re,
#        eps, dir_grhog, dir_grhornomass, dir_grhoc, dir_grhob, dir_grhov, dir_YHe, dir_Tcmb, dir_H0, dir_Omegab)
# grhok = grhom * Omega_k is the constant comoving curvature density
# coefficient 8 pi G rho_K a^2, zero for a flat cosmology. The recombination
# rates and reionization read the five columns after ``amnu``
# (:data:`discoeb.recombination.THERMAL_PARAMETERS`). The model reads each input
# a derivative may move (:data:`DIRECTED_COLUMNS`: the five densities, ``YHe``,
# ``Tcmb``, ``H0``, ``Omegab``) as ``x_nominal + eps dir_x``. ``eps`` is zero in
# every primal solve, and the derivative in ``eps`` is the directional
IX_TAU_START = 5
IX_TAU_END = 6
IX_K = 7
IX_COSMOLOGY = 8
IX_SAVE_INDEX = 9
IX_W_DE_0 = 10
IX_W_DE_A = 11
IX_CS2_DE = 12
IX_GRHOK = 13
IX_GRHOR_NU = 14
IX_NMNU = 15
IX_AMNU = 16
N_PARAM_HIERARCHY = IX_AMNU + 1
IX_YHE = N_PARAM_HIERARCHY
IX_TCMB = N_PARAM_HIERARCHY + 1
IX_H0 = N_PARAM_HIERARCHY + 2
IX_OMEGAB = N_PARAM_HIERARCHY + 3
IX_EPS = N_PARAM_HIERARCHY + len(THERMAL_PARAMETERS)
IX_DIR = IX_EPS + 1
DIRECTED_NAMES = (
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
"""The inputs a trajectory's sensitivity direction may move, in the order of :data:`DIRECTED_COLUMNS`."""
DIRECTED_COLUMNS = (0, 1, 2, 3, 4, IX_YHE, IX_TCMB, IX_H0, IX_OMEGAB)
"""The columns of :data:`DIRECTED_NAMES`: their nominal values, with ``eps`` and the ``dir`` block after the row."""
N_PARAM = IX_DIR + len(DIRECTED_COLUMNS)

PARAM_COLUMNS = {
    "grhog_nominal": 0,
    "grhornomass_nominal": 1,
    "grhoc_nominal": 2,
    "grhob_nominal": 3,
    "grhov_nominal": 4,
    "tau_start": IX_TAU_START,
    "tau_end": IX_TAU_END,
    "k": IX_K,
    "cosmology_tag": IX_COSMOLOGY,
    "save_index": IX_SAVE_INDEX,
    "w_DE_0": IX_W_DE_0,
    "w_DE_a": IX_W_DE_A,
    "cs2_DE": IX_CS2_DE,
    "grhok": IX_GRHOK,
    "grhor_nu": IX_GRHOR_NU,
    "n_mnu": IX_NMNU,
    "amnu": IX_AMNU,
    **{name: N_PARAM_HIERARCHY + i for i, name in enumerate(THERMAL_PARAMETERS)},
    "eps": IX_EPS,
    **{f"dir_{name}": IX_DIR + i for i, name in enumerate(DIRECTED_NAMES)},
}
"""The row's columns by the names the component model's providers take them under
(:mod:`discoeb.species`, :mod:`discoeb.recombination`)."""

TRAJECTORIES_PER_BLOCK = 32
"""Wave modes packed into one CUDA block by modax's thread-local Rodas5P kernel."""

CUDA_MAX_REGISTERS = int(os.environ.get("DISCO_CUDA_MAX_REGISTERS", "168"))
"""Per-thread register cap for that kernel: fewer registers, more resident warps."""


def joint_model(model: CosmologyModel, extra=()) -> Model:
    """Return the components of ``model`` with :class:`~discoeb.recombination.Recombination` appended.

    The recombination states come after the perturbations', so the
    perturbation state keeps the indices of :meth:`CosmologyModel.state_names
    <discoeb.models.CosmologyModel.state_names>`. The five thermal states
    (``ln a``, ``x_H``, ``x_He``, ``T_m`` and the optical depth) are the last
    columns of every history, at ``model.index["lna"]`` and so on. ``extra``
    lists component classes to append after them (the line-of-sight sources).
    """

    return model.build(extra=[Recombination(), *(cls() for cls in extra)])


def thermal_indices(model) -> tuple[int, ...]:
    """Return the flat indices of the five recombination states in ``model``."""

    return tuple(
        model.index[n] for n in ("lna", "thermo_0", "thermo_1", "thermo_2", "kappa")
    )


THERMAL_STATES = ("lna", "thermo", "kappa")
"""The recombination block's states, in model order."""

THERMAL_QUANTITIES = ("a", "opacity", "cs2", "rhonu")
"""What the hierarchy reads of the recombination block: the scale factor, the
opacity, the baryon sound speed and the massive-neutrino density ratio. The
block publishes them and modax interpolates them between the block's steps."""


def _restrict(pattern: np.ndarray, indices) -> np.ndarray:
    """Return the ``(nnz, 2)`` pattern's rows and columns ``indices``, renumbered."""

    position = {idx: k for k, idx in enumerate(indices)}
    kept = [
        (position[r], position[c])
        for r, c in pattern.tolist()
        if r in position and c in position
    ]
    return np.array(sorted(kept), dtype=np.int64).reshape(-1, 2)


def joint_kernel(model: CosmologyModel, extra=()):
    """Return ``(joint model, hierarchy)`` of ``model``, cached per resolved model and ``extra``.

    The hierarchy is a modax :class:`~modax.Hierarchy` of two blocks. The
    recombination block (:data:`THERMAL_STATES`) publishes
    :data:`THERMAL_QUANTITIES`, and the perturbation block's right-hand side
    takes them as given. modax steps the recombination block ahead of the
    perturbations and hands the quantities down as a quintic Hermite
    interpolant. So the perturbation Jacobian has no thermal columns, and the
    thermal providers run once per recombination step. The state keeps the
    joint model's order, thermal states last.
    """

    return _joint_kernel(model.resolved(), tuple(extra))


@functools.lru_cache(maxsize=None)
def _joint_kernel(cosmology_model: CosmologyModel, extra):
    from modax import Block, Hierarchy

    model = joint_model(cosmology_model, extra)
    pattern = model.sparsity()
    thermal = model.state_indices(THERMAL_STATES)
    others = [name for name in model.states if name not in THERMAL_STATES]
    hierarchy_idx = model.state_indices(others)
    recombination = Block(
        model.build(PARAM_COLUMNS, states=THERMAL_STATES, name="recombination_rhs"),
        len(thermal),
        outputs=model.build_function(
            THERMAL_QUANTITIES,
            inputs=("y", "tau"),
            params=PARAM_COLUMNS,
            states=THERMAL_STATES,
            name="thermal_quantities",
        ),
        n_outputs=len(THERMAL_QUANTITIES),
        sparsity=_restrict(pattern, thermal),
        name="recombination",
    )
    hierarchy = Block(
        model.build(
            PARAM_COLUMNS,
            states=others,
            quantities=THERMAL_QUANTITIES,
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
    cosmology_model: CosmologyModel
    model: Model
    ode_fn: object  # the modax Hierarchy
    tau0: jax.Array  # (n_cosmologies,)
    lna_start: jax.Array
    z_re: jax.Array
    params: jax.Array  # (n_k * n_cosmologies, N_PARAM)
    y0: jax.Array  # (n_k * n_cosmologies, nvar)
    order: np.ndarray  # the caller's wave modes, sorted
    k_sorted: np.ndarray


def _prepare_solve(
    model: CosmologyModel, cosmologies, k_values, extra=(), columns=None
) -> _PreparedSolve:
    """Return a batch packed for launch at ``k_values``: endpoints, parameter rows and initial states.

    The cosmologies' values may be traced under ``jit``. ``k_values`` must be
    numbers: they set the launch's shape and order. ``columns`` are the
    cosmologies' :func:`~discoeb.cosmology._columns` if already at hand. Every
    cosmology must suit
    the model (:meth:`CosmologyModel.check
    <discoeb.models.CosmologyModel.check>`), checked when its values are
    numbers. The kernel compiles once per model, and :func:`_pack` once per
    model, wave-mode grid and batch size.

    The derivative holds the endpoints, the parameter rows and the initial
    state fixed: :func:`~discoeb.sensitivities._kept_values` sets the directed
    columns from its own inputs.
    """

    resolved = model.resolved()
    cosmologies = tuple(cosmologies)
    for cosmology in cosmologies:
        resolved.check(cosmology)
    joint, hierarchy = joint_kernel(resolved, extra)
    k = np.asarray(k_values, dtype=np.float64).reshape(-1)
    order = np.argsort(k, kind="stable")
    k_sorted = k[order]
    if columns is None:
        columns = _columns(cosmologies)
    columns = jax.lax.stop_gradient(columns)
    tau0, lna_start, z_re, params, y0 = _pack(
        resolved, tuple(extra), tuple(k_sorted.tolist()), columns
    )
    return _PreparedSolve(
        cosmologies,
        resolved,
        joint,
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
def _pack(model: CosmologyModel, extra, k_sorted, columns):
    """Return ``(tau0, lna_start, z_re, params, y0)`` of a batch's columns, on the device.

    The background endpoints come from the vmapped quadrature
    (:func:`~discoeb.background.background_endpoints`), the rows from
    :func:`_rows`, and the initial state from the joint model's initial
    conditions in ``jax.numpy``.
    """

    joint, _ = joint_kernel(model, extra)
    tau0, lna_start, z_re = background_endpoints(columns)
    n_cosmo, n_k = tau0.shape[0], len(k_sorted)
    densities = density_parameters(columns)
    params = jnp.tile(_rows(columns, densities, tau0, z_re), (n_k, 1))
    k = jnp.repeat(jnp.asarray(k_sorted, dtype=jnp.float64), n_cosmo)
    params = params.at[:, IX_K].set(k)
    params = params.at[:, IX_COSMOLOGY].set(
        jnp.tile(jnp.arange(n_cosmo, dtype=jnp.float64), n_k)
    )
    inputs = _initial_inputs(columns, densities)
    y0 = joint.initial_state(
        k,
        xp=jnp,
        tau=TAU_START,
        lna_start=jnp.tile(lna_start, n_k),
        **{name: jnp.tile(value, n_k) for name, value in inputs.items()},
    )
    return tau0, lna_start, z_re, params, y0


_NAME_OF_COLUMN = {column: name for name, column in PARAM_COLUMNS.items()}


def _rows(columns, densities, tau_end, z_re):
    """Return the cosmologies' parameter rows ``(n_cosmologies, N_PARAM)``, without the wave number.

    ``densities`` is :func:`~discoeb.background.density_parameters` of the
    columns. ``eps`` and the direction block are zero, as in every primal solve.
    """

    named = {
        **{
            f"{name}_nominal": densities[name]
            for name in ("grhog", "grhornomass", "grhoc", "grhob", "grhov")
        },
        **{name: densities[name] for name in ("grhok", "grhor_nu", "n_mnu", "amnu")},
        "tau_start": TAU_START,
        "tau_end": tau_end,
        "save_index": 1.0,
        "w_DE_0": columns["w_DE_0"],
        "w_DE_a": columns["w_DE_a"],
        "cs2_DE": columns["cs2_DE"],
        "YHe_nominal": columns["YHe"],
        "Tcmb_nominal": columns["Tcmb"],
        "H0_nominal": columns["H0"],
        "Omegab_nominal": columns["Omegab"],
        "z_re": z_re,
    }
    n_cosmo = columns["Omegam"].shape
    return jnp.stack(
        [
            jnp.broadcast_to(
                jnp.asarray(named.get(_NAME_OF_COLUMN.get(i), 0.0), jnp.float64),
                n_cosmo,
            )
            for i in range(N_PARAM)
        ],
        axis=1,
    )


def _initial_inputs(columns, densities) -> dict:
    """Return what :meth:`Model.initial_state` reads of the cosmologies besides ``k``, ``tau``, ``lna_start``."""

    return {
        **{name: densities[name] for name in DIRECTED_NAMES[:5]},
        "w_DE_0": columns["w_DE_0"],
        "w_DE_a": columns["w_DE_a"],
        "Tcmb": columns["Tcmb"],
    }


def make_params(
    cosmology, k_values: np.ndarray, tau_end: float, z_re: float = 0.0
) -> np.ndarray:
    """Return packed parameters for the device right-hand side, one row per wave mode."""

    columns = _columns((cosmology,), held=False)
    row = _rows(columns, density_parameters(columns), jnp.asarray([tau_end]), z_re)
    params = np.tile(np.asarray(row), (len(k_values), 1))
    params[:, IX_K] = k_values
    return params


def make_initial_states(
    cosmology, k_values: np.ndarray, model, lna_start
) -> np.ndarray:
    """Return adiabatic initial states plus the fully ionized thermal state, ``(n_k, nvar)``.

    ``cosmology`` is one cosmology or a sequence of ``len(k_values)``, one per
    trajectory. ``lna_start`` is a number or one per trajectory. The initial
    conditions broadcast over the trajectories.
    """

    columns = _columns(_as_batch(cosmology), held=False)
    inputs = _initial_inputs(columns, density_parameters(columns))

    def per(values):
        values = np.asarray(values)
        return values[0] if len(values) == 1 else values

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


def _named_error_weights(weights, model: CosmologyModel, k_launch):
    """Turn ``{state name: weight}`` into solver error weights ``(n_traj, nvar)`` in launch order.

    Each weight is a number or a function of the trajectories' wave numbers
    (``k_launch``, launch order) that returns one per trajectory. Unnamed states
    keep weight one. The names are the model's
    (:meth:`~discoeb.models.CosmologyModel.state_names`), so a caller never needs the
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
    times, the shared ``t_span`` clipped to its ``tau0`` in ``IX_TAU_END``, so
    it stops there and its later saves hold its final state.
    """

    from modax import PIDGains, Rodas5P
    from modax import solve as modax_solve

    method = Rodas5P(lu_precision=lu_precision, gains=PIDGains(pcoeff=0.3, icoeff=0.4))
    # A tuple is the row split into leaves, the first holding IX_TAU_END.
    head = params[0] if isinstance(params, tuple) else params
    if isinstance(head, np.ndarray):
        t_rows = np.minimum(np.asarray(t_span)[None, :], head[:, IX_TAU_END, None])
    else:
        tau_end = jax.lax.stop_gradient(head[:, IX_TAU_END, None])
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
    model: CosmologyModel,
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


def build_keep_hook(model, names):
    """Return a modax save hook ``(y, tau, p) -> tuple`` of the quantities ``names``.

    A name is a state element or a quantity the joint model provides. State
    elements (``clxc``, ``theta_2``, ``psi_0_1``: the
    :meth:`~discoeb.models.CosmologyModel.state_names`, or ``lna``, ``thermo_0`` ...
    ``kappa`` of the recombination block) come straight from the state. Provided
    quantities (``R``, ``sigma``, ``phi`` with the line-of-sight sources
    appended, ``opacity`` or ``cs2`` from the thermal states) come from
    :meth:`~discoeb.component.Model.build_function`, with the state, the time and
    the parameter row as inputs. The hook clamps the time to the trajectory's
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
        f"    t = p[{IX_TAU_END}]",
        "    if tau < t:",
        "        t = tau",
    ]
    if provided:
        # build_function returns a lone output bare, so pair it with tau.
        outputs = provided if len(provided) > 1 else [*provided, "tau"]
        namespace["kept"] = model.build_function(
            outputs, inputs=("y", "tau"), params=PARAM_COLUMNS, name="kept"
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
def _keep_hook(model: CosmologyModel, extra, names):
    """Return the compiled hook for a resolved model, extra components and names (cached)."""

    return build_keep_hook(joint_model(model, extra), names)
