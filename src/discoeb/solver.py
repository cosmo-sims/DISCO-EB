"""Solving the Einstein-Boltzmann system: keep named quantities at chosen times.

:func:`solve_eb` is the interface the CMB code uses. It runs one modax launch
over every (wave mode, cosmology) pair and records only the named quantities,
differentiable in :data:`~discoeb.cosmology.DIFFERENTIABLE_PARAMETERS`.
"""

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .background import TAU_START, background_endpoints
from .cosmology import (
    DIFFERENTIABLE_PARAMETERS,
    UNUSED_BY_THE_SOLVE,
    _as_batch,
    _columns,
    _is_concrete,
)
from .kernel import (
    DIRECTED_NAMES,
    N_COMMON_DIRECTED,
    PERTURB_ATOL,
    PERTURB_FIRST_STEP,
    PERTURB_MAX_STEPS,
    PERTURB_RTOL,
    _prepare_solve,
    _solve_batch,
    parameter_layout,
)
from .models import W0WACDM_MNU, CosmologyBatch, CosmologyModel, _batch_of
from .recombination import THERMO_ERROR_WEIGHT
from .sensitivities import _directed_inputs, _kept_values


def solve_perturbation_history_batch(
    k_values,
    cosmologies: CosmologyBatch,
    *,
    tau_save=None,
    rtol: float = PERTURB_RTOL,
    atol: float = PERTURB_ATOL,
    first_step: float = PERTURB_FIRST_STEP,
    max_steps: int = PERTURB_MAX_STEPS,
):
    """Solve a batch of cosmologies on a shared ``tau_save`` grid in one launch.

    ``cosmologies`` is a model applied to parameter dictionaries,
    ``LCDM([params, ...])`` (:class:`~discoeb.models.CosmologyBatch`).
    One modax launch integrates all ``n_cosmologies * n_k`` trajectories, each
    with its recombination block stepping ahead of its hierarchy
    (:func:`joint_kernel`). Each trajectory stops at its own ``tau0``, and later
    save times hold its final state. Returns ``(history, k_sorted, prepared)``,
    with ``history`` a device array ``(n_k, n_cosmologies, len(tau_save),
    nvar)`` in ascending ``k``.
    """

    batch = _batch_of(cosmologies)
    sol, prepared, k_sorted = _solve_batch(
        batch.model,
        batch.cosmologies,
        k_values,
        tau_save,
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
    )
    hist = jnp.asarray(sol).reshape(
        len(k_sorted), len(prepared.cosmologies), -1, prepared.model.nvar
    )
    return hist, k_sorted, prepared


@dataclass(frozen=True)
class EBSolution:
    """What :func:`solve_eb` kept: the selected quantities on the requested grid.

    ``values`` is a device array ``(n_cosmologies, len(names), n_k, n_tau)``
    with wave modes and times in the caller's order. ``solution[name]`` is that
    quantity's ``(n_cosmologies, n_k, n_tau)`` block. ``tau0`` is each
    cosmology's conformal age, past which its column holds the state today.
    ``values`` is differentiable in the parameters of
    :data:`DIFFERENTIABLE_PARAMETERS` given as JAX values. A solution is a JAX
    pytree, so a jitted function may return one: ``values``, ``tau`` and
    ``tau0`` are its leaves.
    """

    names: tuple[str, ...]
    k: np.ndarray
    tau: jax.Array
    tau0: jax.Array
    values: jax.Array
    model: CosmologyModel

    def __getitem__(self, name: str) -> jax.Array:
        if name not in self.names:
            raise KeyError(f"{name!r} was not kept; the solution holds {self.names}")
        return self.values[:, self.names.index(name)]


jax.tree_util.register_pytree_node(
    EBSolution,
    lambda s: ((s.values, s.tau, s.tau0), (s.names, tuple(s.k.tolist()), s.model)),
    lambda aux, leaves: EBSolution(
        names=aux[0],
        k=np.asarray(aux[1], dtype=np.float64),
        tau=leaves[1],
        tau0=leaves[2],
        values=leaves[0],
        model=aux[2],
    ),
)


def solve_eb(
    cosmologies: CosmologyBatch,
    k_values,
    tau,
    *,
    keep=None,
    extra=(),
    inputs=None,
    rtol: float = PERTURB_RTOL,
    atol: float = PERTURB_ATOL,
    first_step: float = PERTURB_FIRST_STEP,
    max_steps: int = PERTURB_MAX_STEPS,
    error_weights=None,
    thermo_weight: float = THERMO_ERROR_WEIGHT,
) -> EBSolution:
    """Solve the Einstein-Boltzmann system of ``cosmologies`` and keep ``keep`` at ``tau``.

    Like :func:`solve_perturbation_history_batch`, this runs one modax Rodas5P
    launch over every (wave mode, cosmology) pair. Unlike it, the launch records
    only the quantities named in ``keep`` at the requested times, through
    modax's save hook. The kernel evaluates them from the dense-output state at
    each save and stores ``len(keep)`` numbers, so the device holds ``n_k *
    n_cosmologies * len(tau) * len(keep)`` values and never the history. Keeping
    only the comoving curvature perturbation at 1000 times for 128 modes and 128
    cosmologies stores 130 MB instead of the 7 GB of the 55-variable history.

    Parameters
    ----------
    cosmologies
        A cosmological model applied to the cosmologies to solve,
        ``LCDM([params, ...])`` (:class:`~discoeb.models.CosmologyBatch`). The
        model says which species and interactions the solve evolves. Each
        cosmology is a parameter dictionary or a cosmology object. Every value
        may be a JAX array, traced under ``jit`` or not. Only those of
        :data:`DIFFERENTIABLE_PARAMETERS` may be differentiated.
    k_values
        Wave numbers in 1/Mpc, in any order. They must be numbers, not
        traced: they set the launch's shape.
    tau
        Strictly increasing conformal times in Mpc at which to record, starting
        at or after ``TAU_START``. Traced times must start after it. Every launch starts at ``TAU_START``, so a
        first time equal to it records the initial state. A cosmology's
        trajectory ends at its conformal age ``tau0``, and later times hold its
        state today. A last time past every ``tau0`` therefore returns the
        present-day values of the whole batch.
    keep
        Names of the quantities to record: state elements
        (:meth:`~discoeb.models.CosmologyModel.state_names`, or the recombination
        block's ``lna``, ``thermo_0``, ``thermo_1``, ``thermo_2``, ``kappa``) or
        quantities the joint model provides, such as the comoving curvature
        perturbation ``R``, the shear ``sigma``, the expansion rate ``adotoa`` or
        the opacity. Default: every element of the perturbation state, which
        amounts to the hierarchy's full history.
    extra
        Components to append to the model for the hook: stateless
        :class:`~discoeb.component.Component` subclasses whose providers read the
        model's quantities and add their own (the line-of-sight sources of
        :mod:`discocmb.cmb`). The hook cache tells them apart by class.
    inputs
        The ``(n_cosmologies, n_directed)`` directed inputs of the model
        (:func:`directed_inputs`, in the order of its
        :func:`~discoeb.kernel.parameter_layout`), or only the first nine
        every built-in model shares (the five background densities and the
        recombination block's ``YHe``, ``Tcmb``, ``H0``, ``Omegab``),
        replacing those the cosmologies imply. They are differentiable in
        their own right: the solution becomes a function of them, at fixed
        background endpoints of the cosmologies. Given nine, the rest come
        from the cosmologies.
    rtol, atol, first_step, max_steps, error_weights, thermo_weight
        Solver settings, as in :func:`solve_perturbation_history_batch`.
        ``error_weights`` is either ``{state name: weight}`` or modax's array
        ``(n_k * n_cosmologies, nvar)`` in launch order (wave modes ascending,
        cosmologies within). In the dictionary, each weight is a number or a
        function of the trajectories' wave numbers that returns one per
        trajectory, and unnamed states keep weight one.

    Notes
    -----
    The hook compiles once per batch shape and set of names, like the
    right-hand side. Everything before the launch is JAX: the background
    endpoints, the parameter rows and the initial states
    (:func:`~discoeb.kernel._pack`), so ``solve_eb`` traces under ``jit``, and
    the whole solve then runs on the device. Under ``jit`` the cosmologies are
    not checked against the model, since their ``w_DE_*`` and ``Nmnu`` have no
    values yet. The returned ``values`` are differentiable in the
    cosmologies' :func:`differentiable_parameters` of the model under
    ``jax.jvp``, ``jax.jacfwd`` and ``jax.grad``, with or without ``jit``.
    The derivative runs through the model's directed inputs: the background
    densities, the thermal inputs, the curvature, the massive neutrinos and
    the dark-energy parameters. The
    forward-sensitivity launch of :func:`_kept_values` carries it, with a hook
    that stores each kept quantity's derivative beside it, so it also costs
    one joint launch and no history, with one replica of every trajectory per
    differentiated parameter. It is the derivative at fixed ``tau0``,
    reionization redshift, ``ln a`` at the start and initial state.
    """

    batch = _batch_of(cosmologies)
    model = batch.model
    n_cosmo = len(batch)
    names = tuple(model.state_names() if keep is None else keep)
    if len(set(names)) != len(names):
        raise ValueError(f"keep names a quantity twice: {names}")

    tau = jnp.asarray(tau, dtype=jnp.float64).reshape(-1)
    if tau.size == 0:
        raise ValueError("tau must be a non-empty array of conformal times")
    if _is_concrete(tau):
        checked = np.asarray(tau)
        if not np.all(np.isfinite(checked)):
            raise ValueError("tau must be an array of finite conformal times")
        if checked[0] < TAU_START or np.any(np.diff(checked) <= 0.0):
            raise ValueError(
                f"tau must be strictly increasing and from TAU_START = {TAU_START} on"
            )
        # The launch starts at TAU_START, so the hook always records it. Drop
        # that row unless the caller asked for it.
        drop_first = bool(checked[0] > TAU_START)
    else:
        drop_first = True  # traced times must lie after TAU_START
    tau_grid = jnp.concatenate((jnp.asarray([TAU_START]), tau)) if drop_first else tau

    columns = _columns(
        batch.cosmologies,
        keys=model.parameter_keys(),
        differentiable=differentiable_parameters(model),
    )
    prepared = _prepare_solve(model, batch.cosmologies, k_values, extra, columns)
    # The variables of the derivative. JAX arithmetic maps them to the
    # directed inputs (_directed_inputs), and the rule of _kept_values
    # launches one sensitivity replica per variable actually differentiated.
    variables = {
        # The primordial keys the solve never reads carry no tangent, and so
        # no replica, when P(k) differentiates them.
        "columns": {
            k: jax.lax.stop_gradient(v) if k in UNUSED_BY_THE_SOLVE else v
            for k, v in columns.items()
        },
    }
    to_inputs = functools.partial(_inputs_of, model, tuple(extra), 0)
    if inputs is not None:
        # The given inputs are the solve's variables. The cosmologies set
        # everything else, the background endpoints included, and the
        # directed inputs after the first nine when only those are given.
        x = jnp.asarray(inputs, dtype=jnp.float64)
        directed = parameter_layout(model, extra).directed
        if (
            x.ndim == 2
            and x.shape[1] == len(DIRECTED_NAMES) != len(directed)
            and set(directed) <= set(DIRECTED_NAMES)
        ):
            # the superset directed_inputs gives a dictionary: take the model's
            x = x[:, [DIRECTED_NAMES.index(name) for name in directed]]
        if (
            x.ndim != 2
            or x.shape[0] != n_cosmo
            or x.shape[1]
            not in (
                N_COMMON_DIRECTED,
                len(directed),
            )
        ):
            raise ValueError(
                f"inputs must be (n_cosmologies, {len(directed)}) or "
                f"(n_cosmologies, {N_COMMON_DIRECTED}), got {x.shape}"
            )
        variables["inputs"] = x
        to_inputs = functools.partial(_inputs_of, model, tuple(extra), x.shape[1])
    of_variables = _kept_values(
        prepared,
        tau_grid,
        names,
        extra,
        to_inputs=to_inputs,
        drop_first=drop_first,
        error_weights=error_weights,
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
        thermo_weight=thermo_weight,
    )
    return EBSolution(
        names=names,
        k=np.asarray(k_values, dtype=np.float64).reshape(-1),
        tau=tau,
        tau0=prepared.tau0,
        values=of_variables(variables),
        model=model,
    )


def _inputs_of(model, extra, given, variables) -> jax.Array:
    """The directed inputs of the variables of :func:`solve_eb`.

    The cosmologies' columns imply them, and ``variables["inputs"]`` replaces
    the first ``given`` of them.
    """

    implied = _directed_inputs(variables["columns"], model, extra)
    if not given:
        return implied
    return jnp.concatenate([variables["inputs"], implied[:, given:]], axis=1)


def differentiable_parameters(model: CosmologyModel) -> tuple[str, ...]:
    """Return the keys of :data:`DIFFERENTIABLE_PARAMETERS` a solve of ``model`` differentiates.

    Those the model's components declare (:meth:`CosmologyModel.parameter_keys
    <discoeb.models.CosmologyModel.parameter_keys>`): ``w_DE_0`` and
    ``w_DE_a`` need :class:`~discoeb.species.DarkEnergy` and ``mnu``
    :class:`~discoeb.species.MassiveNeutrinos`. Differentiating another key
    raises.
    """

    keys = model.parameter_keys()
    return tuple(key for key in DIFFERENTIABLE_PARAMETERS if key in keys)


def directed_inputs(cosmologies) -> jax.Array:
    """Return the ``(n_cosmologies, n_directed)`` directed inputs the cosmologies imply.

    ``cosmologies`` is a :class:`~discoeb.models.CosmologyBatch`, or one
    cosmology or a sequence of them for :data:`DIRECTED_NAMES`, the superset
    of the built-in models' inputs. These are the model's directed
    parameters: the five background densities and the recombination block's
    ``YHe``, ``Tcmb``, ``H0`` and ``Omegab``, then those of the model's other
    components (curvature, dark energy, massive neutrinos): the variables
    :func:`solve_eb` takes as
    ``inputs`` and differentiates with respect to. A JAX array, traceable
    under ``jit``.
    """

    model = _model_of(cosmologies)
    columns = _columns(
        _cosmologies_of(cosmologies), held=False, keys=model.parameter_keys()
    )
    return _directed_inputs(columns, model)


def conformal_ages(cosmologies) -> jax.Array:
    """Return each cosmology's conformal age ``tau0`` (Mpc), where its trajectories stop.

    The conformal age depends on the background alone, so ``cosmologies`` may
    be a :class:`~discoeb.models.CosmologyBatch`, or one cosmology or a
    sequence of them for a built-in model's background. A JAX array, traceable under ``jit``: the same vmapped quadrature
    :func:`solve_eb` runs.
    """

    model = _model_of(cosmologies)
    columns = _columns(
        _cosmologies_of(cosmologies), held=False, keys=model.parameter_keys()
    )
    return background_endpoints(columns, model)[0]


def state_names(model: CosmologyModel) -> tuple[str, ...]:
    """Return the perturbation state names of ``model``, for ``keep`` and ``error_weights``."""

    return model.state_names()


def _model_of(cosmologies) -> CosmologyModel:
    """Return a batch's model, or :data:`~discoeb.models.W0WACDM_MNU` for bare cosmologies."""

    if isinstance(cosmologies, CosmologyBatch):
        return cosmologies.model
    return W0WACDM_MNU


def _cosmologies_of(cosmologies) -> tuple:
    """Return the cosmologies of a batch, one cosmology or a sequence of them, as a tuple."""

    if isinstance(cosmologies, CosmologyBatch):
        return cosmologies.cosmologies
    return _as_batch(cosmologies)
