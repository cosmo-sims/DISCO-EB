"""Solving the Einstein-Boltzmann system: keep named quantities at chosen times.

:func:`solve_eb` is the interface the CMB code uses. It runs one modax launch
over every (wave mode, cosmology) pair and records only the named quantities,
differentiable in :data:`~discoeb.models._cosmology.DIFFERENTIABLE_PARAMETERS`.
"""

import functools
from collections.abc import Sequence
from typing import Annotated

import jax
import jax.numpy as jnp
import numpy as np

from ..components.recombination import THERMO_ERROR_WEIGHT, ReionizationDepth
from ..models._cosmology import (
    DIFFERENTIABLE_PARAMETERS,
    UNUSED_BY_THE_SOLVE,
    _columns,
    _is_concrete,
)
from ..models._model import Model, ModelSpec, _batch_of, _spec_of
from ._kernel import (
    A_START,
    N_COMMON_DIRECTED,
    PERTURB_ATOL,
    PERTURB_FIRST_STEP,
    PERTURB_MAX_STEPS,
    PERTURB_RTOL,
    _prepare_solve,
    _solve_batch,
    parameter_layout,
)
from ._sensitivities import _directed_inputs, _kept_values


def solve_perturbation_history_batch(
    k_values,
    cosmologies,
    *,
    a_save=None,
    rtol: float = PERTURB_RTOL,
    atol: float = PERTURB_ATOL,
    first_step: float = PERTURB_FIRST_STEP,
    max_steps: int = PERTURB_MAX_STEPS,
):
    """Solve a batch of cosmologies on a shared ``a_save`` grid in one launch.

    ``cosmologies`` is a cosmology, ``LCDM(omega_b=..., ...)``, or a sequence of
    cosmologies of one model. One modax launch integrates all ``n_cosmologies * n_k`` trajectories, each
    with its recombination block stepping ahead of its hierarchy
    (:func:`joint_kernel`), in the scale factor from :data:`A_START` to 1, today.
    Returns ``(history, k_sorted, prepared)``,
    with ``history`` a device array ``(n_k, n_cosmologies, len(a_save),
    nvar)`` in ascending ``k``.
    """

    batch = _batch_of(cosmologies)
    sol, prepared, k_sorted = _solve_batch(
        batch.model,
        batch.cosmologies,
        k_values,
        a_save,
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
    )
    hist = jnp.asarray(sol).reshape(
        len(k_sorted), len(prepared.cosmologies), -1, prepared.model.nvar
    )
    return hist, k_sorted, prepared


def solve_eb(
    cosmologies: Model | Sequence[Model],
    k_values,
    a,
    *,
    keep: Sequence[str] | None = None,
    extra=(),
    inputs=None,
    rtol: float = PERTURB_RTOL,
    atol: float = PERTURB_ATOL,
    first_step: float = PERTURB_FIRST_STEP,
    max_steps: int = PERTURB_MAX_STEPS,
    error_weights=None,
    thermo_weight: float = THERMO_ERROR_WEIGHT,
) -> Annotated[jax.Array, ("cosmology", "name", "k", "a")]:
    """Solve the Einstein-Boltzmann system of ``cosmologies`` and keep ``keep`` at the scale factors ``a``.

    Like :func:`solve_perturbation_history_batch`, this runs one modax Rodas5P
    launch over every (wave mode, cosmology) pair. Unlike it, the launch records
    only the quantities named in ``keep`` at the requested times, through
    modax's save hook. The kernel evaluates them from the dense-output state at
    each save and stores ``len(keep)`` numbers, so the device holds ``n_k *
    n_cosmologies * len(a) * len(keep)`` values and never the history. Keeping
    only the comoving curvature perturbation at 1000 times for 128 modes and 128
    cosmologies stores 130 MB instead of the 7 GB of the 55-variable history.

    Parameters
    ----------
    cosmologies
        A cosmology, ``LCDM(omega_b=..., ...)``, or a sequence of cosmologies of
        one model. The model says which species and interactions the solve
        evolves, and one kernel is compiled per model. Every parameter may be
        a JAX array, traced under ``jit`` or not. Only those of
        :func:`differentiable_parameters` may be differentiated.
    k_values
        Wave numbers in 1/Mpc, in any order. They must be numbers, not
        traced: they set the launch's shape.
    a
        Strictly increasing scale factors at which to record, from
        :data:`A_START` on: the solve integrates in ``a``, and 1 is today
        for every cosmology. Traced values must start after ``A_START``.
        Every launch starts there, so a first value equal to it records the
        initial state. Keep ``tau`` for the conformal time at each, whose
        value at 1 is the conformal age.
    keep
        Names of the quantities to record: state elements
        (:meth:`~discoeb.models.Model.state_names`, or the recombination
        block's ``tau``, ``thermo_0``, ``thermo_1``, ``thermo_2``, ``kappa``,
        ``kappa_reio``) or
        quantities the joint model provides, such as the comoving curvature
        perturbation ``R``, the shear ``sigma``, the expansion rate ``adotoa`` or
        the opacity. Default: every element of the perturbation state, which
        amounts to the hierarchy's full history.
    extra
        Components to append to the model for the hook: stateless
        :class:`~discoeb.components._component.Component` subclasses whose providers read the
        model's quantities and add their own (the line-of-sight sources of
        :mod:`discocmb.cmb`). The hook cache tells them apart by class.
    inputs
        The ``(n_cosmologies, n_directed)`` directed inputs of the model
        (:func:`directed_inputs`, in the order of its
        :func:`~discoeb.solver._kernel.parameter_layout`), or only the first seven
        every built-in model shares (the five background densities and the
        recombination block's ``YHe`` and ``Tcmb``),
        replacing those the cosmologies imply. They are differentiable in
        their own right: the solution becomes a function of them. Given
        seven, the rest come from the cosmologies.
    rtol, atol, first_step, max_steps, error_weights, thermo_weight
        Solver settings, as in :func:`solve_perturbation_history_batch`.
        ``error_weights`` is either ``{state name: weight}`` or modax's array
        ``(n_k * n_cosmologies, nvar)`` in launch order (wave modes ascending,
        cosmologies within). In the dictionary, each weight is a number or a
        function of the trajectories' wave numbers that returns one per
        trajectory, and unnamed states keep weight one.

    Returns
    -------
    jax.Array
        The kept quantities, ``(n_cosmologies, len(keep), n_k, n_a)``:
        cosmologies in the order given, quantities in ``keep`` order (the
        model's :func:`state_names` without ``keep``), wave modes and times
        in the caller's order. ``jnp.unstack(values, axis=1)`` splits it
        into one ``(n_cosmologies, n_k, n_a)`` array per name.

    Notes
    -----
    The hook compiles once per batch shape and set of names, like the
    right-hand side. Everything before the launch is JAX and algebra: the
    parameter rows and the initial states
    (:func:`~discoeb.solver._kernel._pack`), so ``solve_eb`` traces under ``jit``, and
    the whole solve then runs on the device. Under ``jit`` the cosmologies are
    not checked against the model, since their ``w_DE_*`` and ``Nmnu`` have no
    values yet. The returned values are differentiable in the
    cosmologies' :func:`differentiable_parameters` of the model under
    ``jax.jvp``, ``jax.jacfwd`` and ``jax.grad``, with or without ``jit``.
    The derivative runs through the model's directed inputs: the background
    densities, the thermal inputs, the curvature, the massive neutrinos and
    the dark-energy parameters. The
    forward-sensitivity launch of :func:`_kept_values` carries it, with a hook
    that stores each kept quantity's derivative beside it, so it also costs
    one joint launch and no history, with one replica of every trajectory per
    differentiated parameter. It is the derivative at fixed scale factor,
    so at fixed redshift, and at fixed initial state.
    """

    batch = _batch_of(cosmologies)
    model = batch.model
    n_cosmo = len(batch)
    names = tuple(model.state_names() if keep is None else keep)
    if len(set(names)) != len(names):
        raise ValueError(f"keep names a quantity twice: {names}")

    a = jnp.asarray(a, dtype=jnp.float64).reshape(-1)
    if a.size == 0:
        raise ValueError("a must be a non-empty array of scale factors")
    if _is_concrete(a):
        checked = np.asarray(a)
        if not np.all(np.isfinite(checked)):
            raise ValueError("a must be an array of finite scale factors")
        if checked[0] < A_START or np.any(np.diff(checked) <= 0.0):
            raise ValueError(
                f"a must be strictly increasing and from A_START = {A_START} on"
            )
        # The launch starts at A_START, so the hook always records it. Drop
        # that row unless the caller asked for it.
        drop_first = bool(checked[0] > A_START)
    else:
        drop_first = True  # traced values must lie after A_START
    tau_grid = jnp.concatenate((jnp.asarray([A_START]), a)) if drop_first else a

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
    return of_variables(variables)


def _inputs_of(model, extra, given, variables) -> jax.Array:
    """The directed inputs of the variables of :func:`solve_eb`.

    The cosmologies' columns imply them, and ``variables["inputs"]`` replaces
    the first ``given`` of them.
    """

    implied = _directed_inputs(variables["columns"], model, extra)
    if not given:
        return implied
    return jnp.concatenate([variables["inputs"], implied[:, given:]], axis=1)


def differentiable_parameters(model: type[Model] | ModelSpec) -> tuple[str, ...]:
    """Return the keys of :data:`DIFFERENTIABLE_PARAMETERS` a solve of ``model`` differentiates.

    Those the model's components declare (:meth:`Model.parameter_keys
    <discoeb.models.Model.parameter_keys>`): ``w_DE_0`` and
    ``w_DE_a`` need :class:`~discoeb.components.species.DarkEnergy` and ``mnu``
    :class:`~discoeb.components.species.MassiveNeutrinos`. ``z_re`` needs
    :class:`~discoeb.components.recombination.ReionizationDepth`, whose
    state makes the steps resolve the reionization step's derivative.
    Differentiating another key raises.
    """

    spec = _spec_of(model)
    keys = dict(spec.parameter_keys())
    if not spec.has(ReionizationDepth):
        keys.pop("z_re", None)
    return tuple(key for key in DIFFERENTIABLE_PARAMETERS if key in keys)


def directed_inputs(cosmologies) -> jax.Array:
    """Return the ``(n_cosmologies, n_directed)`` directed inputs the cosmologies imply.

    ``cosmologies`` is a cosmology or a sequence of cosmologies of one
    model. These are the model's directed
    parameters: the five background densities and the recombination block's
    ``YHe`` and ``Tcmb``, then those of the model's other
    components (curvature, dark energy, massive neutrinos): the variables
    :func:`solve_eb` takes as
    ``inputs`` and differentiates with respect to. A JAX array, traceable
    under ``jit``.
    """

    batch = _batch_of(cosmologies)
    columns = _columns(batch.cosmologies, keys=batch.model.parameter_keys(), held=False)
    return _directed_inputs(columns, batch.model)


def state_names(model: type[Model] | ModelSpec) -> tuple[str, ...]:
    """Return the perturbation state names of ``model``, for ``keep`` and ``error_weights``."""

    return _spec_of(model).state_names()
