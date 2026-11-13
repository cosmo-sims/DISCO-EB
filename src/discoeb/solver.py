"""Solving the Einstein-Boltzmann system: keep named quantities at chosen times.

:func:`solve_eb` is the interface the CMB code uses. It runs one modax launch
over every (wave mode, cosmology) pair and records only the named quantities,
differentiable in :data:`~discoeb.cosmology.DIFFERENTIABLE_PARAMETERS`.
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .background import TAU_START, endpoints
from .cosmology import (
    DIFFERENTIABLE_PARAMETERS,
    _as_batch,
    _as_cosmology,
    _split_cosmologies,
)
from .kernel import (
    DIRECTED_NAMES,
    PERTURB_ATOL,
    PERTURB_FIRST_STEP,
    PERTURB_MAX_STEPS,
    PERTURB_RTOL,
    _prepare_solve,
    _solve_batch,
)
from .models import CosmologyBatch, CosmologyModel, _batch_of
from .recombination import THERMO_ERROR_WEIGHT
from .sensitivities import _input_vector, _kept_values


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
    :data:`DIFFERENTIABLE_PARAMETERS` given as JAX values.
    """

    names: tuple[str, ...]
    k: np.ndarray
    tau: np.ndarray
    tau0: np.ndarray
    values: jax.Array
    model: CosmologyModel

    def __getitem__(self, name: str) -> jax.Array:
        if name not in self.names:
            raise KeyError(f"{name!r} was not kept; the solution holds {self.names}")
        return self.values[:, self.names.index(name)]


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
        cosmology is a parameter dictionary or a cosmology object. The values
        of :data:`DIFFERENTIABLE_PARAMETERS` may be JAX arrays, traced or not.
        The others must be plain numbers.
    k_values
        Wave numbers in 1/Mpc, in any order.
    tau
        Strictly increasing conformal times in Mpc at which to record, starting
        at or after ``TAU_START``. Every launch starts at ``TAU_START``, so a
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
        The ``(n_cosmologies, 9)`` directed inputs of :data:`DIRECTED_NAMES`
        (the five background densities and the recombination block's ``YHe``,
        ``Tcmb``, ``H0``, ``Omegab``), replacing those the cosmologies imply
        (:func:`directed_inputs`). They are differentiable in their own right:
        the solution becomes a function of them, at fixed background endpoints
        of the cosmologies.
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
    right-hand side. The returned ``values`` are differentiable in the
    cosmologies' ``Omegam``, ``Omegab``, ``H0``, ``Tcmb`` and ``Neff`` under
    ``jax.jvp``, ``jax.jacfwd`` and ``jax.grad``. They are not under ``jit``,
    since the host finds the background endpoints from the concrete values. The
    derivative runs through the five background densities and the
    recombination block's ``Tcmb``, ``H0`` and ``Omegab``. The
    forward-sensitivity launch of :func:`_kept_values` carries it, with a hook
    that stores each kept quantity's derivative beside it, so it also costs one
    joint launch and no history. It is the derivative at fixed ``tau0`` and
    reionization redshift.
    """

    batch = _batch_of(cosmologies)
    model = batch.model
    fixed, theta = _split_cosmologies(batch.cosmologies)
    n_cosmo = len(fixed)

    tau = np.asarray(tau, dtype=np.float64).reshape(-1)
    if tau.size == 0 or not np.all(np.isfinite(tau)):
        raise ValueError("tau must be a non-empty array of finite conformal times")
    if tau[0] < TAU_START or np.any(np.diff(tau) <= 0.0):
        raise ValueError(
            f"tau must be strictly increasing and from TAU_START = {TAU_START} on"
        )
    # The launch starts at TAU_START, so the hook always records it. Drop
    # that row unless the caller asked for it.
    drop_first = bool(tau[0] > TAU_START)
    tau_grid = np.concatenate(([TAU_START], tau)) if drop_first else tau
    settings: dict[str, Any] = dict(
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
        thermo_weight=thermo_weight,
    )
    held = {}

    def prepare(theta):
        """Prepare the solve of the concrete parameters and return its kept-value function."""

        concrete = np.asarray(theta, dtype=np.float64)
        cosmos = tuple(
            _as_cosmology(
                {**fixed[i], **dict(zip(DIFFERENTIABLE_PARAMETERS, concrete[i]))}
            )
            for i in range(n_cosmo)
        )
        prepared = held["prepared"] = _prepare_solve(model, cosmos)
        names = tuple(model.state_names() if keep is None else keep)
        if len(set(names)) != len(names):
            raise ValueError(f"keep names a quantity twice: {names}")
        held["names"] = names
        return _kept_values(
            prepared,
            k_values,
            tau_grid,
            names,
            extra,
            drop_first=drop_first,
            error_weights=error_weights,
            **settings,
        )

    def inputs_of(theta):
        # JAX arithmetic throughout, also for parameters given as plain numbers
        theta = jnp.asarray(theta, dtype=jnp.float64)
        keys = set(fixed[0])
        if (
            isinstance(theta, jax.core.Tracer)
            or n_cosmo == 1
            or any(set(f) != keys for f in fixed)
        ):
            return jnp.stack(
                [_input_vector(fixed[i], theta[i]) for i in range(n_cosmo)]
            )
        # Concrete parameters: one set of eager elementwise JAX operations for
        # the whole batch matches each cosmology alone bit for bit. A last-bit
        # difference in a density can move a step acceptance.
        columns = {
            k: jnp.asarray([f[k] for f in fixed], dtype=jnp.float64) for k in keys
        }
        return _input_vector(columns, [theta[:, j] for j in range(theta.shape[1])]).T

    def solution(values):
        return EBSolution(
            names=held["names"],
            k=np.asarray(k_values, dtype=np.float64).reshape(-1),
            tau=tau,
            tau0=np.asarray(held["prepared"].tau0, dtype=np.float64),
            values=values,
            model=model,
        )

    if inputs is not None:
        # The given inputs are the solve's variables. The cosmologies set
        # everything else, the background endpoints included, as numbers.
        inputs = jnp.asarray(inputs, dtype=jnp.float64)
        if inputs.shape != (n_cosmo, len(DIRECTED_NAMES)):
            raise ValueError(
                f"inputs must be (n_cosmologies, {len(DIRECTED_NAMES)}), got {inputs.shape}"
            )
        return solution(prepare(np.asarray(jax.lax.stop_gradient(theta)))(inputs))

    # The background endpoints need the parameters as numbers, which a custom
    # rule's primals are under jax.grad and jax.jacfwd. The tangent chains the
    # inputs' dependence on the parameters (JAX) into the inputs-level rule of
    # _kept_values.
    @jax.custom_jvp
    def kept_of(theta):
        return prepare(theta)(inputs_of(theta))

    @kept_of.defjvp
    def kept_of_jvp(primals, tangents):
        (theta,), (dtheta,) = primals, tangents
        of_inputs = prepare(theta)
        return jax.jvp(lambda t: of_inputs(inputs_of(t)), (theta,), (dtheta,))

    return solution(kept_of(theta))


def directed_inputs(cosmologies) -> np.ndarray:
    """Return the ``(n_cosmologies, 9)`` directed inputs of :data:`DIRECTED_NAMES` the cosmologies imply.

    ``cosmologies`` is a :class:`~discoeb.models.CosmologyBatch`, one cosmology
    or a sequence of them: the inputs do not depend on the model. These are
    the five background densities and the recombination block's
    ``YHe``, ``Tcmb``, ``H0`` and ``Omegab``: the variables :func:`solve_eb`
    takes as ``inputs`` and differentiates with respect to.
    """

    fixed, theta = _split_cosmologies(_cosmologies_of(cosmologies))
    return np.stack(
        [
            np.asarray(_input_vector(fixed[i], jnp.asarray(theta[i])))
            for i in range(len(fixed))
        ]
    )


def conformal_ages(cosmologies) -> np.ndarray:
    """Return each cosmology's conformal age ``tau0`` (Mpc), where its trajectories stop.

    Reads the same cached background endpoints as :func:`solve_eb`, so a solve
    of the same cosmologies right after reuses them. The conformal age depends
    on the background alone, so ``cosmologies`` may be a
    :class:`~discoeb.models.CosmologyBatch`, one cosmology or a sequence of them.
    """

    batch = tuple(_as_cosmology(c) for c in _cosmologies_of(cosmologies))
    return np.asarray(endpoints(batch)[0], dtype=np.float64)


def state_names(model: CosmologyModel) -> tuple[str, ...]:
    """Return the perturbation state names of ``model``, for ``keep`` and ``error_weights``."""

    return model.state_names()


def _cosmologies_of(cosmologies) -> tuple:
    """Return the cosmologies of a batch, one cosmology or a sequence of them, as a tuple."""

    if isinstance(cosmologies, CosmologyBatch):
        return cosmologies.cosmologies
    return _as_batch(cosmologies)
