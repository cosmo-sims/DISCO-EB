"""``solve_eb``, the interface to the Einstein-Boltzmann solve.

It runs one modax launch over every (wave mode, cosmology) pair and
records only the named quantities at the scale factors given. The
values are differentiable in the model's :func:`differentiable_parameters`.
"""

from collections.abc import Sequence
from typing import Annotated

import jax
import jax.numpy as jnp
import numpy as np

from ..components.recombination import THERMO_ERROR_WEIGHT
from ..models._model import Model, _batch_of, _is_concrete, _model_of
from ._kernel import (
    A_START,
    N_COMMON_DIRECTED,
    PERTURB_ATOL,
    PERTURB_FIRST_STEP,
    PERTURB_MAX_STEPS,
    PERTURB_RTOL,
    _columns,
    _keep_hook,
    _launch,
    _named_error_weights,
    _prepare_solve,
    density_parameters,
    parameter_layout,
)


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
    split_derivatives: bool = False,
) -> Annotated[jax.Array, ("cosmology", "name", "k", "a")]:
    """Solve the Einstein-Boltzmann system of ``cosmologies`` and keep ``keep`` at the scale factors ``a``.

    One modax Rodas5P launch integrates every (wave mode, cosmology) pair
    and records only the quantities named in ``keep`` at the requested
    scale factors, through modax's save hook. The kernel evaluates them from the dense-output state at
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
        perturbation ``R``, the shear ``alpha``, the expansion rate ``aH`` or
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
        recombination block's ``Y_He`` and ``T_cmb``),
        replacing those the cosmologies imply. They are differentiable in
        their own right: the solution becomes a function of them. Given
        seven, the rest come from the cosmologies.
    rtol, atol, first_step, max_steps, error_weights, thermo_weight
        Solver settings.
        ``error_weights`` is either ``{state name: weight}`` or modax's array
        ``(n_k * n_cosmologies, nvar)`` in launch order (wave modes ascending,
        cosmologies within). In the dictionary, each weight is a number or a
        function of the trajectories' wave numbers that returns one per
        trajectory, and unnamed states keep weight one.
    split_derivatives
        How a derivative uses the device's threads. By default a trajectory
        and its sensitivity columns share one thread, which costs the least
        work and suits a batch that fills the device. With ``True`` each
        direction gets a copy of every trajectory in a thread of its own,
        holding the state and one column. The copies repeat the
        primal but run side by side, so one cosmology finishes about five
        times sooner, and a thread's memory no longer grows with the
        parameters.

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
    not checked against the model, since their ``w_DE_*`` and ``N_mnu`` have no
    values yet. The returned values are differentiable in the
    cosmologies' :func:`differentiable_parameters` of the model under
    ``jax.jvp``, ``jax.jacfwd`` and ``jax.grad``, with or without ``jit``.
    The derivative runs through the model's directed inputs: the background
    densities, the thermal inputs, the curvature, the massive neutrinos and
    the dark-energy parameters. modax's forward sensitivities carry it, with
    a hook that stores each kept quantity's derivative beside it, so it also
    costs one joint launch and no history. ``jax.jvp`` integrates one
    sensitivity column, ``jax.jacfwd`` one per direction, and over a batch,
    whose every cosmology is a direction, one per directed input that moves.
    ``jax.grad`` integrates one per directed input that moves, and so does
    every transform under ``jit``, where JAX stages the derivative before it
    knows whether it will transpose it. A thread holds
    at most :data:`~discoeb.solver._kernel.MAX_SENSITIVITY_STATE`
    components, and more columns run as several launches. It is the
    derivative at fixed scale factor, so at fixed redshift, and at fixed
    initial state.
    """

    model, dicts = _batch_of(cosmologies)
    n_cosmo = len(dicts)
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
    a_grid = jnp.concatenate((jnp.asarray([A_START]), a)) if drop_first else a

    columns = _columns(dicts, model)
    prepared = _prepare_solve(model, dicts, k_values, extra, columns)
    # The directed inputs the cosmologies imply, one column each, and those
    # ``inputs`` replaces. A derivative reaches the launch through them, and
    # modax integrates a sensitivity column along their tangent. The
    # primordial keys the solve never reads carry no tangent (_columns).
    x = _directed_inputs(columns, model, extra)
    if inputs is not None:
        given = jnp.asarray(inputs, dtype=jnp.float64)
        directed = parameter_layout(model, extra).directed
        if (
            given.ndim != 2
            or given.shape[0] != n_cosmo
            or given.shape[1] not in (N_COMMON_DIRECTED, len(directed))
        ):
            raise ValueError(
                f"inputs must be (n_cosmologies, {len(directed)}) or "
                f"(n_cosmologies, {N_COMMON_DIRECTED}), got {given.shape}"
            )
        x = [*jnp.unstack(given, axis=1), *x[given.shape[1] :]]

    if isinstance(error_weights, dict):
        error_weights = _named_error_weights(
            error_weights, prepared.model, np.repeat(prepared.k_sorted, n_cosmo)
        )
    n_k = len(prepared.k_sorted)
    # The directed parameters lead the row (parameter_layout). Each is a
    # params leaf of its own, so one that does not move reaches modax as a
    # symbolic zero and gets no column in reverse mode. The rest of the row
    # and the initial state are held fixed, as are the save times.
    rows = jax.lax.stop_gradient(prepared.params)
    params = (*(jnp.tile(column, n_k)[:, None] for column in x), rows[:, len(x) :])
    _, out = _launch(
        prepared,
        params,
        jax.lax.stop_gradient(prepared.y0),
        jax.lax.stop_gradient(a_grid),
        error_weights=error_weights,
        save_hook=_keep_hook(prepared.cosmology_model, tuple(extra), names),
        save_history=False,
        sens_error_control=False,
        split_derivatives=split_derivatives,
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
        thermo_weight=thermo_weight,
    )
    # Rows are k-major (row = k_idx * n_cosmo + cosmology_idx) and each hook
    # row save-major, so this reshape is a view. The transpose to
    # cosmology-major copies only the kept values.
    values = out.reshape(n_k, n_cosmo, a_grid.shape[0], len(names))
    values = jnp.transpose(values[:, :, int(drop_first) :, :], (1, 3, 0, 2))
    if np.any(prepared.order != np.arange(n_k)):
        values = values[:, :, jnp.asarray(np.argsort(prepared.order)), :]
    return values


def _directed_inputs(columns, model, extra=()) -> list[jax.Array]:
    """Return the directed inputs of ``model`` from a batch's columns, one ``(n_cosmologies,)`` array each.

    They are the layout's directed parameters, in its order
    (:func:`~discoeb.solver._kernel.parameter_layout`). Kept apart, not
    stacked, so that each carries its own tangent, or none.
    """

    values = density_parameters(columns, model=model)
    names = parameter_layout(model, extra).directed
    return [jnp.asarray(values[name], dtype=jnp.float64) for name in names]


def differentiable_parameters(model: type[Model]) -> tuple[str, ...]:
    """Return the cosmology keys :func:`solve_eb` differentiates for ``model``.

    Each component marks its own: ``Parameter(differentiable=True)``
    (:meth:`Model.differentiable_parameters
    <discoeb.models.Model.differentiable_parameters>`). For the built-in
    components these are ``omega_b``, ``omega_c``, ``omega_de``, ``T_cmb``,
    ``N_eff`` and ``omega_k``, ``m_nu`` with
    :class:`~discoeb.components.species.MassiveNeutrinos`, ``w_DE_0`` and
    ``w_DE_a`` with :class:`~discoeb.components.species.DarkEnergy` and
    ``z_re`` with :class:`~discoeb.components.recombination.ReionizationDepth`.
    They reach the solve through the directed inputs, and the forward
    sensitivities carry them through the hierarchy and the recombination
    history. The derivative is at fixed scale factor and holds the initial
    state fixed.

    Differentiating another key raises: ``N_mnu``, a count, and ``Y_He`` and
    ``cs2_DE``, whose derivatives of ``P(k)`` miss central differences by
    0.8 and 1 percent. The solve never reads ``A_s``, ``n_s`` and ``k_p``.
    """

    return _model_of(model).differentiable_parameters()


def directed_inputs(cosmologies) -> jax.Array:
    """Return the ``(n_cosmologies, n_directed)`` directed inputs the cosmologies imply.

    ``cosmologies`` is a cosmology or a sequence of cosmologies of one
    model. These are the model's directed
    parameters: the five background densities and the recombination block's
    ``Y_He`` and ``T_cmb``, then those of the model's other
    components (curvature, dark energy, massive neutrinos): the variables
    :func:`solve_eb` takes as
    ``inputs`` and differentiates with respect to. A JAX array, traceable
    under ``jit``.
    """

    model, dicts = _batch_of(cosmologies)
    columns = _columns(dicts, model, held=False)
    return jnp.stack(_directed_inputs(columns, model), axis=1)


def state_names(model: type[Model]) -> tuple[str, ...]:
    """Return the perturbation state names of ``model``, for ``keep`` and ``error_weights``."""

    return _model_of(model).state_names()
