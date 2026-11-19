"""``solve_eb``, the interface to the Einstein-Boltzmann solve.

It runs one modax launch over every (wave mode, cosmology) pair and
records only the named quantities at the scale factors given. The
values are differentiable in every parameter of the cosmologies, in forward
mode.
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
        a JAX array, traced under ``jit`` or not, and differentiated.
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
        ``{name: values}`` of parameter-row columns
        (:func:`solve_inputs` names them) to set in place of those the
        cosmologies imply, each value a number or ``(n_cosmologies,)``. The
        solution becomes a function of them, so this is how to differentiate
        in a derived parameter such as ``omega_g`` on its own. The initial
        state stays the cosmologies'.
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
    values yet.

    The returned values are differentiable in every parameter of the
    cosmologies under ``jax.jvp`` and ``jax.jacfwd``, with or without
    ``jit``. Reverse mode (``jax.grad``, ``jax.vjp``, ``jax.jacrev``)
    raises: modax integrates the derivative along each tangent direction,
    and reverse mode has none to give it. For the gradient of a scalar,
    ``jax.jacfwd`` costs one column per parameter. The derivative reaches the
    kernel through the parameter row. modax's forward sensitivities carry
    it, with a hook that stores each kept quantity's derivative beside it,
    so it also costs one joint launch and no history. ``jax.jvp``
    integrates one sensitivity column and ``jax.jacfwd`` one per direction,
    however many columns of the row a direction moves. Over a batch, take
    ``jax.vmap(jax.jacfwd(f))`` of a function of one cosmology: ``jacfwd``
    of the batch would ask every cosmology for every other's directions. A
    thread holds at most :data:`~discoeb.solver._kernel.MAX_SENSITIVITY_STATE`
    components, and more columns run in threads of their own. It is the
    derivative at fixed scale factor, so at fixed redshift. The initial
    state carries its tangent too, as the start of the sensitivity column,
    at no cost. A key the solve never reads (``A_s``, ``n_s``, ``k_p``)
    has a zero derivative.
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
    # A derivative reaches the launch through the parameter rows and the
    # initial state, and modax integrates a sensitivity column along their
    # tangent. The save times are held fixed.
    layout = prepared.layout
    n_k = len(prepared.k_sorted)
    params = prepared.params
    if inputs is not None:
        unknown = sorted(set(inputs) - {name for name, _ in layout.values})
        if unknown:
            raise ValueError(
                f"inputs {unknown} are not columns of the parameter row: "
                f"{[name for name, _ in layout.values]}"
            )
        for name, value in inputs.items():
            value = jnp.broadcast_to(jnp.asarray(value, jnp.float64), (n_cosmo,))
            params = params.at[:, layout.columns[name]].set(jnp.tile(value, n_k))

    if isinstance(error_weights, dict):
        error_weights = _named_error_weights(
            error_weights, prepared.model, np.repeat(prepared.k_sorted, n_cosmo)
        )
    _, out = _launch(
        prepared,
        params,
        prepared.y0,
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


def solve_inputs(cosmologies) -> dict[str, jax.Array]:
    """Return the parameter-row columns the cosmologies imply, ``(n_cosmologies,)`` each, by name.

    ``cosmologies`` is a cosmology or a sequence of cosmologies of one
    model. The columns are what the kernel reads: the keys it takes as
    given (``omega_c``, ``omega_b``, ``Y_He``, ``T_cmb``, ...) and the
    parameters derived from keys (``omega_g``, ``omega_ur``, ``amnu``, ...).
    :func:`solve_eb` takes any of them as ``inputs``. JAX arrays, traceable
    under ``jit``.
    """

    model, dicts = _batch_of(cosmologies)
    values = density_parameters(_columns(dicts, model), model=model)
    return {
        name: jnp.asarray(values[name], dtype=jnp.float64)
        for name, _ in parameter_layout(model).values
    }


def state_names(model: type[Model]) -> tuple[str, ...]:
    """Return the perturbation state names of ``model``, for ``keep`` and ``error_weights``."""

    return _model_of(model).state_names()
