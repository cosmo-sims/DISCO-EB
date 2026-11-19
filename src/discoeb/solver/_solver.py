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

    One modax Rodas5P launch integrates every (wave mode, cosmology) pair.
    At each save the kernel evaluates the quantities named in ``keep`` from
    the dense-output state and stores them, so the device holds ``n_k *
    n_cosmologies * len(a) * len(keep)`` values and never the history.

    Parameters
    ----------
    cosmologies
        A cosmology, ``LCDM(omega_b=..., ...)``, or a sequence of cosmologies of
        one model. The model sets the species and interactions the solve
        evolves, and each model compiles one kernel. A parameter may be a
        JAX array, traced under ``jit`` or not, and differentiated.
    k_values
        Wave numbers in 1/Mpc, in any order. They must be concrete numbers,
        since they set the launch's shape.
    a
        Strictly increasing scale factors at which to record, from
        :data:`A_START` on. The solve integrates in ``a``, and 1 is today
        for every cosmology. Traced values must start after ``A_START``.
        Every launch starts there, so a first value equal to it records the
        initial state. Keep ``tau`` for the conformal time at each ``a``.
        Its value at 1 is the conformal age.
    keep
        Names of the quantities to record. A name is a state element
        (:meth:`~discoeb.models.Model.state_names`, or the recombination
        block's ``tau``, ``thermo_0``, ``thermo_1``, ``thermo_2``, ``kappa``,
        ``kappa_reio``) or a quantity the model provides, such as the
        comoving curvature perturbation ``R``, the shear ``alpha``, the
        expansion rate ``aH`` or the ``opacity``. Default: every element of
        the perturbation state.
    extra
        Components to append to the model for the hook: stateless
        :class:`~discoeb.components.Component` subclasses whose providers
        read the model's quantities and add their own (DISCO-CMB's
        line-of-sight sources). The hook cache tells them apart by class.
    inputs
        ``{name: values}`` of parameter-row columns
        (:func:`solve_inputs` names them) to set in place of those the
        cosmologies imply, each value a number or ``(n_cosmologies,)``. The
        solution becomes a function of them, which lets you differentiate
        in a derived parameter such as ``omega_g``. The initial state stays
        the cosmologies'.
    rtol, atol, first_step, max_steps
        Settings of the Rodas5P step controller.
    error_weights
        Either ``{state name: weight}`` or modax's array ``(n_k *
        n_cosmologies, nvar)`` in launch order (wave modes ascending,
        cosmologies within). In the dictionary each weight is a number or a
        function of the trajectories' wave numbers that returns one per
        trajectory, and unnamed states keep weight one.
    thermo_weight
        Weight of the recombination block in the error norm.
    split_derivatives
        How a derivative uses the device's threads. By default a trajectory
        and its sensitivity columns share one thread, which costs the least
        work and suits a batch that fills the device. With ``True`` each
        direction gets a copy of every trajectory in a thread of its own,
        holding the state and one column. The copies repeat the primal but
        run side by side, so one cosmology finishes sooner, and a thread's
        memory does not grow with the number of parameters.

    Returns
    -------
    jax.Array
        The kept quantities, ``(n_cosmologies, len(keep), n_k, n_a)``:
        cosmologies in the order given, quantities in ``keep`` order, wave
        modes and scale factors in the caller's order.
        ``jnp.unstack(values, axis=1)`` splits it into one
        ``(n_cosmologies, n_k, n_a)`` array per name.

    Notes
    -----
    The hook compiles once per model, set of names and number of save
    times. Before the launch ``solve_eb`` computes only the parameter rows
    and the initial states, in JAX, so it traces under ``jit``. Under
    ``jit`` the model cannot check the cosmologies, since their values are
    not known.

    The returned values are differentiable in every parameter of the
    cosmologies under ``jax.jvp`` and ``jax.jacfwd``, with or without
    ``jit``. modax integrates one sensitivity column per tangent direction
    in the same launch and stores each kept quantity's derivative beside
    it. Reverse mode (``jax.grad``, ``jax.vjp``, ``jax.jacrev``) raises.
    For the gradient of a scalar use ``jax.jacfwd``. Over a batch, take
    ``jax.vmap(jax.jacfwd(f))`` of a function of one cosmology, since
    ``jacfwd`` of the batch would ask every cosmology for every other's
    directions. The derivative holds the scale factor fixed, so it is the
    derivative at fixed redshift. A parameter the solve never reads
    (``A_s``, ``n_s``, ``k_p``) has a zero derivative.
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
    model. The kernel reads these columns: the keys it takes as
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
