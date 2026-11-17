"""Derivatives of kept quantities by forward sensitivities.

A derivative moves the model's directed parameters, the leading columns of
the parameter row. ``jax.jvp`` of a modax launch integrates one sensitivity
column along the row's tangent, in the trajectory's own thread. A
sensitivity launch maps that over the differentiated variables and stores
each kept quantity's derivative beside its value.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.custom_derivatives import SymbolicZero

from ..components.recombination import REIO_ERROR_WEIGHT, THERMO_ERROR_WEIGHT
from ._kernel import (
    PERTURB_ATOL,
    PERTURB_FIRST_STEP,
    PERTURB_MAX_STEPS,
    PERTURB_RTOL,
    _keep_hook,
    _launch,
    _named_error_weights,
    density_parameters,
    parameter_layout,
)


def _directed_inputs(columns, model, extra=()) -> jax.Array:
    """Return the ``(n_cosmologies, n_directed)`` directed inputs of ``model`` from a batch's columns, in JAX.

    They are the layout's directed parameters (:func:`~discoeb.solver._kernel.parameter_layout`).
    """

    values = density_parameters(columns, model=model)
    names = parameter_layout(model, extra).directed
    return jnp.stack([values[name] for name in names], axis=1)


def _kept_values(
    prepared,
    tau_grid,
    names,
    extra=(),
    *,
    to_inputs=None,
    drop_first: bool = True,
    error_weights=None,
    rtol: float = PERTURB_RTOL,
    atol: float = PERTURB_ATOL,
    first_step: float = PERTURB_FIRST_STEP,
    max_steps: int = PERTURB_MAX_STEPS,
    thermo_weight: float = THERMO_ERROR_WEIGHT,
):
    """Return the hooked launches of :func:`solve_eb` as a function of the directed inputs.

    ``prepared`` is the batch :func:`~discoeb.solver._kernel._prepare_solve` packed at
    its wave modes, and ``tau_grid`` the save times, which may be traced.
    The result, ``of_variables(v)``, takes a pytree ``v`` whose leaves are
    ``(n_cosmologies,)`` or ``(n_cosmologies, m)``, maps it to the
    ``(n_cosmologies, n_directed)`` inputs ``x = to_inputs(v)`` of the
    layout's directed parameters (JAX, cosmology by cosmology; by default
    ``v`` is ``x``) and returns the kept quantities as ``(n_cosmologies,
    len(names), n_k, n_tau)`` in the caller's wave-mode order.

    It is a ``jax.custom_jvp`` in ``v``. The plain hooked launch gives the
    value. A forward-sensitivity launch gives the tangent: ``jax.jvp`` of
    the launch along ``dx / dv_r`` for each *differentiated* variable
    ``v_r``, a column of a leaf whose tangent is not a symbolic zero, mapped
    over ``r`` with ``jax.vmap``. modax integrates one sensitivity column
    along each direction beside the state, one thread per trajectory, and
    stores each kept quantity of :func:`build_keep_hook` with its
    derivative, so the kernel forms the derivative too and stores no
    history. A derivative in the five parameters of the densities therefore
    costs five columns however many inputs the model directs, and a
    variable nobody differentiates costs none. The rule contracts the
    columns with the variables' tangent, which is linear, so ``jax.grad``
    works off the same launch.

    The joint launch uses the fp64 LU and keeps the sensitivities out of step
    control. Its primal agrees with the plain fp32-LU one to ~1e-6 of a
    quantity's scale. ``tau0``, ``ln a`` at the start, the reionization
    redshift and the initial state stay those of ``prepared``.
    """

    model = prepared.cosmology_model
    n_cosmo = len(prepared.cosmologies)
    names = tuple(names)
    extra = tuple(extra)
    if to_inputs is None:
        to_inputs = _identity
    settings: dict[str, Any] = dict(
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
        thermo_weight=thermo_weight,
    )

    order, k_sorted = prepared.order, prepared.k_sorted
    if isinstance(error_weights, dict):
        error_weights = _named_error_weights(
            error_weights, prepared.model, np.repeat(k_sorted, n_cosmo)
        )
    n_k, n_grid, n_keep = len(k_sorted), len(tau_grid), len(names)
    first = 1 if drop_first else 0
    columns = np.asarray(
        prepared.layout.directed_columns
    )  # a constant: no tracer for the rule to close over
    hook = _keep_hook(model, extra, names)

    def arrange(rows):
        # Rows are k-major (row = k_idx * n_cosmo + cosmology_idx) and each
        # hook row is save-major, so this reshape is a view. The transpose to
        # cosmology-major copies only the kept values.
        values = rows.reshape(n_k, n_cosmo, n_grid, n_keep)
        values = jnp.transpose(values[:, :, first:, :], (1, 3, 0, 2))
        if np.any(order != np.arange(n_k)):
            values = values[:, :, jnp.asarray(np.argsort(order)), :]
        return values

    def launch(v, params, y0, t_span, **options):
        # Row k_idx * n_cosmo + ci takes cosmology ci's row of the inputs.
        x = jnp.tile(to_inputs(v), (n_k, 1))
        _, out = _launch(
            prepared,
            params.at[:, columns].set(x),
            y0,
            t_span,
            error_weights=error_weights,
            save_hook=hook,
            save_history=False,
            **settings,
            **options,
        )
        return arrange(out)

    # The rows, the initial state and the save times are arguments, not
    # closed over: under jit they are tracers, and a custom_jvp may not close
    # over those. The derivative holds them fixed, so the rule drops their
    # tangents.
    @jax.custom_jvp
    def kept(v, params, y0, t_span):
        return launch(v, params, y0, t_span)

    def kept_jvp(primals, tangents):
        v, params, y0, t_span = primals
        leaves, treedef = jax.tree_util.tree_flatten(v)
        d_leaves = treedef.flatten_up_to(tangents[0])
        live = [i for i, d in enumerate(d_leaves) if not isinstance(d, SymbolicZero)]
        if not live:  # every tangent zero: no sensitivity launch
            values = kept(v, params, y0, t_span)
            return values, jnp.zeros_like(values)
        moving = [jnp.asarray(leaves[i], dtype=jnp.float64) for i in live]
        widths = [leaf.reshape(n_cosmo, -1).shape[1] for leaf in moving]
        n_rep = sum(widths)

        def joint(*moved):
            full = list(leaves)
            for i, leaf in zip(live, moved):
                full[i] = leaf
            return launch(
                treedef.unflatten(full),
                params,
                y0,
                t_span,
                lu_precision="fp64",
                # Thread-local memory, not registers, bounds the joint system's
                # occupancy. An uncapped register budget halves the resident
                # threads the driver reserves local memory for.
                max_registers=None,
                sens_error_control=False,
                reio_weight=REIO_ERROR_WEIGHT,
            )

        # One unit direction per differentiated column of a leaf. modax
        # integrates a sensitivity column along each beside the state, in
        # the trajectory's own thread.
        units, start = [], 0
        for leaf, m in zip(moving, widths):
            unit = np.zeros((n_rep, n_cosmo, m))
            unit[np.arange(start, start + m), :, np.arange(m)] = 1.0
            units.append(jnp.asarray(unit.reshape((n_rep,) + leaf.shape)))
            start += m
        values, jacobian = jax.vmap(
            lambda *direction: jax.jvp(joint, tuple(moving), direction),
            out_axes=(None, 0),
        )(*units)
        # jacobian[r, c] = d q_c / d v_r. The contraction with the variables'
        # tangent is linear, so jax.grad transposes it.
        dv = jnp.concatenate(
            [jnp.reshape(d_leaves[i], (n_cosmo, -1)) for i in live], axis=1
        )
        return values, jnp.einsum("rc...,cr->c...", jacobian, dv)

    kept.defjvp(kept_jvp, symbolic_zeros=True)

    params = jax.lax.stop_gradient(prepared.params)
    y0 = jax.lax.stop_gradient(prepared.y0)
    t_span = jax.lax.stop_gradient(jnp.asarray(tau_grid, dtype=jnp.float64))

    def of_variables(v):
        return kept(v, params, y0, t_span)

    return of_variables


def _identity(x):
    return jnp.asarray(x, dtype=jnp.float64)
