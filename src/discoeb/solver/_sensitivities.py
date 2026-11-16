"""Derivatives of kept quantities: forward sensitivities along the directed inputs.

The model reads each input a derivative may move (its directed parameters,
led by :data:`~discoeb.solver._kernel.DIRECTED_NAMES`) as
``x_nominal + eps dir_x``. A sensitivity launch replicates every
trajectory once per differentiated variable, with the direction ``dx / dv``
of that variable, and differentiates the launch in ``eps``, a leaf of the
parameter row of its own, with ``jax.jvp``: modax then integrates one
sensitivity column per replica and keeps each quantity's derivative beside it.
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
    value. A forward-sensitivity launch gives the tangent. It replicates every
    trajectory once per *differentiated* variable, a column of a leaf whose
    tangent is not a symbolic zero, and replica ``r`` carries the single
    column ``S = dy / d eps`` along the direction ``dx / dv_r`` (``jax.jvp``
    of ``to_inputs``) through ``eps`` and the ``dir`` block. That column is
    ``jax.jvp`` of the launch in ``eps``, a leaf of the parameter row of its
    own, so modax integrates it alone. modax stores each kept quantity of
    :func:`build_keep_hook` and its derivative along the column, so the
    kernel forms the derivative too and stores no history. A derivative in
    the five parameters of the densities therefore costs five replicas
    however many inputs the model directs, and a variable nobody
    differentiates costs none.

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
    layout = prepared.layout
    ix_eps, ix_dir = layout.ix_eps, layout.ix_dir
    columns = np.asarray(
        layout.directed_columns
    )  # a constant: no tracer for the rule to close over
    n_dir = len(columns)
    hook = _keep_hook(model, extra, names)

    def with_inputs(base, x, replicas=1):
        # Row k_idx * n_cosmo + ci (times replicas) takes cosmology ci's row of x.
        def per_row(v):
            return jnp.repeat(jnp.tile(v, (n_k, 1)), replicas, axis=0)

        return base.at[:, columns].set(per_row(x))

    def arrange(rows):
        # Rows are k-major (row = k_idx * n_cosmo + cosmology_idx) and each
        # hook row is save-major, so this reshape is a view. The transpose to
        # cosmology-major copies only the kept values.
        values = rows.reshape(n_k, n_cosmo, n_grid, n_keep)
        values = jnp.transpose(values[:, :, first:, :], (1, 3, 0, 2))
        if np.any(order != np.arange(n_k)):
            values = values[:, :, jnp.asarray(np.argsort(order)), :]
        return values

    # The rows, the initial state and the save times are arguments, not
    # closed over: under jit they are tracers, and a custom_jvp may not close
    # over those. The derivative holds them fixed, so the rule drops their
    # tangents.
    @jax.custom_jvp
    def kept(v, params, y0, t_span):
        _, out = _launch(
            prepared,
            with_inputs(params, to_inputs(v)),
            y0,
            t_span,
            error_weights=error_weights,
            save_hook=hook,
            save_history=False,
            **settings,
        )
        return arrange(out)

    def kept_jvp(primals, tangents):
        v, params, y0, t_span = primals
        leaves, treedef = jax.tree_util.tree_flatten(v)
        d_leaves = treedef.flatten_up_to(tangents[0])
        x = to_inputs(v)

        # One replica per differentiated column of a leaf, along dx / dv.
        directions, coefficients = [], []
        for i, (leaf, d_leaf) in enumerate(zip(leaves, d_leaves)):
            if isinstance(d_leaf, SymbolicZero):
                continue
            leaf = jnp.asarray(leaf, dtype=jnp.float64)
            m = leaf.reshape(n_cosmo, -1).shape[1]

            def x_of(value, i=i):
                return to_inputs(
                    treedef.unflatten([*leaves[:i], value, *leaves[i + 1 :]])
                )

            for j in range(m):
                unit = jnp.zeros((n_cosmo, m)).at[:, j].set(1.0).reshape(leaf.shape)
                directions.append(jax.jvp(x_of, (leaf,), (unit,))[1])
            coefficients.append(jnp.reshape(d_leaf, (n_cosmo, m)))
        if not directions:  # every tangent zero: no sensitivity launch
            values = kept(v, params, y0, t_span)
            return values, jnp.zeros_like(values)
        V = jnp.stack(directions, axis=1)  # (n_cosmo, n_rep, n_dir)
        dv = jnp.concatenate(coefficients, axis=1)  # (n_cosmo, n_rep)
        n_rep = V.shape[1]

        # Replica r of trajectory i is row i * n_rep + r, along V[:, r].
        row = with_inputs(jnp.repeat(params, n_rep, axis=0), x, n_rep)
        row = row.at[:, ix_dir : ix_dir + n_dir].set(
            jnp.tile(V, (n_k, 1, 1)).reshape(-1, n_dir)
        )
        head, eps, tail = row[:, :ix_eps], row[:, ix_eps:ix_dir], row[:, ix_dir:]

        def kept_of(e):
            # eps is a leaf of its own, so differentiating it carries the one
            # column S = dy / d eps, out of step control.
            _, out = _launch(
                prepared,
                (head, e, tail),
                jnp.repeat(y0, n_rep, axis=0),
                t_span,
                error_weights=None
                if error_weights is None
                else np.repeat(np.asarray(error_weights), n_rep, axis=0),
                save_hook=hook,
                save_history=False,
                lu_precision="fp64",
                # Thread-local memory, not registers, bounds the joint system's
                # occupancy. An uncapped register budget halves the resident
                # threads the driver reserves local memory for.
                max_registers=None,
                sens_error_control=False,
                reio_weight=REIO_ERROR_WEIGHT,
                **settings,
            )
            return out

        out, d_out = jax.jvp(kept_of, (eps,), (jnp.ones_like(eps),))
        out = out.reshape(n_k * n_cosmo, n_rep, n_grid, n_keep)
        d_out = d_out.reshape(n_k * n_cosmo, n_rep, n_grid, n_keep)
        # Every replica integrates the same state, so the first's primal serves.
        values = arrange(out[:, 0])
        # d_out[i, r] = d q_i / d v_r at every save. The contraction with the
        # variables' tangent is linear, so jax.grad transposes it.
        dv_rows = jnp.tile(dv, (n_k, 1))
        tangent = arrange(jnp.einsum("irsq,ir->isq", d_out, dv_rows))
        return values, tangent

    kept.defjvp(kept_jvp, symbolic_zeros=True)

    params = jax.lax.stop_gradient(prepared.params)
    y0 = jax.lax.stop_gradient(prepared.y0)
    t_span = jax.lax.stop_gradient(jnp.asarray(tau_grid, dtype=jnp.float64))

    def of_variables(v):
        return kept(v, params, y0, t_span)

    return of_variables


def _identity(x):
    return jnp.asarray(x, dtype=jnp.float64)
