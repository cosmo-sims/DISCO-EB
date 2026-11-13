"""Derivatives of kept quantities: forward sensitivities along the directed inputs.

The model reads each input a derivative may move (:data:`~discoeb.kernel.DIRECTED_NAMES`)
as ``x_nominal + eps dir_x``. A sensitivity launch replicates every
trajectory once per input and differentiates the launch in ``eps``, a leaf of
the parameter row of its own, with ``jax.jvp``: modax then integrates one
sensitivity column per replica and keeps each quantity's derivative beside it.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .background import density_parameters
from .kernel import (
    DIRECTED_COLUMNS,
    IX_DIR,
    IX_EPS,
    PERTURB_ATOL,
    PERTURB_FIRST_STEP,
    PERTURB_MAX_STEPS,
    PERTURB_RTOL,
    _keep_hook,
    _launch,
    _named_error_weights,
)
from .recombination import THERMO_ERROR_WEIGHT


def _directed_inputs(columns) -> jax.Array:
    """Return the ``(n_cosmologies, 9)`` inputs of :data:`DIRECTED_NAMES` from a batch's columns, in JAX."""

    d = density_parameters(columns)
    return jnp.stack(
        [
            d["grhog"],
            d["grhornomass"],
            d["grhoc"],
            d["grhob"],
            d["grhov"],
            columns["YHe"],
            columns["Tcmb"],
            columns["H0"],
            columns["Omegab"],
        ],
        axis=1,
    )


def _kept_values(
    prepared,
    tau_grid,
    names,
    extra=(),
    *,
    drop_first: bool = True,
    error_weights=None,
    rtol: float = PERTURB_RTOL,
    atol: float = PERTURB_ATOL,
    first_step: float = PERTURB_FIRST_STEP,
    max_steps: int = PERTURB_MAX_STEPS,
    thermo_weight: float = THERMO_ERROR_WEIGHT,
):
    """Return the hooked launches of :func:`solve_eb` as a function of the directed inputs.

    ``prepared`` is the batch :func:`~discoeb.kernel._prepare_solve` packed at
    its wave modes, and ``tau_grid`` the save times, which may be traced.
    The result, ``of_inputs(x)``, takes ``x``, the ``(n_cosmologies, 9)``
    inputs of :data:`DIRECTED_NAMES`, and returns the kept quantities as
    ``(n_cosmologies, len(names), n_k, n_tau)`` in the caller's wave-mode order.
    It is a ``jax.custom_jvp`` in ``x``. The plain hooked launch gives the
    value. A forward-sensitivity launch gives the tangent: it replicates every
    trajectory once per input, and each replica carries the single column
    ``S = dy / d eps`` of its unit direction through ``IX_EPS`` and ``IX_DIR``.
    That column is ``jax.jvp`` of the launch in ``eps``, a leaf of the
    parameter row of its own, so modax integrates it alone. modax stores each
    kept quantity of :func:`build_keep_hook` and its derivative along the
    column, so the kernel forms the derivative too and stores no history.

    The joint launch uses the fp64 LU and keeps the sensitivities out of step
    control. Its primal agrees with the plain fp32-LU one to ~1e-6 of a
    quantity's scale. The reionization redshift and ``tau0`` stay those of
    ``prepared``.
    """

    model = prepared.cosmology_model
    n_cosmo = len(prepared.cosmologies)
    names = tuple(names)
    extra = tuple(extra)
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
            error_weights, model, np.repeat(k_sorted, n_cosmo)
        )
    n_k, n_grid, n_keep = len(k_sorted), len(tau_grid), len(names)
    first = 1 if drop_first else 0
    columns = np.asarray(
        DIRECTED_COLUMNS
    )  # a constant: no tracer for the rule to close over
    n_dir = len(DIRECTED_COLUMNS)
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
    def kept(x, params, y0, t_span):
        _, out = _launch(
            prepared,
            with_inputs(params, x),
            y0,
            t_span,
            error_weights=error_weights,
            save_hook=hook,
            save_history=False,
            **settings,
        )
        return arrange(out)

    @kept.defjvp
    def kept_jvp(primals, tangents):
        x, params, y0, t_span = primals
        dx = tangents[0]
        # Replica j of trajectory i is row i * n_dir + j, along the unit
        # direction e_j. Each carries the single column S = dy / d eps.
        params_rep = jnp.repeat(params, n_dir, axis=0)
        params_rep = params_rep.at[:, IX_DIR : IX_DIR + n_dir].set(
            jnp.tile(jnp.eye(n_dir), (n_k * n_cosmo, 1))
        )
        row = with_inputs(params_rep, x, n_dir)
        head, eps, tail = row[:, :IX_EPS], row[:, IX_EPS:IX_DIR], row[:, IX_DIR:]

        def kept_of(e):
            # eps is a leaf of its own, so differentiating it carries the one
            # column S = dy / d eps, out of step control.
            _, out = _launch(
                prepared,
                (head, e, tail),
                jnp.repeat(y0, n_dir, axis=0),
                t_span,
                error_weights=None
                if error_weights is None
                else np.repeat(np.asarray(error_weights), n_dir, axis=0),
                save_hook=hook,
                save_history=False,
                lu_precision="fp64",
                # Thread-local memory, not registers, bounds the joint system's
                # occupancy. An uncapped register budget halves the resident
                # threads the driver reserves local memory for.
                max_registers=None,
                sens_error_control=False,
                **settings,
            )
            return out

        out, d_out = jax.jvp(kept_of, (eps,), (jnp.ones_like(eps),))
        out = out.reshape(n_k * n_cosmo, n_dir, n_grid, n_keep)
        d_out = d_out.reshape(n_k * n_cosmo, n_dir, n_grid, n_keep)
        # Every replica integrates the same state, so the first's primal serves.
        values = arrange(out[:, 0])
        # d_out[i, j] = d q_i / d x_j at every save. The contraction with the
        # input tangent is linear, so jax.grad transposes it.
        dx_rows = jnp.tile(dx, (n_k, 1))
        tangent = arrange(jnp.einsum("ijsq,ij->isq", d_out, dx_rows))
        return values, tangent

    params = jax.lax.stop_gradient(prepared.params)
    y0 = jax.lax.stop_gradient(prepared.y0)
    t_span = jax.lax.stop_gradient(jnp.asarray(tau_grid, dtype=jnp.float64))

    def of_inputs(x):
        return kept(jnp.asarray(x, dtype=jnp.float64), params, y0, t_span)

    return of_inputs
