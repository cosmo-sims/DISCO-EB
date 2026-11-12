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
from .cosmology import DIFFERENTIABLE_PARAMETERS
from .kernel import (
    DIRECTED_COLUMNS,
    IX_DIR,
    IX_EPS,
    IX_K,
    IX_TAU_END,
    PERTURB_ATOL,
    PERTURB_FIRST_STEP,
    PERTURB_MAX_STEPS,
    PERTURB_RTOL,
    _keep_hook,
    _launch,
    _named_error_weights,
    _pack_batch,
)
from .recombination import THERMO_ERROR_WEIGHT


def _input_vector(fixed: dict, theta_row, xp=jnp) -> jax.Array:
    """Return the inputs of :data:`DIRECTED_NAMES` from a cosmology's parameters, in JAX (or ``xp``)."""

    param = {**fixed, **dict(zip(DIFFERENTIABLE_PARAMETERS, theta_row))}
    d = density_parameters(param, xp=xp)
    return xp.stack(
        [
            d["grhog"],
            d["grhornomass"],
            d["grhoc"],
            d["grhob"],
            d["grhov"],
            param["YHe"],
            param["Tcmb"],
            param["H0"],
            param["Omegab"],
        ]
    )


def _kept_values(
    prepared,
    k_values,
    tau_grid,
    names,
    extra=(),
    *,
    drop_first: bool = True,
    error_weights=None,
    tau_end_of=None,
    rtol: float = PERTURB_RTOL,
    atol: float = PERTURB_ATOL,
    first_step: float = PERTURB_FIRST_STEP,
    max_steps: int = PERTURB_MAX_STEPS,
    thermo_weight: float = THERMO_ERROR_WEIGHT,
):
    """Return the hooked launches of :func:`solve_eb` as a function of the directed inputs.

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
    quantity's scale. The reionization redshift stays that of ``prepared``. So
    does ``tau0``, unless ``tau_end_of`` maps ``x`` to each cosmology's
    conformal age. The trajectories then stop at the age of the given inputs, in
    value only: the derivative is still the one at fixed ``tau0``.
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

    params, y0, t_span, order, k_sorted = _pack_batch(prepared, k_values, tau_grid)
    if isinstance(error_weights, dict):
        error_weights = _named_error_weights(error_weights, model, params[:, IX_K])
    n_k, n_grid, n_keep = len(k_sorted), len(tau_grid), len(names)
    first = 1 if drop_first else 0
    columns = jnp.asarray(DIRECTED_COLUMNS)
    n_dir = len(DIRECTED_COLUMNS)
    t_span_j = jnp.asarray(t_span)

    def with_inputs(base, x, replicas=1):
        # Row k_idx * n_cosmo + ci (times replicas) takes cosmology ci's row of x.
        def per_row(v):
            return jnp.repeat(jnp.tile(v, (n_k, 1)), replicas, axis=0)

        base = jnp.asarray(base).at[:, columns].set(per_row(x))
        if tau_end_of is not None:
            tau_end = jnp.asarray(tau_end_of(jax.lax.stop_gradient(x)))
            base = base.at[:, IX_TAU_END].set(per_row(tau_end[:, None])[:, 0])
        return base

    def arrange(rows):
        # Rows are k-major (row = k_idx * n_cosmo + cosmology_idx) and each
        # hook row is save-major, so this reshape is a view. The transpose to
        # cosmology-major copies only the kept values.
        values = rows.reshape(n_k, n_cosmo, n_grid, n_keep)
        values = jnp.transpose(values[:, :, first:, :], (1, 3, 0, 2))
        if np.any(order != np.arange(n_k)):
            values = values[:, :, jnp.asarray(np.argsort(order)), :]
        return values

    @jax.custom_jvp
    def of_inputs(x):
        _, out = _launch(
            prepared,
            with_inputs(params, x),
            jnp.asarray(y0),
            t_span_j,
            error_weights=error_weights,
            save_hook=_keep_hook(model, extra, names),
            save_history=False,
            **settings,
        )
        return arrange(out)

    @of_inputs.defjvp
    def of_inputs_jvp(primals, tangents):
        (x,), (dx,) = primals, tangents
        # Replica j of trajectory i is row i * n_dir + j, along the unit
        # direction e_j. Each carries the single column S = dy / d eps.
        params_rep = np.repeat(params, n_dir, axis=0)
        params_rep[:, IX_DIR : IX_DIR + n_dir] = np.tile(
            np.eye(n_dir), (n_k * n_cosmo, 1)
        )
        row = with_inputs(params_rep, x, n_dir)
        head, eps, tail = row[:, :IX_EPS], row[:, IX_EPS:IX_DIR], row[:, IX_DIR:]

        def kept_of(e):
            # eps is a leaf of its own, so differentiating it carries the one
            # column S = dy / d eps, out of step control.
            _, out = _launch(
                prepared,
                (head, e, tail),
                jnp.asarray(np.repeat(y0, n_dir, axis=0)),
                t_span_j,
                error_weights=None
                if error_weights is None
                else np.repeat(np.asarray(error_weights), n_dir, axis=0),
                save_hook=_keep_hook(model, extra, names),
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

    return of_inputs
