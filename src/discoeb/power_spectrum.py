"""The linear matter power spectrum and its derivatives."""

import jax
import jax.numpy as jnp

from .background import background_endpoints
from .cosmology import _columns, _is_concrete
from .models import CosmologyBatch, _batch_of
from .solver import EBSolution

MATTER_THERMO_WEIGHT = 0.1
"""Thermal error weight (:data:`discoeb.recombination.THERMO_ERROR_WEIGHT`) of the
matter power spectrum solves. A recombination history ten times looser than
the hierarchy's tolerance moves ``P(k)`` by at most 6e-4 against the weight
of 1 the CMB solves keep for the visibility. It takes 150 driver steps per
trajectory instead of 225 (0.47 s against 0.62 s for 128 cosmologies x 128
modes on an RTX 4070 SUPER)."""

MATTER_POWER_QUANTITIES = ("clxc", "clxb", "z", "vb")
"""What :func:`compute_matter_power_spectrum` reads of a solution: the CDM and
baryon contrasts, and the metric ``z`` and baryon velocity of their rates."""


@jax.custom_jvp
def _age_shift(params):
    """Return ``tau0(params) - tau0(params)``: zero in value, ``d tau0`` in its tangent.

    The value needs no quadrature. Only the tangent integrates the background.
    """

    return jnp.zeros_like(params["H0"])


@_age_shift.defjvp
def _age_shift_jvp(primals, tangents):
    _, d_tau0 = jax.jvp(lambda p: background_endpoints(p)[0], primals, tangents)
    return _age_shift(*primals), d_tau0


def compute_matter_power_spectrum(
    solution: EBSolution, cosmologies: CosmologyBatch
) -> jax.Array:
    """Return the linear ``P(k)`` today in Mpc^3, shape ``(n_cosmologies, n_k)``, from a solve.

    ``solution`` is what :func:`~discoeb.solver.solve_eb` returns for
    ``cosmologies``, the same :class:`~discoeb.models.CosmologyBatch`, with
    :data:`MATTER_POWER_QUANTITIES` kept and a last time at or past every
    cosmology's conformal age, so that it holds the state today::

        batch = LCDM(cosmologies)
        solution = solve_eb(
            batch,
            k,
            [conformal_ages(batch).max()],
            keep=MATTER_POWER_QUANTITIES,
            thermo_weight=MATTER_THERMO_WEIGHT,
        )
        pk = compute_matter_power_spectrum(solution, batch)

    ``P(k)`` is that of the cold dark matter and baryons,

    ``P(k) = 2 pi^2 / k^3 A_s (k / k_p)^(n_s - 1) delta_m(k, tau0)^2``,

    with ``delta_m`` their density-weighted contrast at unit initial
    curvature, at the solution's wave numbers in their order.

    Differentiable in :data:`DIFFERENTIABLE_PARAMETERS` given as JAX values in
    the dictionaries, as the solution is. The derivative is the total one.
    ``solve_eb``'s is at fixed conformal age, so this adds the end-time term:
    ``y(tau0(p)) = y(tau0^*) + y'(tau0^*) (tau0(p) - tau0^*)``, zero in value,
    which carries ``d tau0 / dp`` from the background quadrature. The rates
    are ``clxc' = -k z`` and ``clxb' = -k (z + vb)``, hence ``z`` and ``vb``
    among the kept quantities. The reionization redshift and the initial
    state stay fixed.
    """

    batch = _batch_of(cosmologies)
    missing = [name for name in MATTER_POWER_QUANTITIES if name not in solution.names]
    if missing:
        raise ValueError(
            f"the solution lacks {missing}: solve_eb must keep {MATTER_POWER_QUANTITIES}"
        )
    if len(solution.tau0) != len(batch):
        raise ValueError(
            f"the solution has {len(solution.tau0)} cosmologies, the batch {len(batch)}"
        )
    if _is_concrete(solution.tau) and _is_concrete(solution.tau0):
        if solution.tau[-1] < solution.tau0.max():
            raise ValueError(
                f"the solution's last time {float(solution.tau[-1]):.6g} Mpc "
                f"comes before the conformal age {float(solution.tau0.max()):.6g} "
                "Mpc: P(k) needs today"
            )
    # each (n_cosmologies, n_k), wave modes in the solution's order
    clxc, clxb, z, vb = (solution[name][..., -1] for name in MATTER_POWER_QUANTITIES)
    params = _columns(batch.cosmologies, held=False)

    # End-time term: the contrasts' rates at tau0 times d tau0 / dp.
    ks = jnp.asarray(solution.k, dtype=jnp.float64)
    dtau = _age_shift(params)[:, None]
    clxc = clxc - ks * z * dtau
    clxb = clxb - ks * (z + vb) * dtau

    # density-weighted, grho_c : grho_b = (Omegam - Omegab) : Omegab
    Omegam, Omegab = params["Omegam"][:, None], params["Omegab"][:, None]
    delta_m = ((Omegam - Omegab) * clxc + Omegab * clxb) / Omegam
    A_s, n_s, k_p = (params[key][:, None] for key in ("A_s", "n_s", "k_p"))
    return 2.0 * jnp.pi**2 / ks**3 * A_s * (ks / k_p) ** (n_s - 1.0) * delta_m**2
