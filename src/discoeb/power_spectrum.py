"""The linear matter power spectrum and its derivatives."""

import jax
import jax.numpy as jnp
import numpy as np

from .background import TAU_START, _background_tau, _densities, density_coefficients
from .cosmology import _as_cosmology
from .kernel import (
    DIRECTED_NAMES,
    PERTURB_ATOL,
    PERTURB_FIRST_STEP,
    PERTURB_MAX_STEPS,
    PERTURB_RTOL,
    _launch,
    _pack_batch,
    _prepare_solve,
)
from .models import CosmologyModel
from .sensitivities import _kept_values

MATTER_THERMO_WEIGHT = 0.1
"""Thermal error weight (:data:`discoeb.recombination.THERMO_ERROR_WEIGHT`) of the
matter power spectrum solves. A recombination history ten times looser than
the hierarchy's tolerance moves ``P(k)`` by at most 6e-4 against the weight
of 1 the CMB solves keep for the visibility. It takes 150 driver steps per
trajectory instead of 225 (0.47 s against 0.62 s for 128 cosmologies x 128
modes on an RTX 4070 SUPER)."""


def _primordial_power(ks, A_s, n_s, k_pivot):
    """``2 pi^2 / k^3`` times the primordial curvature spectrum: ``P(k) = this * delta_m^2``."""

    return 2.0 * np.pi**2 / ks**3 * A_s * (ks / k_pivot) ** (n_s - 1.0)


def _power_spectrum_from_final(y_final, k_sorted, cosmologies, k_values, index):
    """Return ``P(k)`` ``(n_cosmo, n_k)`` in the caller's order from final states ``(n_k, n_cosmo, nvar)``.

    ``index`` maps state names to their columns (the joint model's ``index``).
    """

    dens = np.array([density_coefficients(c) for c in cosmologies])
    grhoc, grhob = dens[:, 2][None, :], dens[:, 3][None, :]
    clxc, clxb = y_final[:, :, index["clxc"]], y_final[:, :, index["clxb"]]
    delta_m = (grhoc * clxc + grhob * clxb) / (grhoc + grhob)
    pk_sorted = (
        _primordial_power(
            k_sorted[:, None],
            *(
                np.array([getattr(c, a) for c in cosmologies])[None, :]
                for a in ("A_s", "n_s", "k_pivot")
            ),
        )
        * delta_m**2
    )
    order = np.argsort(np.asarray(k_values, dtype=np.float64), kind="stable")
    pk = np.empty_like(pk_sorted.T)
    pk[:, order] = pk_sorted.T
    return pk


def solve_matter_power_spectrum_batch(
    k_values,
    cosmologies,
    *,
    model: CosmologyModel,
    rtol: float = PERTURB_RTOL,
    atol: float = PERTURB_ATOL,
    first_step: float = PERTURB_FIRST_STEP,
    max_steps: int = PERTURB_MAX_STEPS,
    thermo_weight: float = MATTER_THERMO_WEIGHT,
) -> np.ndarray:
    """Return ``P(k)`` in Mpc^3, shape ``(n_cosmologies, n_k)``, for a batch in one launch.

    ``model`` is the cosmological model (:mod:`discoeb.models`) every
    cosmology of the batch belongs to.
    """

    prepared = _prepare_solve(model, tuple(cosmologies))
    params, y0, t_span, order, k_sorted = _pack_batch(prepared, k_values)
    sol = _launch(
        prepared,
        params,
        y0,
        t_span,
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
        thermo_weight=thermo_weight,
    )
    y_final = np.asarray(sol)[:, -1, :].reshape(
        len(k_sorted), len(prepared.cosmologies), prepared.model.nvar
    )
    return _power_spectrum_from_final(
        y_final, k_sorted, prepared.cosmologies, k_values, prepared.model.index
    )


def _tau0_of(densities, d):
    """Return the conformal age as a JAX function of the five densities, the rest of the background from ``d``."""

    return _background_tau({**d, **dict(zip(DIRECTED_NAMES[:5], densities))})[1][-1]


def matter_power_spectrum_jax(
    k_values,
    cosmology,
    densities=None,
    thermal=None,
    *,
    model: CosmologyModel,
    rtol: float = PERTURB_RTOL,
    atol: float = PERTURB_ATOL,
    first_step: float = PERTURB_FIRST_STEP,
    max_steps: int = PERTURB_MAX_STEPS,
    thermo_weight: float = MATTER_THERMO_WEIGHT,
):
    """Return ``P(k)`` as a JAX array, differentiable in the densities and thermal inputs.

    ``densities`` (the five of :data:`DIRECTED_NAMES`) and ``thermal`` (``YHe,
    Tcmb, H0, Omegab``) default to the cosmology's own. The derivative is the
    total one. The densities enter the hierarchy, the recombination rates
    (through the expansion rate), the matter weighting of ``delta_m`` and the
    conformal age at which the solve reads the state. The last path goes through
    ``y(tau_0(d)) = y(tau_0^*) + y'(tau_0^*) (tau_0(d) - tau_0^*)``, which is
    zero in value and carries ``d tau_0 / d(densities)`` from the background
    quadrature. The solve reads the value at the conformal age of the given
    densities, so a finite difference sees the same end-time dependence. Only
    the initial state stays fixed.

    The solve is the hooked launch of :func:`_kept_values`, keeping the
    contrasts and the rates the end-time term needs at ``tau_0``.
    """

    cosmology = _as_cosmology(cosmology)
    prepared = _prepare_solve(model, (cosmology,))
    d = _densities(cosmology)
    if densities is None:
        densities = [d[name] for name in DIRECTED_NAMES[:5]]
    if thermal is None:
        thermal = [cosmology.Y_He, cosmology.T_cmb, cosmology.H0, cosmology.Omegab]
    x = jnp.concatenate(
        [
            jnp.asarray(densities, dtype=jnp.float64),
            jnp.asarray(thermal, dtype=jnp.float64),
        ]
    )
    kept = _kept_values(
        prepared,
        k_values,
        # a save past tau0 holds the state today
        np.array([TAU_START, 2.0 * float(prepared.tau0[0])]),
        ("clxc", "clxb", "z", "vb"),
        tau_end_of=lambda xs: jnp.stack([_tau0_of(xi[:5], d) for xi in xs]),
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
        thermo_weight=thermo_weight,
    )
    clxc, clxb, z, vb = kept(x[None, :])[0, :, :, 0]

    # End-time term: the state's rate at tau0 (the equations the hook
    # evaluates on the final state) times d tau0 / d densities.
    ks = jnp.asarray(np.asarray(k_values, dtype=np.float64))
    tau0 = _tau0_of(x[:5], d)
    dtau = tau0 - jax.lax.stop_gradient(tau0)
    clxc = clxc - ks * z * dtau
    clxb = clxb - ks * (z + vb) * dtau

    grhoc, grhob = x[2], x[3]
    delta_m = (grhoc * clxc + grhob * clxb) / (grhoc + grhob)
    return (
        _primordial_power(ks, cosmology.A_s, cosmology.n_s, cosmology.k_pivot)
        * delta_m**2
    )
