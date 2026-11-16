"""The linear matter power spectrum and its derivatives."""

import jax
import jax.numpy as jnp


def matter_power_spectrum(
    k,
    *,
    clxc,
    clxb,
    z,
    vb,
    tau0,
    Omegam,
    Omegab,
    A_s,
    n_s,
    k_p,
) -> jax.Array:
    """Return the linear ``P(k)`` today in Mpc^3 from the state today.

    ``P(k)`` is that of the cold dark matter and baryons,

    ``P(k) = 2 pi^2 / k^3 A_s (k / k_p)^(n_s - 1) delta_m(k, tau0)^2``,

    with ``delta_m`` their density-weighted contrast at unit initial
    curvature. Keep ``clxc``, ``clxb``, ``z`` and ``vb`` in
    :func:`~discoeb.solver.solve_eb` at a last time at or past every
    cosmology's conformal age, and pass their values there::

        tau0 = conformal_ages(cosmology)
        values = solve_eb(
            cosmology,
            k,
            tau0.max().reshape(1),
            keep=("clxc", "clxb", "z", "vb"),
            thermo_weight=0.1,
        )
        clxc, clxb, z, vb = jnp.unstack(values[0, :, :, -1])
        pk = matter_power_spectrum(
            k,
            clxc=clxc,
            clxb=clxb,
            z=z,
            vb=vb,
            tau0=tau0[0],
            Omegam=cosmology.Omegam,
            Omegab=cosmology.Omegab,
            A_s=cosmology.A_s,
            n_s=cosmology.n_s,
            k_p=cosmology.k_p,
        )

    Parameters
    ----------
    k
        Wave numbers in 1/Mpc, shape ``(n_k,)``.
    clxc, clxb
        The CDM and baryon density contrasts today, shape ``(..., n_k)``.
    z, vb
        The metric ``z`` and the baryon velocity today, shape ``(..., n_k)``:
        the contrasts' rates, for the end-time term of the derivative.
    tau0
        The conformal age in Mpc, shape ``(...)``, from
        :func:`~discoeb.solver.conformal_ages`. Only its derivative enters.
    Omegam, Omegab, A_s, n_s, k_p
        The matter and baryon density parameters, the primordial amplitude
        and tilt, and the pivot scale in 1/Mpc, each a number or shape
        ``(...)``.

    Returns
    -------
    jax.Array
        ``P(k)`` of shape ``(..., n_k)``.

    Notes
    -----
    Differentiable in :data:`DIFFERENTIABLE_PARAMETERS` given as JAX values,
    as ``solve_eb``'s values are. The derivative is the total one.
    ``solve_eb``'s is at fixed conformal age, so this adds the end-time term
    ``y'(tau0) d tau0``, zero in value. The rates are ``clxc' = -k z`` and
    ``clxb' = -k (z + vb)``. Pass ``tau0`` from ``conformal_ages``: a
    ``tau0`` under ``stop_gradient`` drops the term silently.
    The reionization redshift and the initial state stay fixed.
    """

    def column(x):
        return jnp.asarray(x, dtype=jnp.float64)[..., None]

    ks = jnp.asarray(k, dtype=jnp.float64)
    tau0 = column(tau0)

    # End-time term: the contrasts' rates at tau0 times d tau0 / dp.
    dtau = tau0 - jax.lax.stop_gradient(tau0)
    clxc = clxc - ks * z * dtau
    clxb = clxb - ks * (z + vb) * dtau

    # density-weighted, grho_c : grho_b = (Omegam - Omegab) : Omegab
    Omegam, Omegab = column(Omegam), column(Omegab)
    delta_m = ((Omegam - Omegab) * clxc + Omegab * clxb) / Omegam
    A_s, n_s, k_p = column(A_s), column(n_s), column(k_p)
    return 2.0 * jnp.pi**2 / ks**3 * A_s * (ks / k_p) ** (n_s - 1.0) * delta_m**2
