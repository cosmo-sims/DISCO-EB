"""The linear matter power spectrum and its derivatives."""

from typing import Annotated

import jax
import jax.numpy as jnp


def matter_power_spectrum(
    k,
    *,
    clxc,
    clxb,
    omega_c,
    omega_b,
    A_s,
    n_s,
    k_p,
) -> Annotated[jax.Array, ("k",)]:
    """Return the linear ``P(k)`` in Mpc^3 from the density contrasts.

    One cosmology's ``P(k)``, that of the cold dark matter and baryons,

    ``P(k) = 2 pi^2 / k^3 A_s (k / k_p)^(n_s - 1) delta_m(k)^2``,

    with ``delta_m`` their density-weighted contrast at unit initial
    curvature. Keep ``clxc`` and ``clxb`` in
    :func:`~discoeb.solver.solve_eb` at the scale factor of interest, 1
    for today, and pass their values there::

        values = solve_eb(cosmology, k, [1.0], keep=("clxc", "clxb"), thermo_weight=0.1)
        clxc, clxb = jnp.unstack(values[0, :, :, -1])
        pk = matter_power_spectrum(
            k,
            clxc=clxc,
            clxb=clxb,
            omega_c=cosmology.omega_c,
            omega_b=cosmology.omega_b,
            A_s=cosmology.A_s,
            n_s=cosmology.n_s,
            k_p=cosmology.k_p,
        )

    For a batch, ``jax.vmap`` it over the cosmologies. The keyword
    arguments are mapped over their leading axis and ``k`` is held::

        batched = jax.vmap(matter_power_spectrum, in_axes=(None,))
        pk = batched(k, clxc=clxc, ..., k_p=k_p)  # (n_cosmologies, n_k)

    Parameters
    ----------
    k
        Wave numbers in 1/Mpc, shape ``(n_k,)``.
    clxc, clxb
        The CDM and baryon density contrasts, shape ``(n_k,)``.
    omega_c, omega_b, A_s, n_s, k_p
        The cold dark matter and baryon densities, the primordial amplitude
        and tilt, and the pivot scale in 1/Mpc, each a scalar.

    Returns
    -------
    jax.Array
        ``P(k)`` of shape ``(n_k,)``.

    Notes
    -----
    Differentiable in the model's
    :func:`~discoeb.solver.differentiable_parameters` given as JAX values,
    as ``solve_eb``'s values are: the derivative at fixed scale factor.
    """

    ks, clxc, clxb, omega_c, omega_b, A_s, n_s, k_p = (
        jnp.asarray(x, dtype=jnp.float64)
        for x in (k, clxc, clxb, omega_c, omega_b, A_s, n_s, k_p)
    )
    # density-weighted, grho_c : grho_b = omega_c : omega_b
    delta_m = (omega_c * clxc + omega_b * clxb) / (omega_c + omega_b)
    return 2.0 * jnp.pi**2 / ks**3 * A_s * (ks / k_p) ** (n_s - 1.0) * delta_m**2
