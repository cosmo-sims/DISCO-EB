"""
Couplings between species are defined by :class:`~discoeb.components._component.Interaction`
classes. These contribute to the derivatives of species' states, but hold no state themselves.
"""

from __future__ import annotations

from ._component import Interaction, derivative


class ThomsonScattering(Interaction):
    """Thomson scattering between the photons and the baryons.

    Provides the momentum exchange (drag on ``theta_b`` and its reaction on
    the photon dipole), the scattering source ``Pi = sigma_g / 5 + (3/5)
    E_2`` of the quadrupoles, and the damping ``-kappa' X_l`` of every
    higher temperature and polarization multipole. ``opacity`` is ``kappa'
    = a n_e sigma_T``. The buffers it fills start at zero, so it leaves the
    unscattered monopole alone.
    """

    def photbar(self, rho_g, rho_b):
        """The photon-to-baryon density ratio ``rho_gamma / rho_b``."""
        return rho_g / rho_b

    def slip(self, theta_b, theta_g):
        return theta_b - theta_g

    def Pi(self, sigma_g, E2):
        return sigma_g * (1.0 / 5.0) + 9.0 * E2 * (1.0 / 15.0)

    @derivative("theta_b")
    def drag(self, photbar, opacity, slip):
        return -(4.0 / 3.0) * photbar * opacity * slip

    @derivative("F")
    def scattering_g(self, opacity, slip, Pi, F, k, rsa, output):
        damp = (1.0 - rsa) * opacity
        inv_k2 = 1.0 / (k * k)
        output[1] = damp * (4.0 / 3.0) * slip * k * inv_k2
        output[2] = -damp * (F[2] - Pi)
        for ell in range(3, len(F)):
            output[ell] = -damp * F[ell]

    @derivative("E")
    def scattering_E(self, opacity, Pi, E, rsa, output):
        damp = (1.0 - rsa) * opacity
        output[0] = -damp * (E[0] - Pi)
        for i in range(1, len(E)):
            output[i] = -damp * E[i]
