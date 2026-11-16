"""
Couplings between species are defined by :class:`~discoeb.components._component.Interaction`
classes. These contribute to the derivatives of species' states, but hold no state themselves.
"""

from __future__ import annotations

from ._component import Interaction, contributes


class ThomsonScattering(Interaction):
    """Thomson scattering between the photons and the baryons.

    Provides the momentum exchange (drag on ``vb`` and its reaction on
    ``qg``), the scattering source ``Pi = pig/10 + (9/15) E_2`` of the
    quadrupoles, and the damping ``-kappa' X_l`` of every higher temperature
    and polarization multipole. ``opacity`` is ``kappa' = a n_e sigma_T``.
    The buffers it fills start at zero, so it leaves the unscattered monopole
    alone.
    """

    def photbar(self, grhog_t, grhob_t):
        """The photon-to-baryon density ratio ``R^-1 = rho_gamma / rho_b``."""
        return grhog_t / grhob_t

    def slip(self, vb, qg):
        return (4.0 / 3.0) * vb - qg

    def polter(self, pig, E2):
        return pig * (1.0 / 10.0) + 9.0 * E2 * (1.0 / 15.0)

    @contributes("d_vb")
    def d_vb_thomson(self, photbar, opacity, slip):
        return -photbar * opacity * slip

    @contributes("d_theta")
    def d_theta_thomson(self, opacity, slip, polter, theta, rsa, output):
        damp = (1.0 - rsa) * opacity
        output[1] = damp * slip
        output[2] = -damp * (theta[2] - polter)
        for ell in range(3, len(theta)):
            output[ell] = -damp * theta[ell]

    @contributes("d_e")
    def d_e_thomson(self, opacity, polter, e, rsa, output):
        damp = (1.0 - rsa) * opacity
        output[0] = -damp * (e[0] - polter)
        for i in range(1, len(e)):
            output[i] = -damp * e[i]
