"""Interactions between species: components that write other components' derivatives.

An :class:`~discoeb.component.Interaction` owns no state. It reads the
species' quantities and adds to their derivatives, so a model couples two
species by listing the interaction beside them.
"""

from __future__ import annotations

from .component import Interaction


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

    def d_vb(self, photbar, opacity, slip):
        return -photbar * opacity * slip

    def d_theta(self, opacity, slip, polter, theta, rsa, output):
        damp = (1.0 - rsa) * opacity
        output[1] = damp * slip
        output[2] = -damp * (theta[2] - polter)
        for ell in range(3, len(theta)):
            output[ell] = -damp * theta[ell]

    def d_e(self, opacity, polter, e, rsa, output):
        damp = (1.0 - rsa) * opacity
        output[0] = -damp * (e[0] - polter)
        for i in range(1, len(e)):
            output[i] = -damp * e[i]
