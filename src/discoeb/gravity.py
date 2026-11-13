"""Gravity in synchronous gauge: the metric, the Friedmann sum and the initial mode.

Conventions follow CAMB: ``k`` in 1/Mpc, conformal time ``tau`` in Mpc,
``etak = k eta``, velocities ``q_i = (1 + w_i) theta_i / k`` and density
coefficients ``grho_i_t = 8 pi G rho_i a^2``. The metric enters the fluid
equations of :mod:`discoeb.species` through ``0.5 h' = k z`` and the shear
``sigma`` (see :class:`Metric`). The curved-geometry relations follow
CLASS's ``perturbations.c`` with ``K = -grhok / 3`` and ``s2^2 = 1 - 3K/k^2``.

The components here are neither species nor interactions. :class:`Background`
turns the summed density ``grho`` into the expansion rate, :class:`Metric`
holds ``etak`` and the Einstein constraints, :class:`Curvature` the spatial
curvature, and :class:`AdiabaticMode` the growing-mode quantities the
species' initial conditions read. Every model in :mod:`discoeb.models`
includes all four.
"""

from __future__ import annotations

import math

from .component import Component, Parameter, Species, State

# Density coefficients in CAMB's convention, ``8 pi G rho / c^2`` in 1/Mpc^2:
# ``H^2 = sum_i grho_i / 3`` gives the Hubble rate in 1/Mpc and ``a'/a = a H``
# the conformal one. The perturbation kernel's ``grho_i_t = 8 pi G rho_i a^2``
# are these times ``a^2``.

GRHOM_PER_H0SQ = 3.33795017e-11  # 8 pi G rho_crit / c^2 per (H0 in km/s/Mpc)^2
GRHOG_PER_TCMB4 = 1.49594245e-13  # photons, per K^4
GRHOR_PER_TCMB4 = 3.39739477e-14  # one massless neutrino flavour, per K^4
C2OK = 1.62581581e4  # K / eV: amnu = m_nu c^2 / (k_B T_nu0) = mnu * C2OK / Tcmb

SUMS = ("grho", "gpres", "dgrho", "dgq", "dgpi")
"""The quantities summed over species.

``grho`` is the Friedmann density and ``gpres = Sum 8 pi G p_i a^2`` the
pressure (CAMB's, so the comoving curvature perturbation can divide by
``rho + p``). ``dgrho = Sum grho_i_t delta_i`` and ``dgq = Sum grho_i_t q_i``
source the Einstein constraints. ``dgpi = Sum grho_i_t pi_i`` is the
anisotropic stress the line-of-sight sources' Weyl potential needs. Spatial
curvature adds ``grhok`` to ``grho`` but no pressure, since it is no fluid."""


class Background(Component):
    """Powers of the scale factor, the Friedmann equation, ``1/tau`` and ``1/k^2``.

    It declares the Hubble constant ``H0`` (km/s/Mpc), which the
    recombination rates read too, and the critical density ``grhom`` derived
    from it.
    """

    H0 = Parameter(directed=True)
    grhom = Parameter(value=lambda H0: GRHOM_PER_H0SQ * H0**2)

    def a2(self, a):
        return a * a

    def inv_a(self, a):
        return 1.0 / a

    def inv_a2(self, inv_a):
        return inv_a * inv_a

    def inv_tau(self, tau):
        return 1.0 / tau

    def inv_k2(self, k):
        return 1.0 / (k * k)

    def adotoa(self, grho):
        """The conformal Hubble rate ``a'/a = sqrt(grho / 3)``."""
        return math.sqrt(grho * (1.0 / 3.0))


class AdiabaticMode(Component):
    """The adiabatic growing mode deep in radiation domination.

    CAMB's ``initial()`` series at ``x = k tau << 1``, normalised to unit
    comoving curvature: the neutrino fraction of the radiation ``Rv``, the
    matter-radiation parameter ``om`` and their combinations. The host
    evaluates them from the background densities when it assembles the
    initial state, and the states' ``initial`` functions read them by name.
    The primordial spectrum ``A_s (k / k_p)^(n_s - 1)`` scales the mode. The
    solve never reads it, the matter power spectrum does.
    """

    A_s = Parameter()
    n_s = Parameter()
    k_p = Parameter(0.05)

    def x(self, k, tau):
        return k * tau

    def Rv(self, grhog, grhornomass):
        return grhornomass / (grhog + grhornomass)

    def Rp15(self, Rv):
        return 4.0 * Rv + 15.0

    def om(self, grhob, grhoc, grhog, grhornomass):
        return (grhob + grhoc) / math.sqrt(3.0 * (grhog + grhornomass))

    def omtau(self, om, tau):
        return om * tau


class Curvature(Species):
    """Spatial curvature: a density ``grhok = 8 pi G rho_K a^2`` and the metric factors."""

    Omegak = Parameter(0.0)
    grhok = Parameter(value=lambda grhom, Omegak: grhom * Omegak, directed=True)

    def grho(self, grhok):
        return grhok

    def s2(self, grhok, inv_k2):
        """``s2^2 = 1 - 3K/k^2 = 1 + grhok/k^2`` (1 when flat)."""
        return 1.0 + grhok * inv_k2

    def inv_s2(self, s2):
        return 1.0 / s2

    def Kcurv(self, grhok):
        return -grhok * (1.0 / 3.0)


class Metric(Component):
    """The synchronous-gauge metric: ``etak = k eta`` and the Einstein constraints.

    ``h' = (k^2 s2^2 eta + 1.5 a^2 delta_rho) / (0.5 aH)`` with ``z = h'/(2k)``,
    ``eta' = (1.5 a^2 (rho+p) theta + 0.5 K h') / (k^2 s2^2)``, and the shear
    ``sigma = k alpha = (z + 1.5 dgq/k^2) / s2^2``.
    """

    etak = State(
        initial=lambda k, x, Rp15: -k * (1.0 - x * x / 12.0 * (-10.0 / Rp15 + 1.0))
    )

    def z(self, dgrho, k, s2, etak, adotoa):
        return (0.5 * dgrho / k + s2 * etak) / adotoa

    def sigma(self, z, dgq, inv_k2, inv_s2):
        return (z + 1.5 * dgq * inv_k2) * inv_s2

    def R(self, etak, k, adotoa, dgq, grho, gpres, grhok):
        """The comoving curvature perturbation ``R``, unity for the adiabatic mode.

        ``R`` is the gauge-invariant curvature on slices where the total
        momentum density vanishes. From synchronous gauge the slicing shifts
        by ``T = theta / k^2``, with ``theta = k dgq / (grho_fluid + gpres)``
        the total velocity divergence (``dgq = 8 pi G a^2 (rho + p) theta /
        k``), and the curvature ``eta`` becomes ``eta + aH T``. The sign is
        CAMB's: the initial conditions set ``R = 1`` (``etak = -k`` at
        ``k tau << 1``). Ma & Bertschinger's ``phi`` convention flips it, so
        in matter domination the Newtonian potentials are
        ``phi = psi = -3 R / 5`` outside the horizon. ``R`` stays constant on
        super-horizon scales up to ``O((k tau)^2)`` and decays with the
        mode's potentials through horizon entry.
        """
        return -etak / k - adotoa * dgq / (k * (grho - grhok + gpres))

    def d_etak(self, dgq, Kcurv, z, inv_s2):
        return (0.5 * dgq + Kcurv * z) * inv_s2
