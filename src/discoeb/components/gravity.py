"""
Gravity in synchronous gauge including the metric, Friedmann sum and initial mode.

:class:`Kinematics` holds the powers of the scale factor and the wave number
every component reads. :class:`Background` defines the Friedmann equation, turning
the summed density ``grho`` into the expansion rate. :class:`Metric`
holds ``etak`` and the Einstein constraints, :class:`Curvature` the spatial
curvature, and :class:`AdiabaticMode` the growing-mode quantities the
species' initial conditions read.
"""

from __future__ import annotations

import math

from jax.typing import ArrayLike

from ._component import Component, Derived, Gravity, Parameter, State, Sum

# Density coefficients in CAMB's convention, ``8 pi G rho / c^2`` in 1/Mpc^2:
# ``H^2 = sum_i grho_i / 3`` gives the Hubble rate in 1/Mpc and ``a'/a = a H``
# the conformal one. The perturbation kernel's ``grho_i_t = 8 pi G rho_i a^2``
# are these times ``a^2``.

GRHOM_PER_H0SQ = 3.33795017e-11  # 8 pi G rho_crit / c^2 per (H0 in km/s/Mpc)^2
GRHO_PER_OMEGA = 1.0e4 * GRHOM_PER_H0SQ  # per unit physical density omega = Omega h^2
GRHOG_PER_TCMB4 = 1.49594245e-13  # photons, per K^4
GRHOR_PER_TCMB4 = 3.39739477e-14  # one massless neutrino flavour, per K^4
C2OK = 1.62581581e4  # K / eV: amnu = m_nu c^2 / (k_B T_nu0) = mnu * C2OK / Tcmb


class Kinematics(Component):
    """Powers of the scale factor, ``1/tau`` and ``1/k^2``, for every component to read."""

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


class Background(Gravity):
    """The Friedmann equation of general relativity: the expansion rate from the fluids.

    The species give their densities as physical ones, ``omega = Omega
    h^2``, so the Hubble constant is no parameter: it is what this rate
    comes to today, ``H0 = adotoa(a = 1)``
    (:attr:`Model.H0 <discoeb.models.Model.H0>`). A theory with another
    Friedmann equation replaces this component and provides ``adotoa`` (and
    ``adotoa_prime`` if something reads it) its own way.
    """

    grho = Sum()
    """The Friedmann density ``Sum 8 pi G rho_i a^2``. Curvature is no fluid
    and adds nothing: the Friedmann equation reads it as ``Kcurv``."""
    gpres = Sum()
    """The pressure ``Sum 8 pi G p_i a^2``."""

    def adotoa(self, grho, Kcurv):
        """The conformal Hubble rate ``a'/a = sqrt(grho / 3 - K)``."""
        return math.sqrt(grho * (1.0 / 3.0) - Kcurv)

    def adot(self, a, adotoa):
        """``a' = a adotoa``, the rate of the scale factor in conformal time: the solve's clock."""
        return a * adotoa

    def adotoa_prime(self, grho, gpres):
        """``(a'/a)' = -(grho + 3 gpres) / 6``, the acceleration equation. Curvature drops out."""
        return -(grho + 3.0 * gpres) * (1.0 / 6.0)


class AdiabaticMode(Component):
    """The adiabatic growing mode deep in radiation domination.

    CAMB's ``initial()`` series at ``x = k tau << 1``, normalised to unit
    comoving curvature: the free-streaming fraction of the radiation ``Rv``,
    the matter-radiation parameter ``om`` and their combinations. Both come
    from sums over the species (``grho_matter``, ``grho_radiation``,
    ``grho_free_streaming``), ratios in which the scale factor cancels. The host
    evaluates them from the background densities when it assembles the
    initial state, and the states' ``initial`` functions read them by name.
    The primordial spectrum ``A_s (k / k_p)^(n_s - 1)`` scales the mode. The
    solve never reads it, the matter power spectrum does.
    """

    A_s: ArrayLike = Parameter()
    n_s: ArrayLike = Parameter()
    k_p: ArrayLike = Parameter(0.05)

    grho_matter = Sum()
    """The density of the pressureless species."""
    grho_radiation = Sum()
    """The density of the relativistic species."""
    grho_free_streaming = Sum()
    """The density of the radiation that streams freely (the neutrinos)."""

    def x(self, k, tau):
        return k * tau

    def Rv(self, grho_free_streaming, grho_radiation):
        return grho_free_streaming / grho_radiation

    def Rp15(self, Rv):
        return 4.0 * Rv + 15.0

    def om(self, grho_matter, grho_radiation):
        """``grho_m / sqrt(3 grho_r)`` of the densities today: ``a`` cancels between ``1/a`` and ``1/a^2``."""
        return grho_matter / math.sqrt(3.0 * grho_radiation)

    def omtau(self, om, tau):
        return om * tau


class Curvature(Gravity):
    """Spatial curvature ``K = -grhok / 3``, with ``omega_k = Omega_k h^2``, and the metric factors.

    It is geometry, not a fluid: it enters the Friedmann equation through
    ``Kcurv`` and none of the sums.
    """

    omega_k: ArrayLike = Parameter(0.0)
    grhok: ArrayLike = Derived(lambda omega_k: GRHO_PER_OMEGA * omega_k, directed=True)

    def s2(self, grhok, inv_k2):
        """``s2^2 = 1 - 3K/k^2 = 1 + grhok/k^2`` (1 when flat)."""
        return 1.0 + grhok * inv_k2

    def inv_s2(self, s2):
        return 1.0 / s2

    def Kcurv(self, grhok):
        return -grhok * (1.0 / 3.0)


class Metric(Gravity):
    """The synchronous-gauge metric: ``etak = k eta`` and the Einstein constraints.

    ``h' = (k^2 s2^2 eta + 1.5 a^2 delta_rho) / (0.5 aH)`` with ``z = h'/(2k)``,
    ``eta' = (1.5 a^2 (rho+p) theta + 0.5 K h') / (k^2 s2^2)``, and the shear
    ``sigma = k alpha = (z + 1.5 dgq/k^2) / s2^2``.
    """

    etak = State(
        initial=lambda k, x, Rp15: -k * (1.0 - x * x / 12.0 * (-10.0 / Rp15 + 1.0))
    )

    dgrho = Sum()
    """The density perturbation ``Sum grho_i_t delta_i``, which sources ``z``."""
    dgq = Sum()
    """The momentum density ``Sum grho_i_t q_i``, which sources ``sigma`` and ``etak``."""
    dgrho_streaming = Sum()
    """The density perturbation of the species under the radiation streaming
    approximation, which ``dgrho`` leaves out (CLASS takes ``h'`` without them)."""
    dgpi = Sum()
    """The anisotropic stress ``Sum grho_i_t pi_i``."""
    # The equations here read neither of the last two. The line-of-sight
    # sources' Weyl potential reads both.

    def z(self, dgrho, k, s2, etak, adotoa):
        return (0.5 * dgrho / k + s2 * etak) / adotoa

    def sigma(self, z, dgq, inv_k2, inv_s2):
        return (z + 1.5 * dgq * inv_k2) * inv_s2

    def R(self, etak, k, adotoa, dgq, grho, gpres):
        """The comoving curvature perturbation ``R``, unity for the adiabatic mode.

        ``R`` is the gauge-invariant curvature on slices where the total
        momentum density vanishes. From synchronous gauge the slicing shifts
        by ``T = theta / k^2``, with ``theta = k dgq / (grho + gpres)``
        the total velocity divergence (``dgq = 8 pi G a^2 (rho + p) theta /
        k``), and the curvature ``eta`` becomes ``eta + aH T``. The sign is
        CAMB's: the initial conditions set ``R = 1`` (``etak = -k`` at
        ``k tau << 1``). Ma & Bertschinger's ``phi`` convention flips it, so
        in matter domination the Newtonian potentials are
        ``phi = psi = -3 R / 5`` outside the horizon. ``R`` stays constant on
        super-horizon scales up to ``O((k tau)^2)`` and decays with the
        mode's potentials through horizon entry.
        """
        return -etak / k - adotoa * dgq / (k * (grho + gpres))

    def d_etak(self, dgq, Kcurv, z, inv_s2):
        return (0.5 * dgq + Kcurv * z) * inv_s2
