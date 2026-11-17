"""
Gravity in synchronous gauge including the metric, Friedmann sum and initial mode.

The variables are those of Ma & Bertschinger (1995). :class:`Background`
turns the summed density ``rho`` into the conformal Hubble rate ``aH``.
:class:`Metric` holds ``eta`` and the Einstein constraints for ``h'`` and
``alpha``, :class:`Curvature` the spatial curvature, and
:class:`AdiabaticMode` the growing-mode quantities the species' initial
conditions read.
"""

from __future__ import annotations

import math

from jax.typing import ArrayLike

from ._component import Component, Gravity, Parameter, State, Sum, derivative

# Every density is a physical density parameter, ``omega_i(a) = rho_i(a) /
# rho_100`` with ``rho_100 = 3 H100^2 / (8 pi G)`` the critical density of a
# universe expanding at 100 km/s/Mpc. A species' value today is its
# parameter, ``omega_c = Omega_c h^2``. The Friedmann equation is then
# ``H^2 = H100^2 rho - K / a^2`` and ``4 pi G a^2 rho`` is ``1.5 H100^2 a^2
# rho``, in 1/Mpc^2.

H100 = 1.0 / 2997.92458  # 100 km/s/Mpc over c, in 1/Mpc
H100_SQ = H100 * H100
OMEGA_G_PER_TCMB4 = 4.481620077629859e-07  # photons, per K^4
OMEGA_NU_PER_TCMB4 = 1.0178087140228338e-07  # one massless neutrino flavour, per K^4
C2OK = 1.62581581e4  # K / eV: amnu = m_nu c^2 / (k_B T_nu0) = m_nu * C2OK / T_cmb


class Background(Gravity):
    """The Friedmann equation of general relativity: the expansion rate from the fluids.

    The species give their densities as physical ones, ``omega = Omega
    h^2``, so the Hubble constant is no parameter: it is what this rate
    comes to today, ``H_0 = aH(a = 1)``
    (:attr:`Model.H_0 <discoeb.models.Model.H_0>`). A theory with another
    Friedmann equation replaces this component and provides ``aH`` (and
    ``aH_prime`` if something reads it) its own way.
    """

    rho = Sum()
    """The total density ``Sum omega_i(a)``. Curvature is no fluid and adds
    nothing: the Friedmann equation reads it as ``K``."""
    pressure = Sum()
    """The total pressure ``Sum w_i omega_i(a)``."""

    def four_pi_G_a2(self, a):
        """``4 pi G a^2`` for a density in units of ``omega``, in 1/Mpc^2."""
        return 1.5 * H100_SQ * a * a

    def aH(self, rho, K, a):
        """The conformal Hubble rate ``a'/a = sqrt(H100^2 a^2 rho - K)``."""
        return math.sqrt(H100_SQ * a * a * rho - K)

    def a_prime(self, a, aH):
        """``a' = a aH``, the rate of the scale factor in conformal time: the solve's clock."""
        return a * aH

    def aH_prime(self, rho, pressure, four_pi_G_a2):
        """``(a'/a)' = -(4 pi G a^2 / 3) (rho + 3 p)``, the acceleration equation. Curvature drops out."""
        return -four_pi_G_a2 * (rho + 3.0 * pressure) * (1.0 / 3.0)


class AdiabaticMode(Component):
    """The adiabatic growing mode deep in radiation domination.

    The series of Ma & Bertschinger at ``x = k tau << 1``, normalised to
    unit comoving curvature: the free-streaming fraction of the radiation
    ``Rv``, the matter-radiation parameter ``om`` and their combinations.
    Both come from sums over the species (``rho_matter``, ``rho_radiation``,
    ``rho_free_streaming``). The host evaluates them from the background
    densities when it assembles the initial state, and the states'
    ``initial`` functions read them by name. The primordial spectrum ``A_s
    (k / k_p)^(n_s - 1)`` scales the mode. The solve never reads it, the
    matter power spectrum does.
    """

    A_s: ArrayLike = Parameter(solved=False)
    n_s: ArrayLike = Parameter(solved=False)
    k_p: ArrayLike = Parameter(0.05, solved=False)

    rho_matter = Sum()
    """The density of the pressureless species."""
    rho_radiation = Sum()
    """The density of the relativistic species."""
    rho_free_streaming = Sum()
    """The density of the radiation that streams freely (the neutrinos)."""

    def x(self, k, tau):
        return k * tau

    def Rv(self, rho_free_streaming, rho_radiation):
        return rho_free_streaming / rho_radiation

    def Rp15(self, Rv):
        return 4.0 * Rv + 15.0

    def om(self, rho_matter, rho_radiation, a):
        """``H100 omega_m / sqrt(omega_r)`` of the densities today.

        The scale factor cancels: ``a`` times ``a^-3`` over ``a^-2``.
        """
        return H100 * a * rho_matter / math.sqrt(rho_radiation)

    def omtau(self, om, tau):
        return om * tau


class Curvature(Gravity):
    """Spatial curvature ``K = -omega_k H100^2``, with ``omega_k = Omega_k h^2``, and the metric factor.

    It is geometry, not a fluid: it enters the Friedmann equation through
    ``K`` and none of the sums.
    """

    omega_k: ArrayLike = Parameter(0.0, directed=True, differentiable=True)

    def K(self, omega_k):
        return -H100_SQ * omega_k

    def curved(self, K, k):
        """``1 - 3 K / k^2``, the factor curvature puts on ``k^2`` in the constraints (1 when flat)."""
        inv_k2 = 1.0 / (k * k)
        return 1.0 - 3.0 * K * inv_k2

    def inv_curved(self, curved):
        return 1.0 / curved


class Metric(Gravity):
    """The synchronous-gauge metric: ``eta`` and the Einstein constraints.

    With ``k_c^2 = k^2 (1 - 3 K / k^2)``, which is ``k^2`` when flat,

    ``h' = 2 (k_c^2 eta + 4 pi G a^2 delta_rho) / aH``,
    ``eta' = (4 pi G a^2 (rho + p) theta + K h' / 2) / k_c^2``,
    ``alpha = (h' + 6 eta') / (2 k^2) = (h' / 2 + 12 pi G a^2 (rho + p) theta / k^2) / k_c^2``.
    """

    eta = State(initial=lambda x, Rp15: -(1.0 - x * x / 12.0 * (-10.0 / Rp15 + 1.0)))

    delta_rho = Sum()
    """The density perturbation ``Sum omega_i delta_i``, which sources ``h'``."""
    momentum = Sum()
    """The momentum density ``Sum (omega_i + p_i) theta_i``, which sources ``eta'`` and ``alpha``."""
    delta_rho_streaming = Sum()
    """The density perturbation of the species under the radiation streaming
    approximation, which ``delta_rho`` leaves out (CLASS takes ``h'`` without them)."""
    stress = Sum()
    """The anisotropic stress ``Sum (omega_i + p_i) sigma_i``."""
    # The equations here read neither of the last two. The line-of-sight
    # sources' Weyl potential reads both.

    def h_prime(self, delta_rho, four_pi_G_a2, k, curved, eta, aH):
        return (four_pi_G_a2 * delta_rho + (k * k * curved) * eta) * (2.0 / aH)

    def alpha(self, h_prime, momentum, four_pi_G_a2, k, inv_curved):
        inv_k2 = 1.0 / (k * k)
        # the state-independent factors first: each product with a state
        # is one more operation in every Jacobian sweep
        return (0.5 * h_prime + (3.0 * four_pi_G_a2 * inv_k2) * momentum) * (
            inv_k2 * inv_curved
        )

    def R(self, eta, k, aH, momentum, rho, pressure):
        """The comoving curvature perturbation ``R``, unity for the adiabatic mode.

        ``R`` is the gauge-invariant curvature on slices where the total
        momentum density vanishes. From synchronous gauge the slicing shifts
        by ``T = theta / k^2``, with ``theta = momentum / (rho + p)`` the
        total velocity divergence, and the curvature ``eta`` becomes ``eta +
        aH T``. The initial conditions set ``R = 1`` (``eta = -1`` at ``k
        tau << 1``), so in matter domination the Newtonian potentials are
        ``phi = psi = -3 R / 5`` outside the horizon. ``R`` stays constant
        on super-horizon scales up to ``O((k tau)^2)`` and decays with the
        mode's potentials through horizon entry.
        """
        inv_k2 = 1.0 / (k * k)
        return -eta - aH * momentum * inv_k2 / (rho + pressure)

    @derivative("eta")
    def momentum_constraint(self, momentum, four_pi_G_a2, K, h_prime, k, inv_curved):
        inv_k2 = 1.0 / (k * k)
        return (four_pi_G_a2 * momentum + (0.5 * K) * h_prime) * (inv_k2 * inv_curved)
