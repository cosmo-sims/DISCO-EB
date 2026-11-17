"""
Each species is a :mod:`discoeb.components._component` that owns its own state.

The variables are those of Ma & Bertschinger (1995): a density contrast
``delta`` and a velocity divergence ``theta`` per fluid. The Boltzmann
hierarchies are vector states: ``F`` holds the photons' ``F_0 .. F_lmax``,
``E`` holds ``E_2 .. E_lmax``, ``N`` the neutrino multipoles and ``psi``
one row per massive-neutrino momentum bin. A density is a physical density
parameter, ``omega_i(a)`` (:mod:`discoeb.components.gravity`). Their equations loop
over multipoles and share the free-streaming tail in :func:`stream`. The
truncations are component settings: ``LMAX_G`` and ``LMAX_POL`` on
:class:`Photons`, ``LMAX_NR`` on :class:`MasslessNeutrinos`, ``LMAX_NU`` and
``NQMAX`` on :class:`MassiveNeutrinos`.
"""

from __future__ import annotations

import math

from jax.typing import ArrayLike

from ._component import (
    Component,
    Derived,
    Parameter,
    Setting,
    Species,
    State,
    contributes,
    derivative,
)
from ._util import neutrino_momentum_bins
from .gravity import C2OK, OMEGA_G_PER_TCMB4, OMEGA_NU_PER_TCMB4


def _needs_octupole(**lmax):
    """Raise ``ValueError`` unless every hierarchy reaches ``l = 3``.

    The quadrupole equations couple to the octupole.
    """

    low = {name: ell for name, ell in lmax.items() if ell < 3}
    if low:
        raise ValueError(f"a hierarchy must reach at least the octupole; got {low}")


# --- matter -------------------------------------------------------------------------
# Each density today, ``omega_i``, is a directed parameter, a leading column
# of the parameter row that a derivative may move.


class ColdDarkMatter(Species):
    """Cold dark matter. Synchronous gauge leaves the choice of the initial hypersurface open.
    The equations of :mod:`discoeb.components.gravity` take the one in which this
    species is at rest, so it has a density contrast but no velocity
    (``fixes_gauge``).
    """

    fixes_gauge = True

    # adiabatic mode: 3/4 of the photon density contrast
    delta_c = State(initial=lambda F: 0.75 * F[0])

    omega_c: ArrayLike = Parameter(directed=True, differentiable=True)  # Omega_c h^2

    def rho_c(self, omega_c, a):
        inv_a = 1.0 / a
        return omega_c * inv_a * inv_a * inv_a

    @contributes("rho", "rho_matter")
    def rho_cdm(self, rho_c):
        return rho_c

    @contributes("delta_rho")
    def delta_rho_c(self, rho_c, delta_c):
        return rho_c * delta_c

    @derivative("delta_c")
    def continuity_c(self, h_prime):
        """CDM is pressureless and at rest in synchronous gauge: ``delta_c' = -h' / 2``."""
        return -0.5 * h_prime


class Baryons(Species):
    """Baryon continuity and Euler equations, without the Thomson drag.

    :class:`ThomsonScattering` adds the drag.
    """

    delta_b = State(initial=lambda F: 0.75 * F[0])
    theta_b = State(initial=lambda F, k: 0.75 * k * F[1])

    # The recombination rates read omega_b and the helium fraction Y_He too.
    omega_b: ArrayLike = Parameter(directed=True, differentiable=True)  # Omega_b h^2
    # not differentiable: dP/dYHe misses central differences by 0.8 percent
    Y_He: ArrayLike = Parameter(directed=True)

    def rho_b(self, omega_b, a):
        inv_a = 1.0 / a
        return omega_b * inv_a * inv_a * inv_a

    @contributes("rho", "rho_matter")
    def rho_baryons(self, rho_b):
        return rho_b

    @contributes("delta_rho")
    def delta_rho_b(self, rho_b, delta_b):
        return rho_b * delta_b

    @contributes("momentum")
    def momentum_b(self, rho_b, theta_b):
        return rho_b * theta_b

    @derivative("delta_b")
    def continuity_b(self, h_prime, theta_b):
        return -theta_b - 0.5 * h_prime

    @derivative("theta_b")
    def euler_b(self, aH, theta_b, k, cs2, delta_b):
        return -aH * theta_b + k * k * cs2 * delta_b


class Fluid(Species):
    """A perfect fluid with a constant equation of state ``w`` and rest-frame sound speed ``cs2``.

    The fluid equations of :class:`DarkEnergy` at constant ``w``, where the
    adiabatic sound speed is ``w`` itself:

    ``delta' = -(1+w)(theta + h'/2) - 3 (cs2 - w) aH delta
               - 9 (1+w)(cs2 - w) (aH)^2 theta / k^2``,
    ``theta' = -(1 - 3 cs2) aH theta + cs2 k^2 delta / (1+w)``,

    with no anisotropic stress. Its physical density today is ``omega``
    (``Omega h^2``) and scales as ``a^(-3(1+w))``. It starts in the adiabatic
    mode, ``delta = (1 + w) delta_c`` at rest. ``w = cs2 = 0`` is a second cold
    dark matter and ``w = cs2 = 1/3`` a dark radiation without shear. The
    growing mode's series does not count it: a subclass that is matter or
    radiation contributes its ``density`` to ``rho_matter`` or
    ``rho_radiation``.

    Its names (``delta``, ``theta``, ``w``, ``cs2``) are generic, so a model
    must rename it appropriately::

        LCDM.add(Fluid.named("dr"), name="LCDM+dr")   # dr_omega, dr_w, dr_delta ...
    """

    delta = State(initial=lambda w, delta_c: (1.0 + w) * delta_c)
    theta = State()

    omega: ArrayLike = Parameter(0.0, directed=True)
    w: ArrayLike = Parameter(0.0, column=True)
    cs2: ArrayLike = Parameter(0.0, column=True)

    def __init__(self, **settings):
        super().__init__(**settings)
        if type(self) is Fluid:
            # Unnamed, its cs2 would stand in for the baryons' sound speed.
            raise ValueError("a Fluid needs a name of its own: Fluid.named('dr')")

    @classmethod
    def check(cls, omega, w):
        """The Euler equation divides by ``1 + w``."""
        if float(w) == -1.0:
            raise ValueError("a fluid needs w != -1; use a cosmological constant")
        if float(omega) < 0.0:
            raise ValueError(f"a fluid's density omega = {omega} must not be negative")

    def density(self, omega, a, w):
        return omega * a ** (-3.0 - 3.0 * w)

    @contributes("rho")
    def rho_fluid(self, density):
        return density

    @contributes("pressure")
    def pressure_fluid(self, density, w):
        return w * density

    @contributes("delta_rho")
    def delta_rho_fluid(self, density, delta):
        return density * delta

    @contributes("momentum")
    def momentum_fluid(self, density, w, theta):
        return density * (1.0 + w) * theta

    @derivative("delta")
    def continuity(self, delta, theta, h_prime, k, aH, w, cs2):
        inv_k2 = 1.0 / (k * k)
        return (
            -(1.0 + w) * (theta + 0.5 * h_prime)
            - 3.0 * (cs2 - w) * aH * delta
            - 9.0 * (1.0 + w) * (cs2 - w) * aH**2 * inv_k2 * theta
        )

    @derivative("theta")
    def euler(self, delta, theta, k, aH, w, cs2):
        return -(1.0 - 3.0 * cs2) * aH * theta + cs2 / (1.0 + w) * k**2 * delta


# --- the radiation streaming approximation ------------------------------------------


class RadiationStreaming(Component):
    """The radiation streaming approximation of CLASS (Blas, Lesgourgues & Tram 2011).

    Well inside the horizon, once the photons decouple, the photon and
    massless-neutrino hierarchies oscillate at ``k`` about a particular
    solution driven by the metric, and the oscillation decays as
    ``1/(k tau)``. Where ``rsa`` is 1 the model swaps the hierarchies for
    that solution,

        delta = 4 (aH h' - k^2 eta) / k^2,   theta = -h'/2,   no shear, no polarization,

    (synchronous gauge), takes ``h'``
    without the streaming species as CLASS does, and freezes their states.
    The switch is CLASS's: ``k tau > RSA_K_TAU`` (45) and ``kappa' tau <
    RSA_KAPPA_TAU`` (``tau_c / tau > 5``). Like CLASS, which switches at a
    fixed ``tau_free_streaming``, it only turns on: past ``a = RSA_A_LATCH``
    it stays on through reionization. CLASS's Thomson corrections for
    reionization are missing. ``RSA = 0`` (``DISCO_RSA=0``) turns it off.

    It provides the switch ``rsa`` and the streaming solution ``rsa_delta``
    and ``rsa_theta``, and names no species. Each species that follows it reads
    those and adds its own density to the sum ``delta_rho_streaming``. The
    solution is general relativity's (:class:`NoRadiationStreaming` for
    another theory).
    """

    RSA = Setting(1)
    RSA_K_TAU = Setting(45.0)
    RSA_KAPPA_TAU = Setting(0.2)
    RSA_A_LATCH = Setting(0.02)

    def rsa(self, k, tau, opacity, a):
        # Branch-free steps 0.5 + 0.5 sign(x), and "or" as x + y - x y, so one
        # expression serves the device, plain floats and JAX arrays on the host.
        inside = 0.5 + 0.5 * math.copysign(1.0, k * tau - self.RSA_K_TAU)
        decoupled = 0.5 + 0.5 * math.copysign(1.0, self.RSA_KAPPA_TAU - opacity * tau)
        latched = 0.5 + 0.5 * math.copysign(1.0, a - self.RSA_A_LATCH)
        return self.RSA * inside * (decoupled + latched - decoupled * latched)

    def rsa_delta(self, aH, h_prime, eta, k):
        """The streaming radiation's density contrast, ``4 (aH h' - k^2 eta) / k^2``."""
        inv_k2 = 1.0 / (k * k)
        return 4.0 * aH * h_prime * inv_k2 - 4.0 * eta

    def rsa_theta(self, h_prime):
        """The streaming radiation's velocity divergence, ``theta = -h' / 2``."""
        return -0.5 * h_prime


class NoRadiationStreaming(Component):
    """The full hierarchies at all times: what :class:`RadiationStreaming` provides, switched off.

    ``LCDM.replace(RadiationStreaming, NoRadiationStreaming, name=...)`` is
    a model without the approximation, for a theory of gravity whose
    constraints the streaming solution does not know. The constants fold
    away when the kernel compiles.
    """

    def rsa(self):
        return 0.0

    def rsa_delta(self):
        return 0.0

    def rsa_theta(self):
        return 0.0


# --- radiation ----------------------------------------------------------------------


def stream(k, kv, h_prime, alpha, dlnf0, x, inv_tau, output):
    """Free streaming of a hierarchy ``x`` (``x_0 .. x_L``), into ``output``.

    The lowest three multipoles carry the metric sources (Ma & Bertschinger,
    synchronous gauge)::

        x_0' = -kv x_1 + (h' / 6) dlnf0
        x_1' = (kv / 3) (x_0 - 2 x_2)
        x_2' = (kv / 5) (2 x_1 - 3 x_3) - (2/15) k^2 alpha dlnf0

    and above them ``x_l' = kv (l x_{l-1} - (l+1) x_{l+1}) / (2l+1)``,
    closing at ``L = len(x) - 1`` with ``x_L' = kv x_{L-1} - (L+1) x_L / tau``.
    ``kv`` is ``k v`` and ``dlnf0`` the log-derivative of the background
    distribution: ``k`` and ``-4`` for massless particles, the bin's own for a
    massive-neutrino momentum bin. The Thomson interaction adds collisional
    damping to the same derivatives.
    """

    output[0] = -kv * x[1] + h_prime * (1.0 / 6.0) * dlnf0
    output[1] = kv * (1.0 / 3.0) * (x[0] - 2.0 * x[2])
    output[2] = (
        kv * (1.0 / 5.0) * (2.0 * x[1] - 3.0 * x[3])
        - (2.0 / 15.0) * k * k * alpha * dlnf0
    )
    lmax = len(x) - 1
    for ell in range(3, lmax):
        output[ell] = (
            kv * (ell * x[ell - 1] - (ell + 1) * x[ell + 1]) * (1.0 / (2 * ell + 1))
        )
    output[lmax] = kv * x[lmax - 1] - (lmax + 1) * x[lmax] * inv_tau


class Photons(Species):
    """Photon temperature multipoles ``F`` (``F_0 .. F_LMAX_G``) and E-modes ``E``.

    ``F_0 = delta_g``, ``F_1 = 4 theta_g / (3 k)`` and ``F_2 = 2 sigma_g``.
    ``E[l - 2]`` is ``E_l`` for ``l = 2 .. LMAX_POL``. This is the
    collisionless part of the hierarchy. :class:`ThomsonScattering` damps the
    multipoles and sources the polarization. The E-mode hierarchy couples
    upwards with ``(l+3)(l-1)/(l+1)`` in place of ``l+1`` and closes with
    ``k L E_{L-1} / (2L+1) - (L+3) E_L / tau``. Both hierarchies stream
    freely above their lowest three (``F``) or one (``E``) multipoles.
    The E-modes start at zero.
    """

    LMAX_G = Setting(15)
    LMAX_POL = Setting(15)

    def _F_initial(self, x, omtau, output):
        """The monopole and dipole of the growing mode."""
        output[0] = x * x / 3.0 * (1.0 - omtau / 5.0)
        output[1] = x**3 / 27.0 * (1.0 - omtau / 5.0)

    F = State(lambda self: self.LMAX_G + 1, initial=_F_initial)
    E = State(lambda self: self.LMAX_POL - 1)

    T_cmb: ArrayLike = Parameter(
        directed=True, differentiable=True
    )  # K, read by the recombination rates too
    omega_g: ArrayLike = Derived(
        lambda T_cmb: OMEGA_G_PER_TCMB4 * T_cmb**4, directed=True
    )

    def __init__(self, **settings):
        super().__init__(**settings)
        _needs_octupole(LMAX_G=self.LMAX_G, LMAX_POL=self.LMAX_POL)

    def rho_g(self, omega_g, a):
        inv_a = 1.0 / a
        inv_a2 = inv_a * inv_a
        return omega_g * inv_a2 * inv_a2

    @contributes("rho", "rho_radiation")
    def rho_photons(self, rho_g):
        return rho_g

    def delta_g(self, F, rsa, rsa_delta):
        return F[0] + rsa * (rsa_delta - F[0])

    def theta_g(self, F, k, rsa, rsa_theta):
        theta = 0.75 * k * F[1]
        return theta + rsa * (rsa_theta - theta)

    def sigma_g(self, F, rsa):
        return 0.5 * (1.0 - rsa) * F[2]

    def E2(self, E, rsa):
        """The E-mode quadrupole, ``E[0]`` (none under the RSA)."""
        return (1.0 - rsa) * E[0]

    @contributes("delta_rho")
    def delta_rho_g(self, rho_g, F, rsa):
        """The hierarchy's density, zero under the RSA.

        As in CLASS, the ``h'`` that the RSA density reads leaves the
        streaming species out.
        """
        return rho_g * (1.0 - rsa) * F[0]

    @contributes("momentum")
    def momentum_g(self, rho_g, theta_g):
        return (4.0 / 3.0) * rho_g * theta_g

    @contributes("delta_rho_streaming")
    def delta_rho_streaming_g(self, rsa, rsa_delta, rho_g):
        """The photons' density under the RSA.

        ``delta_rho`` (and so ``h'``) leaves it out, but as in CLASS
        everything else that reads the density includes it.
        """
        return rsa * rsa_delta * rho_g

    @contributes("pressure")
    def pressure_g(self, rho_g):
        return rho_g * (1.0 / 3.0)

    @contributes("stress")
    def stress_g(self, rho_g, sigma_g):
        return (4.0 / 3.0) * rho_g * sigma_g

    @derivative("F")
    def streaming_g(self, k, h_prime, alpha, F, tau, rsa, output):
        inv_tau = 1.0 / tau
        stream(k, k, h_prime, alpha, -4.0, F, inv_tau, output)
        for ell in range(len(F)):
            output[ell] = (1.0 - rsa) * output[ell]

    @derivative("E")
    def streaming_E(self, k, E, tau, rsa, output):
        inv_tau = 1.0 / tau
        lmax = len(E) + 1
        output[0] = -k * E[1] * (1.0 / 3.0)
        for ell in range(3, lmax):
            upfac = (ell + 3.0) * (ell - 1.0) / (ell + 1.0)
            output[ell - 2] = (
                k * (ell * E[ell - 3] - upfac * E[ell - 1]) * (1.0 / (2 * ell + 1))
            )
        output[lmax - 2] = (
            k * lmax * E[lmax - 3] * (1.0 / (2 * lmax + 1))
            - (lmax + 3) * E[lmax - 2] * inv_tau
        )
        for ell in range(len(E)):
            output[ell] = (1.0 - rsa) * output[ell]


class MasslessNeutrinos(Species):
    """Massless neutrinos ``N`` (``N_0 .. N_LMAX_NR``) exhibit collisionless free streaming."""

    LMAX_NR = Setting(15)

    def _N_initial(self, x, Rv, Rp15, omtau, F, output):
        """The growing mode's neutrino monopole, dipole, quadrupole and octupole."""
        output[0] = F[0]
        output[1] = (4.0 * Rv + 23.0) / Rp15 * x**3 / 27.0
        output[2] = (
            -4.0
            / 3.0
            * x
            * x
            / Rp15
            * (1.0 + omtau / 4.0 * (4.0 * Rv - 5.0) / (2.0 * Rv + 15.0))
        )
        output[3] = -4.0 / 21.0 / Rp15 * x**3

    N = State(lambda self: self.LMAX_NR + 1, initial=_N_initial)

    N_eff: ArrayLike = Parameter(differentiable=True)
    omega_ur: ArrayLike = Derived(
        lambda N_eff, T_cmb: N_eff * (OMEGA_NU_PER_TCMB4 * T_cmb**4), directed=True
    )

    def __init__(self, **settings):
        super().__init__(**settings)
        _needs_octupole(LMAX_NR=self.LMAX_NR)

    def rho_ur(self, omega_ur, a):
        inv_a = 1.0 / a
        inv_a2 = inv_a * inv_a
        return omega_ur * inv_a2 * inv_a2

    @contributes("rho", "rho_radiation", "rho_free_streaming")
    def rho_neutrinos(self, rho_ur):
        return rho_ur

    @contributes("delta_rho")
    def delta_rho_ur(self, rho_ur, N, rsa):
        return rho_ur * (1.0 - rsa) * N[0]

    @contributes("momentum")
    def momentum_ur(self, rho_ur, N, k, rsa, rsa_theta):
        theta = 0.75 * k * N[1]
        return (4.0 / 3.0) * rho_ur * (theta + rsa * (rsa_theta - theta))

    @contributes("delta_rho_streaming")
    def delta_rho_streaming_ur(self, rsa, rsa_delta, rho_ur):
        return rsa * rsa_delta * rho_ur

    @contributes("pressure")
    def pressure_ur(self, rho_ur):
        return rho_ur * (1.0 / 3.0)

    @contributes("stress")
    def stress_ur(self, rho_ur, N, rsa):
        return (2.0 / 3.0) * rho_ur * (1.0 - rsa) * N[2]

    @derivative("N")
    def streaming_ur(self, k, h_prime, alpha, N, tau, rsa, output):
        inv_tau = 1.0 / tau
        stream(k, k, h_prime, alpha, -4.0, N, inv_tau, output)
        for ell in range(len(N)):
            output[ell] = (1.0 - rsa) * output[ell]


NQ_NU_BACKGROUND = 8
"""Momentum nodes of the massive-neutrino background density and pressure."""


class MassiveNeutrinos(Species):
    """Massive neutrinos on ``NQMAX`` momentum bins: ``psi[i, l]`` (Ma & Bertschinger).

    Bin ``i`` has node ``q_i``, weight ``w_i`` and ``dlnf0/dlnq``, and
    streams freely (:func:`stream`) with ``kv = k v_i``,
    ``v_i = q / sqrt(q^2 + a^2 amnu^2)``. The moments ``Sum w psi0 / v``,
    ``Sum w psi1`` and ``Sum w v psi2`` enter the Einstein sources. The
    background density ratio ``rhonu`` and pressure ratio ``pnu`` integrate
    the Fermi-Dirac distribution on :data:`NQ_NU_BACKGROUND` nodes. Deep in radiation domination the neutrinos are
    still relativistic, so each bin starts as the massless hierarchy,
    ``psi_l = -(1/4) N_l dlnf0/dlnq``. The quadrature nodes and weights are
    instance constants the providers read through ``self``. Each bin adds
    ``LMAX_NU + 1`` states. 3 bins give LCDM_MNU ``nvar = 98``.
    5 bins compile about 4x slower and gain no ``P(k)`` accuracy against
    CLASS.
    """

    LMAX_NU = Setting(15)
    NQMAX = Setting(3)

    def _psi_initial(self, N, output):
        """Still relativistic: each bin starts as the massless hierarchy scaled by dlnf0/dlnq."""
        for i in range(self.NQMAX):
            for ell in range(4):
                output[i, ell] = -0.25 * N[ell] * self.dlnf0[i]

    psi = State(lambda self: (self.NQMAX, self.LMAX_NU + 1), initial=_psi_initial)

    N_mnu: ArrayLike = Parameter(0.0)  # flavours, a count: not differentiable
    m_nu: ArrayLike = Parameter(0.0, differentiable=True)  # eV, each
    omega_nu_flavour: ArrayLike = Derived(
        lambda T_cmb: OMEGA_NU_PER_TCMB4 * T_cmb**4, directed=True
    )
    n_mnu: ArrayLike = Derived(lambda N_mnu: N_mnu, column=True)
    amnu: ArrayLike = Derived(lambda m_nu, T_cmb: m_nu * C2OK / T_cmb, directed=True)

    def __init__(self, **settings):
        super().__init__(**settings)
        _needs_octupole(LMAX_NU=self.LMAX_NU)
        q, w, dlnf0 = neutrino_momentum_bins(self.NQMAX)
        self.inv_q = tuple(1.0 / float(x) for x in q)
        self.weight = tuple(float(x) for x in w)
        self.dlnf0 = tuple(float(x) for x in dlnf0)
        q, w, _ = neutrino_momentum_bins(NQ_NU_BACKGROUND)
        self.nu_q = tuple(float(x) for x in q)
        self.nu_w = tuple(float(x) for x in w)

    def massive(self, amnu):
        """1 with a mass and 0 without, branch-free so it runs on arrays too."""
        return min(max(amnu * 1.0e30, 0.0), 1.0)

    def rhonu(self, a, amnu, massive):
        """``rho_nu / rho_nu(massless)`` of one flavour, 1 if massless."""
        total = 0.0
        for i in range(len(self.nu_q)):
            total += self.nu_w[i] * math.sqrt(1.0 + (a * amnu / self.nu_q[i]) ** 2)
        return 1.0 + massive * (total - 1.0)

    def pnu(self, a, amnu, massive):
        """``p_nu / rho_nu(massless)`` of one flavour: ``(rhonu - d rhonu / d ln a) / 3``, 1/3 if massless."""
        total = 0.0
        for i in range(len(self.nu_q)):
            total += self.nu_w[i] / math.sqrt(1.0 + (a * amnu / self.nu_q[i]) ** 2)
        return (1.0 + massive * (total - 1.0)) * (1.0 / 3.0)

    def rho_nu_massless(self, omega_nu_flavour, n_mnu, a):
        """The density the flavours would have without their mass."""
        inv_a = 1.0 / a
        inv_a2 = inv_a * inv_a
        return omega_nu_flavour * n_mnu * inv_a2 * inv_a2

    @contributes("rho")
    def rho_nu(self, rho_nu_massless, rhonu):
        return rho_nu_massless * rhonu

    @contributes("delta_rho")
    def delta_rho_nu(self, rho_nu_massless, a, amnu, psi):
        total = 0.0
        for i in range(self.NQMAX):
            inv_v = math.sqrt(1.0 + (a * amnu * self.inv_q[i]) ** 2)
            total += self.weight[i] * psi[i, 0] * inv_v
        return rho_nu_massless * total

    @contributes("momentum")
    def momentum_nu(self, rho_nu_massless, k, psi):
        total = 0.0
        for i in range(self.NQMAX):
            total += self.weight[i] * psi[i, 1]
        return rho_nu_massless * k * total

    @contributes("pressure")
    def pressure_nu(self, rho_nu_massless, pnu):
        """The background pressure, from ``pnu = (rhonu - d rhonu / d ln a) / 3``."""
        return rho_nu_massless * pnu

    @contributes("stress")
    def stress_nu(self, rho_nu_massless, a, amnu, psi):
        """The shear ``(2/3) Sum w v psi2``: the pressure weighting ``q/epsilon`` of a massive bin."""
        total = 0.0
        for i in range(self.NQMAX):
            inv_v = math.sqrt(1.0 + (a * amnu * self.inv_q[i]) ** 2)
            total += self.weight[i] * psi[i, 2] / inv_v
        return (2.0 / 3.0) * rho_nu_massless * total

    @derivative("psi")
    def streaming_nu(self, k, h_prime, alpha, a, amnu, psi, tau, output):
        inv_tau = 1.0 / tau
        for i in range(self.NQMAX):
            inv_v = math.sqrt(1.0 + (a * amnu * self.inv_q[i]) ** 2)
            kv = k * (1.0 / inv_v)
            stream(k, kv, h_prime, alpha, self.dlnf0[i], psi[i], inv_tau, output[i])


# --- dark energy --------------------------------------------------------------------


class _DarkEnergyDensity(Species):
    """The dark-energy density today and its part of the Friedmann sum, for either kind.

    ``omega_de`` is its physical density today, ``Omega_de h^2``.
    :meth:`Model.from_density_parameters
    <discoeb.models.Model.from_density_parameters>` finds the one that makes
    a model expand at a given ``H_0``.
    """

    omega_de: ArrayLike = Parameter(directed=True, differentiable=True)

    @contributes("rho")
    def rho_dark_energy(self, rho_de):
        return rho_de


class CosmologicalConstant(_DarkEnergyDensity):
    """A cosmological constant: density only, no perturbations."""

    def rho_de(self, omega_de):
        return omega_de

    @contributes("pressure")
    def pressure_lambda(self, rho_de):
        return -rho_de


class DarkEnergy(_DarkEnergyDensity):
    """A CPL fluid ``w = w_0 + w_a (1 - a)`` with rest-frame sound speed ``cs2_DE``.

    Synchronous-gauge fluid equations after Ballesteros & Lesgourgues 2010, with
    the adiabatic sound speed ``c_a^2 = w - w' / (3 (1+w) aH)``, ``w' = -w_a aH a``:

    ``delta_de' = -(1+w)(theta_de + h'/2) - 3 (cs2 - w) aH delta_de
                  - 9 (1+w)(cs2 - c_a^2) (aH)^2 theta_de / k^2``,
    ``theta_de' = -(1 - 3 cs2) aH theta_de + cs2 k^2 delta_de / (1+w)``.

    Valid for ``w != -1``. Use :class:`CosmologicalConstant` for ``w = -1``.
    The fluid starts in the adiabatic mode, ``delta_de = (1 + w_0) delta_c`` at rest.
    """

    # adiabatic, at rest
    delta_de = State(initial=lambda w_DE_0, delta_c: (1.0 + w_DE_0) * delta_c)
    theta_de = State()

    w_DE_0: ArrayLike = Parameter(-1.0, directed=True, differentiable=True)
    w_DE_a: ArrayLike = Parameter(0.0, directed=True, differentiable=True)
    # not differentiable: dP/dcs2_DE misses central differences by 1 percent
    cs2_DE: ArrayLike = Parameter(1.0, directed=True)

    @classmethod
    def check(cls, w_DE_0, w_DE_a):
        """The fluid equations divide by ``1 + w``."""
        if float(w_DE_0) == -1.0 and float(w_DE_a) == 0.0:
            raise ValueError(
                "it evolves a dark-energy fluid, which needs w != -1; use a model "
                "with a cosmological constant for w_DE_0 = -1, w_DE_a = 0"
            )

    def w_de(self, a, w_DE_0, w_DE_a):
        return w_DE_0 + w_DE_a * (1.0 - a)

    def rho_de(self, omega_de, a, w_DE_0, w_DE_a):
        """``omega_de`` times ``rho(a) / rho(1)`` of the CPL equation of state."""
        return (
            omega_de
            * a ** (-3.0 * (1.0 + w_DE_0 + w_DE_a))
            * math.exp(3.0 * (a - 1.0) * w_DE_a)
        )

    @contributes("pressure")
    def pressure_de(self, rho_de, w_de):
        return w_de * rho_de

    @contributes("delta_rho")
    def delta_rho_de(self, rho_de, delta_de):
        return rho_de * delta_de

    @contributes("momentum")
    def momentum_de(self, rho_de, w_de, theta_de):
        return rho_de * (1.0 + w_de) * theta_de

    def w_de_prime(self, w_DE_a, aH, a):
        return -w_DE_a * aH * a

    def ca2_de(self, w_de, w_de_prime, aH):
        return w_de - w_de_prime / (3.0 * (1.0 + w_de) * aH)

    @derivative("delta_de")
    def continuity_de(self, delta_de, theta_de, h_prime, k, aH, w_de, cs2_DE, ca2_de):
        inv_k2 = 1.0 / (k * k)
        return (
            -(1.0 + w_de) * (theta_de + 0.5 * h_prime)
            - 3.0 * (cs2_DE - w_de) * aH * delta_de
            - 9.0 * (1.0 + w_de) * (cs2_DE - ca2_de) * aH**2 * inv_k2 * theta_de
        )

    @derivative("theta_de")
    def euler_de(self, delta_de, theta_de, k, aH, w_de, cs2_DE):
        return (
            -(1.0 - 3.0 * cs2_DE) * aH * theta_de
            + cs2_DE / (1.0 + w_de) * k**2 * delta_de
        )
