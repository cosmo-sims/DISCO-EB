"""The species of the linear Einstein-Boltzmann system, as components.

Each species is a :mod:`discoeb.component` component that owns its
perturbation variables and states its own equations. A function names what
it needs in its signature, as in ``d_vb(adotoa, vb, k, cs2, clxb)``, and
:class:`~discoeb.component.Equations` injects it. A species sees another only
through the quantities it publishes. Couplings between species live in
:mod:`discoeb.interactions`, the metric and the Friedmann sum in
:mod:`discoeb.gravity`. Conventions are CAMB's (see :mod:`discoeb.gravity`).

The Boltzmann hierarchies are vector states: ``theta`` holds ``Theta_0 ..
Theta_lmax``, ``e`` holds ``E_2 .. E_lmax``, ``n`` the neutrino multipoles
and ``psi`` one row per massive-neutrino momentum bin. Their equations loop
over multipoles and share the free-streaming tail in :func:`stream`. The
truncations are component settings: ``LMAX_G`` and ``LMAX_POL`` on
:class:`Photons`, ``LMAX_NR`` on :class:`MasslessNeutrinos`, ``LMAX_NU`` and
``NQMAX`` on :class:`MassiveNeutrinos`. ``DISCO_<NAME>`` in the environment,
a model's settings or a constructor argument overrides each one, and the
equations read them as ``self.NQMAX`` like any attribute.

Each state carries its adiabatic initial condition (CAMB's ``initial()``
series at ``k tau << 1``, using the growing-mode quantities of
:class:`~discoeb.gravity.AdiabaticMode`). The photon monopole, dipole and
quadrupole are ``theta[0]``, ``theta[1]``, ``theta[2]``. The scalar names
``clxg``, ``qg``, ``pig`` and ``E2`` exist for the equations that read them.
"""

from __future__ import annotations

import math

from jax.typing import ArrayLike

from .component import (
    Component,
    Derived,
    Parameter,
    Setting,
    Species,
    State,
    contributes,
)
from .gravity import C2OK, GRHOG_PER_TCMB4, GRHOR_PER_TCMB4
from .util import neutrino_momentum_bins


def _needs_octupole(**lmax):
    """Raise ``ValueError`` unless every hierarchy reaches ``l = 3``.

    The quadrupole equations couple to the octupole.
    """

    low = {name: ell for name, ell in lmax.items() if ell < 3}
    if low:
        raise ValueError(f"a hierarchy must reach at least the octupole; got {low}")


# --- matter -------------------------------------------------------------------------
# Each density coefficient ``grho_i = 8 pi G rho_i a^2`` today is a directed
# parameter: the row holds a nominal value and a direction, the generated
# provider reads ``grho_i = nominal + eps * direction``, and modax's
# sensitivity in ``eps`` is the derivative along that direction.


class ColdDarkMatter(Species):
    """Cold dark matter, the species whose rest frame defines the gauge.

    Synchronous gauge leaves the choice of the initial hypersurface open.
    The equations of :mod:`discoeb.gravity` take the one on which this
    species is at rest, so it has a density contrast and no velocity
    (``fixes_gauge``). A model must keep it, at whatever density. Dark
    matter that feels a force has a velocity of its own and is another
    species beside it (:class:`Fluid`).
    """

    fixes_gauge = True

    # adiabatic mode: 3/4 of the photon density contrast
    clxc = State(initial=lambda theta: 0.75 * theta[0])

    Omegam: ArrayLike = Parameter()  # CDM and baryons
    grhoc: ArrayLike = Derived(
        lambda grhom, Omegam, Omegab: grhom * (Omegam - Omegab), directed=True
    )

    def grhoc_t(self, grhoc, inv_a):
        return grhoc * inv_a

    @contributes("grho")
    def grho_c(self, grhoc_t):
        return grhoc_t

    @contributes("grho_matter")
    def grho_matter_c(self, grhoc_t):
        return grhoc_t

    @contributes("dgrho")
    def dgrho_c(self, grhoc_t, clxc):
        return grhoc_t * clxc

    def d_clxc(self, z, k):
        """CDM is pressureless and at rest in synchronous gauge: ``clxc' = -k z``."""
        return -k * z


class Baryons(Species):
    """Baryon continuity and Euler equations, without the Thomson drag.

    :class:`ThomsonScattering` adds the drag.
    """

    clxb = State(initial=lambda theta: 0.75 * theta[0])
    vb = State(initial=lambda theta: 0.75 * theta[1])

    # The recombination rates read Omegab and the helium fraction YHe too.
    Omegab: ArrayLike = Parameter(directed=True)
    YHe: ArrayLike = Parameter(directed=True)
    grhob: ArrayLike = Derived(lambda grhom, Omegab: grhom * Omegab, directed=True)

    def grhob_t(self, grhob, inv_a):
        return grhob * inv_a

    @contributes("grho")
    def grho_b(self, grhob_t):
        return grhob_t

    @contributes("grho_matter")
    def grho_matter_b(self, grhob_t):
        return grhob_t

    @contributes("dgrho")
    def dgrho_b(self, grhob_t, clxb):
        return grhob_t * clxb

    @contributes("dgq")
    def dgq_b(self, grhob_t, vb):
        return grhob_t * vb

    def d_clxb(self, z, vb, k):
        return -k * (z + vb)

    def d_vb(self, adotoa, vb, k, cs2, clxb):
        return -adotoa * vb + k * cs2 * clxb


class Fluid(Species):
    """A perfect fluid with a constant equation of state ``w`` and rest-frame sound speed ``cs2``.

    The fluid equations of :class:`DarkEnergy` at constant ``w``, where the
    adiabatic sound speed is ``w`` itself:

    ``delta' = -(1+w)(theta + k z) - 3 (cs2 - w) aH delta
               - 9 (1+w)(cs2 - w) (aH)^2 theta / k^2``,
    ``theta' = -(1 - 3 cs2) aH theta + cs2 k^2 delta / (1+w)``,

    with no anisotropic stress. Its density today is ``Omega`` of the
    critical one and scales as ``a^(-3(1+w))``. It starts in the adiabatic
    mode, ``delta = (1 + w) clxc`` at rest. ``w = cs2 = 0`` is a second cold
    dark matter and ``w = cs2 = 1/3`` a dark radiation without shear. The
    growing mode's series does not count it: a subclass that is matter or
    radiation contributes its ``grho_t`` to ``grho_matter`` or
    ``grho_radiation``.

    Its names (``delta``, ``theta``, ``w``, ``cs2``) are generic, so a model
    lists it under a name of its own, any number of times::

        LCDM.add(Fluid.named("dr"), name="LCDM+dr")   # dr_Omega, dr_w, dr_delta ...
    """

    delta = State(initial=lambda w, clxc: (1.0 + w) * clxc)
    theta = State()

    Omega: ArrayLike = Parameter(0.0)
    w: ArrayLike = Parameter(0.0, column=True)
    cs2: ArrayLike = Parameter(0.0, column=True)
    grho0: ArrayLike = Derived(lambda grhom, Omega: grhom * Omega, directed=True)

    @classmethod
    def check(cls, Omega, w):
        """The Euler equation divides by ``1 + w``."""
        if float(w) == -1.0:
            raise ValueError("a fluid needs w != -1; use a cosmological constant")
        if float(Omega) < 0.0:
            raise ValueError(f"a fluid's density Omega = {Omega} must not be negative")

    def grho_t(self, grho0, a, w):
        return grho0 * a ** (-1.0 - 3.0 * w)

    @contributes("grho")
    def grho_fluid(self, grho_t):
        return grho_t

    @contributes("gpres")
    def gpres_fluid(self, grho_t, w):
        return w * grho_t

    @contributes("dgrho")
    def dgrho_fluid(self, grho_t, delta):
        return grho_t * delta

    @contributes("dgq")
    def dgq_fluid(self, grho_t, w, theta, k):
        return grho_t * (1.0 + w) * theta / k

    def d_delta(self, delta, theta, z, k, adotoa, w, cs2, inv_k2):
        return (
            -(1.0 + w) * (theta + k * z)
            - 3.0 * (cs2 - w) * adotoa * delta
            - 9.0 * (1.0 + w) * (cs2 - w) * adotoa**2 * inv_k2 * theta
        )

    def d_theta(self, delta, theta, k, adotoa, w, cs2):
        return -(1.0 - 3.0 * cs2) * adotoa * theta + cs2 / (1.0 + w) * k**2 * delta


# --- the radiation streaming approximation ------------------------------------------


class RadiationStreaming(Component):
    """The radiation streaming approximation of CLASS (Blas, Lesgourgues & Tram 2011).

    Well inside the horizon, once the photons decouple, the photon and
    massless-neutrino hierarchies oscillate at ``k`` about a particular
    solution driven by the metric, and the oscillation decays as
    ``1/(k tau)``. Where ``rsa`` is 1 the model swaps the hierarchies for
    that solution,

        delta = 4 (aH h' - k^2 eta) / k^2,   theta = -h'/2,   no shear, no polarization,

    (synchronous gauge, ``h' = 2 k z``, ``k eta = etak``), takes ``h'``
    without the streaming species as CLASS does, and freezes their states.
    The switch is CLASS's: ``k tau > RSA_K_TAU`` (45) and ``kappa' tau <
    RSA_KAPPA_TAU`` (``tau_c / tau > 5``). Like CLASS, which switches at a
    fixed ``tau_free_streaming``, it only turns on: past ``a = RSA_A_LATCH``
    it stays on through reionization. CLASS's Thomson corrections for
    reionization are missing. ``RSA = 0`` (``DISCO_RSA=0``) turns it off.

    It provides the switch ``rsa`` and the streaming solution ``rsa_delta``
    and ``rsa_q``, and names no species. Each species that follows it reads
    those and adds its own density to the sum ``dgrho_streaming``. The
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

    def rsa_delta(self, adotoa, z, etak, k):
        """The streaming radiation's density contrast, ``4 (aH h' - k^2 eta) / k^2``."""
        return (8.0 * adotoa * z - 4.0 * etak) / k

    def rsa_q(self, z):
        """The streaming radiation's ``q = 4 theta / (3k)`` with ``theta = -h'/2 = -k z``."""
        return -(4.0 / 3.0) * z


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

    def rsa_q(self):
        return 0.0


# --- radiation ----------------------------------------------------------------------


def stream(k, kv, z, sigma, dlnf0, x, inv_tau, output):
    """Free streaming of a hierarchy ``x`` (``x_0 .. x_L``), into ``output``.

    The lowest three multipoles carry the metric sources (CLASS
    ``perturbations.c``, synchronous gauge, ``h' = 2 k z``)::

        x_0' = -kv x_1 + (k z / 3) dlnf0
        x_1' = (kv / 3) (x_0 - 2 x_2)
        x_2' = (kv / 5) (2 x_1 - 3 x_3) - (2/15) k sigma dlnf0

    and above them ``x_l' = kv (l x_{l-1} - (l+1) x_{l+1}) / (2l+1)``,
    closing at ``L = len(x) - 1`` with ``x_L' = kv x_{L-1} - (L+1) x_L / tau``.
    ``kv`` is ``k v`` and ``dlnf0`` the log-derivative of the background
    distribution: ``k`` and ``-4`` for massless particles, the bin's own for a
    massive-neutrino momentum bin. The Thomson interaction adds collisional
    damping to the same derivatives.
    """

    output[0] = -kv * x[1] + k * z * (1.0 / 3.0) * dlnf0
    output[1] = kv * (1.0 / 3.0) * (x[0] - 2.0 * x[2])
    output[2] = (
        kv * (1.0 / 5.0) * (2.0 * x[1] - 3.0 * x[3]) - (2.0 / 15.0) * k * sigma * dlnf0
    )
    lmax = len(x) - 1
    for ell in range(3, lmax):
        output[ell] = (
            kv * (ell * x[ell - 1] - (ell + 1) * x[ell + 1]) * (1.0 / (2 * ell + 1))
        )
    output[lmax] = kv * x[lmax - 1] - (lmax + 1) * x[lmax] * inv_tau


class Photons(Species):
    """Photon temperature multipoles ``theta`` (``Theta_0 .. Theta_LMAX_G``) and E-modes ``e``.

    ``e[l - 2]`` is ``E_l`` for ``l = 2 .. LMAX_POL``. This is the
    collisionless part of the hierarchy. :class:`ThomsonScattering` damps the
    multipoles and sources the polarization. The E-mode hierarchy couples
    upwards with ``(l+3)(l-1)/(l+1)`` in place of ``l+1`` and closes with
    ``k L E_{L-1} / (2L+1) - (L+3) E_L / tau``. Both hierarchies stream
    freely above their lowest three (``theta``) or one (``e``) multipoles.
    The E-modes start at zero.
    """

    LMAX_G = Setting(15)
    LMAX_POL = Setting(15)

    def _theta_initial(self, x, omtau, output):
        """The monopole ``delta_gamma`` and dipole ``q_gamma`` of the growing mode."""
        output[0] = x * x / 3.0 * (1.0 - omtau / 5.0)
        output[1] = x**3 / 27.0 * (1.0 - omtau / 5.0)

    theta = State(lambda self: self.LMAX_G + 1, initial=_theta_initial)
    e = State(lambda self: self.LMAX_POL - 1)

    Tcmb: ArrayLike = Parameter(directed=True)  # K, read by the recombination rates too
    grhog: ArrayLike = Derived(lambda Tcmb: GRHOG_PER_TCMB4 * Tcmb**4, directed=True)

    def __init__(self, **settings):
        super().__init__(**settings)
        _needs_octupole(LMAX_G=self.LMAX_G, LMAX_POL=self.LMAX_POL)

    def grhog_t(self, grhog, inv_a2):
        return grhog * inv_a2

    @contributes("grho")
    def grho_g(self, grhog_t):
        return grhog_t

    @contributes("grho_radiation")
    def grho_radiation_g(self, grhog_t):
        return grhog_t

    def clxg(self, theta, rsa, rsa_delta):
        return theta[0] + rsa * (rsa_delta - theta[0])

    def qg(self, theta, rsa, rsa_q):
        return theta[1] + rsa * (rsa_q - theta[1])

    def pig(self, theta, rsa):
        return (1.0 - rsa) * theta[2]

    def E2(self, e, rsa):
        """The E-mode quadrupole, ``e[0]`` (none under the RSA)."""
        return (1.0 - rsa) * e[0]

    @contributes("dgrho")
    def dgrho_g(self, grhog_t, theta, rsa):
        """The hierarchy's density, zero under the RSA.

        As in CLASS, the ``h'`` that the RSA density reads leaves the
        streaming species out.
        """
        return grhog_t * (1.0 - rsa) * theta[0]

    @contributes("dgq")
    def dgq_g(self, grhog_t, qg):
        return grhog_t * qg

    @contributes("dgrho_streaming")
    def dgrho_streaming_g(self, rsa, rsa_delta, grhog_t):
        """The photons' density under the RSA.

        ``dgrho`` (and so ``h'``) leaves it out, but as in CLASS everything
        else that reads the density includes it.
        """
        return rsa * rsa_delta * grhog_t

    @contributes("gpres")
    def gpres_g(self, grhog_t):
        return grhog_t * (1.0 / 3.0)

    @contributes("dgpi")
    def dgpi_g(self, grhog_t, pig):
        return grhog_t * pig

    def d_theta(self, k, z, sigma, theta, inv_tau, rsa, output):
        stream(k, k, z, sigma, -4.0, theta, inv_tau, output)
        for ell in range(len(theta)):
            output[ell] = (1.0 - rsa) * output[ell]

    def d_e(self, k, e, inv_tau, rsa, output):
        lmax = len(e) + 1
        output[0] = -k * e[1] * (1.0 / 3.0)
        for ell in range(3, lmax):
            upfac = (ell + 3.0) * (ell - 1.0) / (ell + 1.0)
            output[ell - 2] = (
                k * (ell * e[ell - 3] - upfac * e[ell - 1]) * (1.0 / (2 * ell + 1))
            )
        output[lmax - 2] = (
            k * lmax * e[lmax - 3] * (1.0 / (2 * lmax + 1))
            - (lmax + 3) * e[lmax - 2] * inv_tau
        )
        for ell in range(len(e)):
            output[ell] = (1.0 - rsa) * output[ell]


class MasslessNeutrinos(Species):
    """Massless neutrinos ``n`` (``N_0 .. N_LMAX_NR``): collisionless free streaming."""

    LMAX_NR = Setting(15)

    def _n_initial(self, x, Rv, Rp15, omtau, theta, output):
        """The growing mode's neutrino monopole, dipole, quadrupole and octupole."""
        output[0] = theta[0]
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

    n = State(lambda self: self.LMAX_NR + 1, initial=_n_initial)

    Neff: ArrayLike = Parameter()
    grhornomass: ArrayLike = Derived(
        lambda Neff, Tcmb: Neff * (GRHOR_PER_TCMB4 * Tcmb**4), directed=True
    )

    def __init__(self, **settings):
        super().__init__(**settings)
        _needs_octupole(LMAX_NR=self.LMAX_NR)

    def grhor_t(self, grhornomass, inv_a2):
        return grhornomass * inv_a2

    @contributes("grho")
    def grho_r(self, grhor_t):
        return grhor_t

    @contributes("grho_radiation")
    def grho_radiation_r(self, grhor_t):
        return grhor_t

    @contributes("grho_free_streaming")
    def grho_free_streaming_r(self, grhor_t):
        return grhor_t

    @contributes("dgrho")
    def dgrho_r(self, grhor_t, n, rsa):
        return grhor_t * (1.0 - rsa) * n[0]

    @contributes("dgq")
    def dgq_r(self, grhor_t, n, rsa, rsa_q):
        return grhor_t * (n[1] + rsa * (rsa_q - n[1]))

    @contributes("dgrho_streaming")
    def dgrho_streaming_r(self, rsa, rsa_delta, grhor_t):
        return rsa * rsa_delta * grhor_t

    @contributes("gpres")
    def gpres_r(self, grhor_t):
        return grhor_t * (1.0 / 3.0)

    @contributes("dgpi")
    def dgpi_r(self, grhor_t, n, rsa):
        return grhor_t * (1.0 - rsa) * n[2]

    def d_n(self, k, z, sigma, n, inv_tau, rsa, output):
        stream(k, k, z, sigma, -4.0, n, inv_tau, output)
        for ell in range(len(n)):
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

    def _psi_initial(self, n, output):
        """Still relativistic: each bin starts as the massless hierarchy scaled by dlnf0/dlnq."""
        for i in range(self.NQMAX):
            for ell in range(4):
                output[i, ell] = -0.25 * n[ell] * self.dlnf0[i]

    psi = State(lambda self: (self.NQMAX, self.LMAX_NU + 1), initial=_psi_initial)

    Nmnu: ArrayLike = Parameter(0.0)  # flavours
    mnu: ArrayLike = Parameter(0.0)  # eV, each
    grhor_nu: ArrayLike = Derived(lambda Tcmb: GRHOR_PER_TCMB4 * Tcmb**4, directed=True)
    n_mnu: ArrayLike = Derived(lambda Nmnu: Nmnu, column=True)
    amnu: ArrayLike = Derived(lambda mnu, Tcmb: mnu * C2OK / Tcmb, directed=True)

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

    def nu_norm(self, grhor_nu, n_mnu, inv_a2):
        return grhor_nu * n_mnu * inv_a2

    @contributes("grho")
    def grho_nu(self, nu_norm, rhonu):
        return nu_norm * rhonu

    @contributes("dgrho")
    def dgrho_nu(self, nu_norm, a, amnu, psi):
        total = 0.0
        for i in range(self.NQMAX):
            inv_v = math.sqrt(1.0 + (a * amnu * self.inv_q[i]) ** 2)
            total += self.weight[i] * psi[i, 0] * inv_v
        return nu_norm * total

    @contributes("dgq")
    def dgq_nu(self, nu_norm, psi):
        total = 0.0
        for i in range(self.NQMAX):
            total += self.weight[i] * psi[i, 1]
        return nu_norm * total

    @contributes("gpres")
    def gpres_nu(self, nu_norm, pnu):
        """The background pressure, from ``pnu = (rhonu - d rhonu / d ln a) / 3``."""
        return nu_norm * pnu

    @contributes("dgpi")
    def dgpi_nu(self, nu_norm, a, amnu, psi):
        """The shear ``Sum w v psi2``: the pressure weighting ``q/epsilon`` of a massive bin."""
        total = 0.0
        for i in range(self.NQMAX):
            inv_v = math.sqrt(1.0 + (a * amnu * self.inv_q[i]) ** 2)
            total += self.weight[i] * psi[i, 2] / inv_v
        return nu_norm * total

    def d_psi(self, k, z, sigma, a, amnu, psi, inv_tau, output):
        for i in range(self.NQMAX):
            inv_v = math.sqrt(1.0 + (a * amnu * self.inv_q[i]) ** 2)
            kv = k * (1.0 / inv_v)
            stream(k, kv, z, sigma, self.dlnf0[i], psi[i], inv_tau, output[i])


# --- dark energy --------------------------------------------------------------------


class _DarkEnergyDensity(Species):
    """The dark-energy density today and its part of the Friedmann sum, for either kind.

    Its density today closes the budget: the background sets it so that the
    densities sum to the critical one.
    """

    grhov: ArrayLike = Derived(directed=True, closes_budget=True)

    @contributes("grho")
    def grho_de(self, grhov_t):
        return grhov_t


class CosmologicalConstant(_DarkEnergyDensity):
    """A cosmological constant: density only, no perturbations."""

    def grhov_t(self, grhov, a2):
        return grhov * a2

    @contributes("gpres")
    def gpres_lambda(self, grhov_t):
        return -grhov_t


class DarkEnergy(_DarkEnergyDensity):
    """A CPL fluid ``w = w_0 + w_a (1 - a)`` with rest-frame sound speed ``cs2_DE``.

    Synchronous-gauge fluid equations after Ballesteros & Lesgourgues 2010, with
    the adiabatic sound speed ``c_a^2 = w - w' / (3 (1+w) aH)``, ``w' = -w_a aH a``:

    ``clxq' = -(1+w)(thetaq + k z) - 3 (cs2 - w) aH clxq
              - 9 (1+w)(cs2 - c_a^2) (aH)^2 thetaq / k^2``,
    ``thetaq' = -(1 - 3 cs2) aH thetaq + cs2 k^2 clxq / (1+w)``.

    Valid for ``w != -1``. Use :class:`CosmologicalConstant` for ``w = -1``.
    The fluid starts in the adiabatic mode, ``clxq = (1 + w_0) clxc`` at rest.
    """

    # adiabatic, at rest
    clxq = State(initial=lambda w_DE_0, clxc: (1.0 + w_DE_0) * clxc)
    thetaq = State()

    w_DE_0: ArrayLike = Parameter(-1.0, directed=True)
    w_DE_a: ArrayLike = Parameter(0.0, directed=True)
    cs2_DE: ArrayLike = Parameter(1.0, directed=True)

    @classmethod
    def check(cls, w_DE_0, w_DE_a):
        """The fluid equations divide by ``1 + w``."""
        if float(w_DE_0) == -1.0 and float(w_DE_a) == 0.0:
            raise ValueError(
                "it evolves a dark-energy fluid, which needs w != -1; use a model "
                "with a cosmological constant for w_DE_0 = -1, w_DE_a = 0"
            )

    def w_Q(self, a, w_DE_0, w_DE_a):
        return w_DE_0 + w_DE_a * (1.0 - a)

    def rho_Q(self, a, w_DE_0, w_DE_a):
        """``rho_Q(a) / rho_Q(1)`` for the CPL equation of state."""
        return a ** (-3.0 * (1.0 + w_DE_0 + w_DE_a)) * math.exp(
            3.0 * (a - 1.0) * w_DE_a
        )

    def grhov_t(self, grhov, rho_Q, a2):
        return grhov * rho_Q * a2

    @contributes("gpres")
    def gpres_de(self, grhov_t, w_Q):
        return w_Q * grhov_t

    @contributes("dgrho")
    def dgrho_de(self, grhov_t, clxq):
        return grhov_t * clxq

    @contributes("dgq")
    def dgq_de(self, grhov_t, w_Q, thetaq, k):
        return grhov_t * (1.0 + w_Q) * thetaq / k

    def w_Q_prime(self, w_DE_a, adotoa, a):
        return -w_DE_a * adotoa * a

    def ca2_Q(self, w_Q, w_Q_prime, adotoa):
        return w_Q - w_Q_prime / (3.0 * (1.0 + w_Q) * adotoa)

    def d_clxq(self, clxq, thetaq, z, k, adotoa, w_Q, cs2_DE, ca2_Q, inv_k2):
        return (
            -(1.0 + w_Q) * (thetaq + k * z)
            - 3.0 * (cs2_DE - w_Q) * adotoa * clxq
            - 9.0 * (1.0 + w_Q) * (cs2_DE - ca2_Q) * adotoa**2 * inv_k2 * thetaq
        )

    def d_thetaq(self, clxq, thetaq, k, adotoa, w_Q, cs2_DE):
        return (
            -(1.0 - 3.0 * cs2_DE) * adotoa * thetaq + cs2_DE / (1.0 + w_Q) * k**2 * clxq
        )
