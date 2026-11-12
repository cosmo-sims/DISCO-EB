"""The linear Einstein-Boltzmann system in synchronous gauge, as components.

Each species is a :mod:`discoeb.component` component that owns its
perturbation variables and states its own equations. The Einstein
constraints, the Friedmann equation and the Thomson coupling are components
too. A function names what it needs in its signature, as in
``d_vb(adotoa, vb, k, cs2, clxb)``, and :class:`~discoeb.component.Model`
injects it. A species sees another only through the quantities it publishes.

Conventions follow CAMB: ``k`` in 1/Mpc, conformal time ``tau`` in Mpc,
``etak = k eta``, velocities ``q_i = (1 + w_i) theta_i / k`` and density
coefficients ``grho_i_t = 8 pi G rho_i a^2``. The metric enters the fluid
equations through ``0.5 h' = k z`` and the shear ``sigma`` (see
:class:`Metric`). The curved-geometry relations follow CLASS's
``perturbations.c`` with ``K = -grhok / 3`` and ``s2^2 = 1 - 3K/k^2``.

The Boltzmann hierarchies are vector states: ``theta`` holds ``Theta_0 ..
Theta_lmax``, ``e`` holds ``E_2 .. E_lmax``, ``n`` the neutrino multipoles
and ``psi`` one row per massive-neutrino momentum bin. Their equations loop
over multipoles and share the free-streaming tail in :func:`stream`. The
truncations are component settings: ``LMAX_G`` and ``LMAX_POL`` on
:class:`Photons`, ``LMAX_NR`` on :class:`MasslessNeutrinos`, ``LMAX_NU`` and
``NQMAX`` on :class:`MassiveNeutrinos`. ``DISCO_<NAME>`` in the environment
or a constructor argument overrides each one, and the equations read them
as ``self.NQMAX`` like any attribute.

Each state carries its adiabatic initial condition (CAMB's ``initial()``
series at ``k tau << 1``, using the growing-mode quantities of
:class:`AdiabaticMode`). :meth:`Model.sparsity` traces the Jacobian's
sparsity pattern through the equations. So the layout, the initial state
and the sparse solver all come from the components. The photon monopole,
dipole and quadrupole are ``theta[0]``, ``theta[1]``, ``theta[2]``. The
scalar names ``clxg``, ``qg``, ``pig`` and ``E2`` exist for the equations
that read them.

The background uses the same species. :meth:`Model.evaluate` computes the
kernel's Friedmann sum ``grho`` on JAX arrays at ``a``, which gives the
expansion rate the conformal-time quadrature and the thermal history
integrate (:func:`expansion_rate`, :func:`dtauda`).
:func:`density_parameters` closes the density budget at ``a = 1`` with the
same sum.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np

from .component import Component, Interaction, Model, Setting, Species, State
from .spline_interpolation import FastUniformCubicSpline1D
from .util import generalized_gauss_laguerre_weights

SUMS = ("grho", "gpres", "dgrho", "dgq", "dgpi")
"""The quantities summed over species.

``grho`` is the Friedmann density and ``gpres = Sum 8 pi G p_i a^2`` the
pressure (CAMB's, so the comoving curvature perturbation can divide by
``rho + p``). ``dgrho = Sum grho_i_t delta_i`` and ``dgq = Sum grho_i_t q_i``
source the Einstein constraints. ``dgpi = Sum grho_i_t pi_i`` is the
anisotropic stress the line-of-sight sources' Weyl potential needs. Spatial
curvature adds ``grhok`` to ``grho`` but no pressure, since it is no fluid."""


def _needs_octupole(**lmax):
    """Raise ``ValueError`` unless every hierarchy reaches ``l = 3``.

    The quadrupole equations couple to the octupole.
    """

    low = {name: ell for name, ell in lmax.items() if ell < 3}
    if low:
        raise ValueError(f"a hierarchy must reach at least the octupole; got {low}")


# --- scales, the Friedmann equation and the metric ----------------------------------


class Background(Component):
    """Powers of the scale factor, the Friedmann equation, ``1/tau`` and ``1/k^2``."""

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
    """

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


# --- matter -------------------------------------------------------------------------
# The row holds each density coefficient ``grho_i = 8 pi G rho_i a^2`` today as a
# nominal value plus ``eps`` times its direction, so modax's sensitivity in
# ``eps`` is the derivative along that direction.


class ColdDarkMatter(Species):
    # adiabatic mode: 3/4 of the photon density contrast
    clxc = State(initial=lambda theta: 0.75 * theta[0])

    def grhoc(self, grhoc_nominal, eps, dir_grhoc):
        return grhoc_nominal + eps * dir_grhoc

    def grhoc_t(self, grhoc, inv_a):
        return grhoc * inv_a

    def grho(self, grhoc_t):
        return grhoc_t

    def dgrho(self, grhoc_t, clxc):
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

    def grhob(self, grhob_nominal, eps, dir_grhob):
        return grhob_nominal + eps * dir_grhob

    def grhob_t(self, grhob, inv_a):
        return grhob * inv_a

    def grho(self, grhob_t):
        return grhob_t

    def dgrho(self, grhob_t, clxb):
        return grhob_t * clxb

    def dgq(self, grhob_t, vb):
        return grhob_t * vb

    def d_clxb(self, z, vb, k):
        return -k * (z + vb)

    def d_vb(self, adotoa, vb, k, cs2, clxb):
        return -adotoa * vb + k * cs2 * clxb


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

    def dgrho_streaming(self, rsa, rsa_delta, grhog_t, grhor_t):
        """The streaming photons' and neutrinos' density under the RSA.

        ``dgrho`` (and so ``h'``) leaves it out, but as in CLASS everything
        else that reads the density includes it.
        """
        return rsa * rsa_delta * (grhog_t + grhor_t)


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

    def __init__(self, **settings):
        super().__init__(**settings)
        _needs_octupole(LMAX_G=self.LMAX_G, LMAX_POL=self.LMAX_POL)

    def grhog(self, grhog_nominal, eps, dir_grhog):
        return grhog_nominal + eps * dir_grhog

    def grhog_t(self, grhog, inv_a2):
        return grhog * inv_a2

    def grho(self, grhog_t):
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

    def dgrho(self, grhog_t, theta, rsa):
        """The hierarchy's density, zero under the RSA.

        As in CLASS, the ``h'`` that the RSA density reads leaves the
        streaming species out.
        """
        return grhog_t * (1.0 - rsa) * theta[0]

    def dgq(self, grhog_t, qg):
        return grhog_t * qg

    def gpres(self, grhog_t):
        return grhog_t * (1.0 / 3.0)

    def dgpi(self, grhog_t, pig):
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

    def __init__(self, **settings):
        super().__init__(**settings)
        _needs_octupole(LMAX_NR=self.LMAX_NR)

    def grhornomass(self, grhornomass_nominal, eps, dir_grhornomass):
        return grhornomass_nominal + eps * dir_grhornomass

    def grhor_t(self, grhornomass, inv_a2):
        return grhornomass * inv_a2

    def grho(self, grhor_t):
        return grhor_t

    def dgrho(self, grhor_t, n, rsa):
        return grhor_t * (1.0 - rsa) * n[0]

    def dgq(self, grhor_t, n, rsa, rsa_q):
        return grhor_t * (n[1] + rsa * (rsa_q - n[1]))

    def gpres(self, grhor_t):
        return grhor_t * (1.0 / 3.0)

    def dgpi(self, grhor_t, n, rsa):
        return grhor_t * (1.0 - rsa) * n[2]

    def d_n(self, k, z, sigma, n, inv_tau, rsa, output):
        stream(k, k, z, sigma, -4.0, n, inv_tau, output)
        for ell in range(len(n)):
            output[ell] = (1.0 - rsa) * output[ell]


class MassiveNeutrinos(Species):
    """Massive neutrinos on ``NQMAX`` momentum bins: ``psi[i, l]`` (Ma & Bertschinger).

    Bin ``i`` has node ``q_i``, weight ``w_i`` and ``dlnf0/dlnq``, and
    streams freely (:func:`stream`) with ``kv = k v_i``,
    ``v_i = q / sqrt(q^2 + a^2 amnu^2)``. The moments ``Sum w psi0 / v``,
    ``Sum w psi1`` and ``Sum w v psi2`` enter the Einstein sources. The
    recombination block provides the background density ratio ``rhonu`` and
    pressure ratio ``pnu``. Deep in radiation domination the neutrinos are
    still relativistic, so each bin starts as the massless hierarchy,
    ``psi_l = -(1/4) N_l dlnf0/dlnq``. The quadrature nodes and weights are
    instance constants the providers read through ``self``. Each bin adds
    ``LMAX_NU + 1`` states. 3 bins give the default layout ``nvar = 98``.
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

    def __init__(self, **settings):
        super().__init__(**settings)
        _needs_octupole(LMAX_NU=self.LMAX_NU)
        q, w, dlnf0 = neutrino_momentum_bins(self.NQMAX)
        self.inv_q = tuple(1.0 / float(x) for x in q)
        self.weight = tuple(float(x) for x in w)
        self.dlnf0 = tuple(float(x) for x in dlnf0)

    def nu_norm(self, grhor_nu, n_mnu, inv_a2):
        return grhor_nu * n_mnu * inv_a2

    def grho(self, nu_norm, rhonu):
        return nu_norm * rhonu

    def dgrho(self, nu_norm, a, amnu, psi):
        total = 0.0
        for i in range(self.NQMAX):
            inv_v = math.sqrt(1.0 + (a * amnu * self.inv_q[i]) ** 2)
            total += self.weight[i] * psi[i, 0] * inv_v
        return nu_norm * total

    def dgq(self, nu_norm, psi):
        total = 0.0
        for i in range(self.NQMAX):
            total += self.weight[i] * psi[i, 1]
        return nu_norm * total

    def gpres(self, nu_norm, pnu):
        """The background pressure, from the recombination block's ``pnu = (rhonu - d rhonu / d ln a) / 3``."""
        return nu_norm * pnu

    def dgpi(self, nu_norm, a, amnu, psi):
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
    """The dark-energy density today and its part of the Friedmann sum, for either kind."""

    def grhov(self, grhov_nominal, eps, dir_grhov):
        return grhov_nominal + eps * dir_grhov

    def grho(self, grhov_t):
        return grhov_t


class CosmologicalConstant(_DarkEnergyDensity):
    """A cosmological constant: density only, no perturbations."""

    def grhov_t(self, grhov, a2):
        return grhov * a2

    def gpres(self, grhov_t):
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

    def w_Q(self, a, w_DE_0, w_DE_a):
        return w_DE_0 + w_DE_a * (1.0 - a)

    def rho_Q(self, a, w_DE_0, w_DE_a):
        """``rho_Q(a) / rho_Q(1)`` for the CPL equation of state."""
        return a ** (-3.0 * (1.0 + w_DE_0 + w_DE_a)) * math.exp(
            3.0 * (a - 1.0) * w_DE_a
        )

    def grhov_t(self, grhov, rho_Q, a2):
        return grhov * rho_Q * a2

    def gpres(self, grhov_t, w_Q):
        return w_Q * grhov_t

    def dgrho(self, grhov_t, clxq):
        return grhov_t * clxq

    def dgq(self, grhov_t, w_Q, thetaq, k):
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


# --- the model ----------------------------------------------------------------------


def perturbation_model(
    layout, thermal_history: Component | None = None, extra=()
) -> Model:
    """The components a :class:`~discoeb.perturbations.PerturbationLayout` enables.

    ``thermal_history`` provides ``a``, ``opacity``, ``cs2``, ``rhonu`` and
    ``pnu`` as functions of ``tau``. In the solve it is the recombination
    block of :func:`discoeb.perturbations.joint_kernel`, in a test anything.
    Without it the model still gives the state layout, the initial conditions
    and the sparsity pattern. The model appends the ``extra`` components (the
    line-of-sight sources of :mod:`discocmb.cmb`). The components take the
    layout's truncations as settings, so a layout built from the components'
    own (environment-loaded) settings comes back unchanged. The state order
    matches the layout's indices: metric, CDM, baryons, photon temperature
    and polarization, massless neutrinos, then the dark-energy fluid and the
    massive-neutrino bins when enabled.
    """

    components = [
        Background(),
        RadiationStreaming(),
        AdiabaticMode(),
        Curvature(),
        Metric(),
        ColdDarkMatter(),
        Baryons(),
        Photons(LMAX_G=layout.lmaxg, LMAX_POL=layout.lmaxpol),
        MasslessNeutrinos(LMAX_NR=layout.lmaxr),
    ]
    if thermal_history is not None:
        components.insert(0, thermal_history)
    if layout.enable_dark_energy:
        components.append(DarkEnergy())
    else:
        components.append(CosmologicalConstant())
    if layout.has_massive_neutrinos:
        components.append(MassiveNeutrinos(LMAX_NU=layout.lmaxnu, NQMAX=layout.nqmax))
    components.append(ThomsonScattering())
    components.extend(extra)
    return Model(components, sums=SUMS)


def perturbation_sparsity(layout) -> np.ndarray:
    """The ``(nnz, 2)`` index array of the Jacobian's nonzeros for ``layout``.

    modax's Rodas5P kernel uses the pattern twice. It colours the column
    intersection graph, so a Jacobian costs ``n_colours + 1`` Enzyme sweeps
    rather than ``nvar + 1``. And ``modax._sparse_direct`` orders it,
    factorises it symbolically and compiles a sparse LU and a pair of sparse
    triangular solves for that exact structure. :meth:`Model.sparsity`
    traces the pattern through the equations, resolving the providers on
    values that carry the state elements they depend on. So the tridiagonal
    free-streaming tails of :func:`stream` and the Einstein constraints that
    tie the metric, the fluids and the lowest multipoles together come
    straight from the code that writes them. The pattern is exact: the
    superset a colouring needs (a subset would silently corrupt entries
    where two columns of a colour group overlap) and no more. At the default
    layout it has 165 entries, 11 colours and 198 slots of ``L + U`` after
    AMD's fill-in (``tests/test_eb_sparsity.py``).

    The kernel factorises ``I/(h gamma) - J``. modax keeps a slot for every
    diagonal in its packed layout and resolves ``slot(i, i)`` through that
    table even where ``J`` has no entry (an undamped free-streaming
    multipole has none). This needs modax at or after commit 9f54d12
    (2026-11-01). Older versions put ``1/(h gamma)`` on another column's
    entry of the same colour.
    """

    return perturbation_model(layout).sparsity()


@functools.lru_cache(maxsize=None)
def layout_state_names(layout) -> tuple[str, ...]:
    """The state names of :func:`perturbation_model` in the layout's index order."""

    return tuple(perturbation_model(layout).state)


# --- the expansion history ----------------------------------------------------------
# Density coefficients in CAMB's convention, ``8 pi G rho / c^2`` in 1/Mpc^2:
# ``H^2 = sum_i grho_i / 3`` gives the Hubble rate in 1/Mpc and ``a'/a = a H``
# the conformal one. The perturbation kernel's ``grho_i_t = 8 pi G rho_i a^2``
# are these times ``a^2``.

GRHOM_PER_H0SQ = 3.33795017e-11  # 8 pi G rho_crit / c^2 per (H0 in km/s/Mpc)^2
GRHOG_PER_TCMB4 = 1.49594245e-13  # photons, per K^4
GRHOR_PER_TCMB4 = 3.39739477e-14  # one massless neutrino flavour, per K^4
C2OK = 1.62581581e4  # K / eV: amnu = m_nu c^2 / (k_B T_nu0) = mnu * C2OK / Tcmb

NQ_NEUTRINO_TABLE = 100
"""Momentum nodes for the massive-neutrino background density. The ``rho_nu``
table and the budget closing both integrate the Fermi-Dirac distribution on
these."""


def _neutrino_momentum_bins(nqmax: int) -> tuple[jax.Array, jax.Array]:
    fermi_dirac_const = 7 * jnp.pi**4 / 120
    if nqmax == 3:
        q = jnp.array([0.913201, 3.37517, 7.79184])
        dlfdlq = -q / (1 + jnp.exp(-q))
        w = jnp.array([0.0687359, 3.31435, 2.29911]) / (-0.25 * dlfdlq)
    elif nqmax == 4:
        q = jnp.array([0.7, 2.62814, 5.90428, 12.0])
        dlfdlq = -q / (1 + jnp.exp(-q))
        w = jnp.array([0.0200251, 1.84539, 3.52736, 0.289427]) / (-0.25 * dlfdlq)
    elif nqmax == 5:
        q = jnp.array([0.583165, 2.0, 4.0, 7.26582, 13.0])
        dlfdlq = -q / (1 + jnp.exp(-q))
        w = jnp.array([0.0081201, 0.689407, 2.8063, 2.05156, 0.12681]) / (
            -0.25 * dlfdlq
        )
    else:
        alpha = 1
        q, w = generalized_gauss_laguerre_weights(nqmax, alpha)
        w *= q**3 / (1 + jnp.exp(-q)) * q**-alpha
    return q, w / fermi_dirac_const


def neutrino_momentum_bins(nqmax: int):
    """Return ``(q, w, dlnf0_dlnq)`` momentum bins as read-only NumPy arrays.

    ``q`` are the nodes (in units of ``k_B T_nu0 / c``) and ``w`` the weights,
    normalised so one massless flavour integrates to unit density. For
    ``nqmax`` of 3, 4 or 5 they are CAMB's high-accuracy formulas, for higher
    counts a generalised Gauss-Laguerre rule. ``dlnf0/dlnq = -q/(1 +
    exp(-q))`` is the log-derivative of the Fermi-Dirac background
    distribution that sources the ``psi`` hierarchy. Cached per ``nqmax`` and
    precision, because the rule's eigendecomposition, op by op, cost a second
    of compiles per call.
    """

    return _momentum_bins(int(nqmax), bool(jax.config.read("jax_enable_x64")))


@functools.lru_cache(maxsize=None)
def _momentum_bins(nqmax: int, x64: bool):
    with jax.ensure_compile_time_eval():
        q, w = _neutrino_momentum_bins(nqmax)
    q = np.asarray(q, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    dlfdlq = -q / (1.0 + np.exp(-q))
    for array in (q, w, dlfdlq):
        array.flags.writeable = False
    return q, w, dlfdlq


# The massive-neutrino density relative to the massless flavour depends only on
# ``y = amnu * a``, so one table over ``ln y`` serves every cosmology. Below the
# table the neutrinos are relativistic and the ratio is 1, which also covers
# ``amnu = 0``.
_LN_Y_MIN, _LN_Y_MAX = math.log(1.0e-6), math.log(1.0e4)
_NEUTRINO_TABLE: dict = {}


def _neutrino_table(n_y_grid: int = 1000):
    """The spline of ``ln(rho_nu / rho_nu,massless)`` over ``ln y``.

    Built on first use rather than at import, so it picks up the
    ``jax_enable_x64`` setting in force then. It stays concrete even when
    first used inside a ``jit``.
    """

    key = (n_y_grid, bool(jax.config.read("jax_enable_x64")))
    table = _NEUTRINO_TABLE.get(key)
    if table is None:
        with jax.ensure_compile_time_eval():
            log_y = jnp.linspace(_LN_Y_MIN, _LN_Y_MAX, n_y_grid)
            q, w, _ = (
                jnp.asarray(v) for v in neutrino_momentum_bins(NQ_NEUTRINO_TABLE)
            )
            ratio = jnp.sum(
                w[None, :]
                * jnp.sqrt(1.0 + (jnp.exp(log_y)[:, None] / q[None, :]) ** 2),
                axis=1,
            )
            table = FastUniformCubicSpline1D(log_y, jnp.log(ratio))
        _NEUTRINO_TABLE[key] = table
    return table


def neutrino_density_ratio(a, amnu, xp=jnp):
    """``rho_nu(a) / rho_nu,massless`` of one flavour of mass ``amnu`` (1 when massless).

    With ``xp=jnp`` (the default) it reads the shared table, so it traces
    under ``jit``. With ``xp=np`` it sums the same quadrature directly, for
    closing the budget on the host for a cosmology given as floats.
    """

    if xp is np:
        q, w, _ = neutrino_momentum_bins(NQ_NEUTRINO_TABLE)
        y = np.asarray(a, dtype=np.float64) * float(amnu)
        return np.sum(w * np.sqrt(1.0 + (y[..., None] / q) ** 2), axis=-1)
    massive = amnu > 0.0
    ln_y = jnp.log(jnp.where(massive, amnu, 1.0) * a)
    ln_y = jnp.clip(ln_y, _LN_Y_MIN, _LN_Y_MAX)
    ratio = jnp.exp(jnp.vectorize(_neutrino_table().evaluate)(ln_y))
    return jnp.where(massive, ratio, 1.0)


_BACKGROUND_MODEL: list = []


def background_model() -> Model:
    """The species whose ``grho`` is the Friedmann sum, with every component enabled.

    At ``w = -1`` the dark-energy fluid is the cosmological constant, and a
    massive-neutrino density with ``n_mnu = 0`` vanishes, so one model serves
    every cosmology. The hierarchy truncations don't affect the background.
    """

    if not _BACKGROUND_MODEL:
        _BACKGROUND_MODEL.append(
            Model(
                [
                    Background(),
                    Curvature(),
                    ColdDarkMatter(),
                    Baryons(),
                    Photons(),
                    MasslessNeutrinos(),
                    DarkEnergy(),
                    MassiveNeutrinos(),
                ],
                sums=SUMS,
            )
        )
    return _BACKGROUND_MODEL[0]


def density_parameters(param, xp=jnp) -> dict:
    """The background densities of a parameter dictionary, keyed as the model names them.

    ``grhom`` is the critical density, ``grhog`` the photons', ``grhor`` one
    massless neutrino flavour's, ``grhornomass`` the ``Neff`` massless
    flavours'. ``grhoc``, ``grhob`` and ``grhok`` are the cold dark matter,
    baryons and curvature, and ``grhor_nu``, ``n_mnu`` and ``amnu`` the
    massive neutrinos'. ``grhov``, the dark energy's, closes the budget at
    ``a = 1`` with the model's own Friedmann sum:
    ``grhov = grhom - grho(a=1 | grhov=0)``. The dictionary also holds
    ``OmegaDE``, ``grho_mnu_today`` and the dark-energy parameters. Pass
    ``xp=jnp`` for a traced cosmology and ``xp=np`` for one given as floats.
    """

    H0, Tcmb = param["H0"], param["Tcmb"]
    grhom = GRHOM_PER_H0SQ * H0**2
    grhor = GRHOR_PER_TCMB4 * Tcmb**4
    d = {
        "grhom": grhom,
        "grhog": GRHOG_PER_TCMB4 * Tcmb**4,
        "grhor": grhor,
        "grhornomass": param["Neff"] * grhor,
        "grhoc": grhom * (param["Omegam"] - param["Omegab"]),
        "grhob": grhom * param["Omegab"],
        "grhok": grhom * param["Omegak"],
        "grhor_nu": grhor,
        "n_mnu": param["Nmnu"],
        "amnu": param["mnu"] * C2OK / Tcmb,
        "w_DE_0": param["w_DE_0"],
        "w_DE_a": param["w_DE_a"],
        "cs2_DE": param.get("cs2_DE", 1.0),
    }
    rhonu_today = neutrino_density_ratio(1.0, d["amnu"], xp)
    without_dark_energy = background_model().evaluate(
        "grho",
        math=(math if xp is np else jnp),
        a=1.0,
        rhonu=rhonu_today,
        grhov=0.0,
        **d,
    )
    d["grhov"] = grhom - without_dark_energy
    d["OmegaDE"] = d["grhov"] / grhom
    d["grho_mnu_today"] = grhor * d["n_mnu"] * rhonu_today
    return d


def expansion_rate(a, densities):
    """The conformal Hubble rate ``a'/a = a H`` at scalar ``a``, from the model's ``adotoa``."""

    rhonu = neutrino_density_ratio(a, densities["amnu"])
    return background_model().evaluate(
        "adotoa", math=jnp, a=a, rhonu=rhonu, **densities
    )


def dtauda(a, densities):
    """``d tau / d a = 1 / (a^2 H)`` at scalar ``a`` (``jax.vmap`` over an array)."""

    return 1.0 / (a * expansion_rate(a, densities))
