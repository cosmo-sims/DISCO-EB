"""One ODE solve from the cosmological parameters to the perturbations.

The background, the recombination history and the Einstein-Boltzmann hierarchy
of :mod:`discoeb.perturbations` are solved in one launch, as one state per wave
mode: :class:`Recombination` adds ``ln a`` and RECFAST's ``(x_H, x_He, T_m)`` to
the trajectory's state and provides ``a``, ``opacity``, ``c_s^2`` and ``rho_nu``
to the hierarchy from them, in place of the tabulated thermal history
(:class:`discoeb.perturbations.ThermalHistory`). The expansion rate it integrates
is the model's own Friedmann sum, the one the hierarchy already reads, and the
recombination rates are :func:`discoeb.thermodynamics_recfast.recfast_rhs`
statement for statement, written as providers.

These four states are *drivers* (``State(..., driver=True)``): their equations
read nothing of the hierarchy, so their rows of the Jacobian hold only their
own four columns, while their columns -- ``a`` through every density and the
opacity through every photon multipole's damping -- are declared in every
row. The Jacobian is block lower triangular and modax's direct solver fills
nothing in; what the fusion costs is four more Enzyme sweeps for the four dense
columns, the recombination rates in every stage evaluation of every wave mode,
and the error control of the thermal history by the solver's step size.

``Recombination`` carries RECFAST's three states and RECFAST's equations
unaltered by default (the one departure is that ``x_H`` is held to ``[0, 1]``,
outside which RECFAST's own solve never evaluates them). Two experiments that
do alter the equations are kept behind switches, off by default, as the record
of what they buy: ``DISCO_FUSED_SMOOTH_SWITCHES=1`` and ``DISCO_N_THERMO=1``
(``x_H`` alone, helium in Saha equilibrium, ``T_m`` from the closed-form
Compton-decoupling curve), the latter being what puts the fused solve ahead of
the table pipeline at 128 cosmologies x 128 modes.

The only thing solved before the launch is a quadrature: each cosmology's
conformal age ``tau_0``, where its trajectories stop, and the scale factor at
``TAU_START``, where they begin (:func:`background_endpoints`, a vmapped JAX
program of a few milliseconds for a batch).
"""

from __future__ import annotations

import math
import os

import jax
import jax.numpy as jnp
import numpy as np

from .component import Component, Setting, State
from .eb_model import (
    density_parameters,
    dtauda,
    neutrino_density_ratio,
    neutrino_momentum_bins,
    perturbation_model,
)
from . import perturbations as P
from .thermodynamics_recfast import (
    A2P_s, A2P_t, AGauss1, AGauss2, Bfact, CB1, CB1_He1, CDB, CDB_He, CK, CK_He, CL, CL_He, CL_PSt,
    CR, CT, DZ_H_SAHA_BLEND, H_frac, L_He2St_ion, L_He_2Pt, L_He_2p, Lambda, Lambda_He,
    NQ_NU_BACKGROUND, T_0, T_1, Z_H_SAHA_END, a_PPB, a_VF, a_trip, b_PPB, b_VF, b_trip, bigH, c_PPB,
    const_EionHe12s, const_c, const_c_Mpc_s, const_dens_fac, const_h, const_h_c_L_He_2St_kB,
    const_kB, const_mH, const_mHe_mH, d_PPB, sigma_He_2Ps, sigma_He_2Pt, wGauss1, wGauss2, zGauss1, zGauss2,
)

# The perturbation row (:data:`discoeb.perturbations.PARAM_COLUMNS`) plus what
# RECFAST reads of the cosmology.
IX_YHE = P.N_PARAM
IX_TCMB = P.N_PARAM + 1
IX_H0 = P.N_PARAM + 2
IX_OMEGAB = P.N_PARAM + 3
N_PARAM = P.N_PARAM + 4
PARAM_COLUMNS = {**P.PARAM_COLUMNS, "YHe": IX_YHE, "Tcmb": IX_TCMB, "H0": IX_H0, "Omegab": IX_OMEGAB}

FREEZE_RATES = os.environ.get("DISCO_FUSED_FREEZE_RATES", "") == "1"
"""Diagnostic: with ``DISCO_FUSED_FREEZE_RATES=1`` the recombination rates are zero (the
thermal state stays fully ionized), which compiles RECFAST out of the kernel and
isolates its cost per step."""

FROZEN_COUPLING = os.environ.get("DISCO_FUSED_FROZEN_COUPLING", "") == "1"
"""Experimental: pass ``a``, the opacity, ``c_s^2`` and ``rho_nu`` to the hierarchy
through :func:`freeze`, so Enzyme sees no dependence of the hierarchy on the
driver states. The iteration matrix is then block diagonal (a Rosenbrock-W
iteration, which keeps the order), the driver columns share colours with the
hierarchy's and the LU has no coupling block. Measured: ten times the steps
(3900 per mode), so the hierarchy's rows need their thermal columns -- the
Thomson damping ``-kappa' Theta_l`` is stiff *through* the opacity, not only
through ``Theta_l``. Kept as the record of that measurement."""

SMOOTH_SWITCHES = os.environ.get("DISCO_FUSED_SMOOTH_SWITCHES", "") == "1"
"""Experiment, off by default (it alters RECFAST's equations): blend RECFAST's helium Saha-to-rate switch (over ``x_He`` in [0.9, 0.95]) and the
Compton-coupling switch (over a factor of two in the time-scale ratio) with
smoothsteps instead of jumping. The stand-alone RECFAST solve steps to every
point of its 2048-point save grid and never notices the jumps; a hierarchy-sized
step does: at weight 1 the jumps cost 20 rejected steps per mode (the table
kernel has 0.4), 372 steps instead of 358, 15 percent of the time at
128 x 512 modes, and an opacity five times less accurate (1.6e-2 against
3.4e-3 of the kernel-side spline tables). ``DISCO_FUSED_SMOOTH_SWITCHES=1``
enables it."""

THERMO_ERROR_WEIGHT = 1.0
"""Weight of the four driver states in the step-size error norm (their error is
held to ``rtol / weight``). ``P(k)`` is 2e-4 from the table pipeline at every
weight from 0.3 to 100; what the weight sets is the thermal state's own
accuracy and the step count: 0.3 gives the opacity to 3e-2 at 341 steps per
mode, 1 to 3e-3 at 358, 3 to 2e-3 at 379, 10 to 8e-4 at 454, 100 to 2e-4 at
719 (the kernel-side spline tables of the table pipeline deliver 3e-4 at
most, 1e-5 rms), and 0.1 or 0 blows up."""


def freeze(x, scale):
    """``x`` to within ``1/scale``, with zero derivative under Enzyme.

    An integer round trip: ``fptosi`` has no derivative, so the value passes
    and the tangent does not. ``scale`` is a power of two chosen so that
    ``x * scale`` fits in 64 bits over the quantity's range.
    """

    return float(int(x * scale)) / scale


# --- RECFAST's rates as plain device helpers --------------------------------------------


def saha_slope(z, Tcmb, Nnow, E_ion, offset, g_factor, f_factor):
    """``d x / d z`` of the Saha ionization fraction at ``z`` (:func:`thermodynamics_recfast.Saha`).

    One exit: numba-cuda-mlir gives a function returning from several branches
    a result slot, and inlining it at two call sites of one caller merges the
    two slots into a memref phi that LLVM's typed-pointer verifier rejects.
    """

    T = Tcmb * (1.0 + z)
    phi = (CR * T) ** 1.5 / Nnow / (1.0 + z) ** 3
    R = g_factor * phi * math.exp(-E_ion / T)
    slope = 0.0
    if R > 1.0e-20 and T <= E_ion * 0.1:
        alpha = R + offset
        beta = R * f_factor
        dR = R * (-1.5 / (1.0 + z) + E_ion / (Tcmb * (1.0 + z) ** 2))
        discr = alpha * alpha + 4.0 * beta
        if discr < 1.0e-35:
            root = math.sqrt(1.0e-35)
            droot = 0.0
        else:
            root = math.sqrt(discr)
            droot = (alpha + 2.0 * f_factor) * dR / root
        denom = root + alpha
        slope = (2.0 * f_factor * dR * denom - 2.0 * beta * (droot + dR)) / (denom * denom)
    return slope


def saha_fraction(T, z, Nnow, E_ion, offset, g_factor, f_factor):
    """The Saha ionization fraction of one stage (:func:`thermodynamics_recfast.Saha`), scalar."""

    phi = (CR * T) ** 1.5 / Nnow / (1.0 + z) ** 3
    R = g_factor * phi * math.exp(-E_ion / T)
    fully = T > E_ion * 0.1
    alpha = 1.0 if fully else R + offset
    beta = f_factor * (1.0 + f_factor) if fully else R * f_factor
    discr = max(alpha * alpha + 4.0 * beta, 1.0e-35)
    return 2.0 * beta / (math.sqrt(discr) + alpha) if R > 1.0e-20 else 0.0


def helium_saha(zred, Tcmb, Nnow, fHe):
    """``x_He`` in Saha equilibrium, both stages, as RECFAST starts it."""

    T = Tcmb * (1.0 + zred)
    return (
        saha_fraction(T, zred, Nnow, CB1_He1, 1.0, 4.0, fHe)
        + saha_fraction(T, zred, Nnow, const_EionHe12s, 1.0 + fHe, 1.0, fHe)
    ) / fHe


def matter_temperature_fit(zred, Tcmb, xe, fHe, Hz):
    """``(T_m, d ln T_m / d ln a)`` without a ``T_m`` equation.

    With ``y = 1 / (t_Compton H)`` and ``y`` proportional to ``(1+z)^{5/2}`` (matter
    domination, frozen ``x_e``), RECFAST's temperature equation for
    ``u = T_m / T_rad`` is ``2.5 y du/dy = y (u - 1) + u``, one universal curve:
    ``u = 1 - 1/y`` while Compton coupling holds, ``u`` proportional to
    ``y^{0.4}`` (``T_m`` to ``(1+z)^2``) once it has let go. The closed form
    ``u = (y^p / (y^p + c))^{0.4/p}`` with ``p = 0.705``, ``c = 0.968`` fits that
    curve to 2 percent everywhere.
    """

    Trad = Tcmb * (1.0 + zred)
    y = max(CT * Trad**4 * xe / ((1.0 + xe + fHe) * Hz), 1.0e-12)
    p = 0.7052
    c = 0.9682
    yp = y**p
    ratio = (yp / (yp + c)) ** (0.4 / p)
    dlnu_dlny = 0.4 * c / (yp + c)
    return Trad * ratio, -1.0 - 2.5 * dlnu_dlny


def hydrogen_rate(x_H, xe, Tmat, z, n_H, Hz, Tcmb, Nnow, K_peebles):
    """``d x_H / d z``: Saha above ``Z_H_SAHA_END``, blended into the rate equation below it."""

    saha = saha_slope(z, Tcmb, Nnow, CB1, 0.0, 1.0, 1.0)
    f0 = saha
    if z < Z_H_SAHA_END:
        fu = 1.105
        one_minus_x_H = 1.0 - x_H
        Tmat_1e4 = Tmat / 1.0e4
        Rdown = 1.0e-19 * a_PPB * Tmat_1e4**b_PPB / (1.0 + c_PPB * Tmat_1e4**d_PPB)
        Rup = Rdown * (CR * Tmat) ** 1.5 * math.exp(-CDB / Tmat)
        rate_diff = xe * x_H * n_H * Rdown - Rup * one_minus_x_H * math.exp(-CL / Tmat)
        K_Lambda_n_1_minus_x_H = K_peebles * Lambda * n_H * one_minus_x_H
        denom_term = 1.0 / fu + K_Lambda_n_1_minus_x_H / fu + K_peebles * Rup * n_H * one_minus_x_H
        f0 = rate_diff * (1.0 + K_Lambda_n_1_minus_x_H) / (Hz * (1.0 + z) * denom_term)
        if z > Z_H_SAHA_END - 2.0 * DZ_H_SAHA_BLEND:
            s_blend = (Z_H_SAHA_END - z) / (2.0 * DZ_H_SAHA_BLEND)
            w_full = s_blend * s_blend * (3.0 - 2.0 * s_blend)
            f0 = w_full * f0 + (1.0 - w_full) * saha
    return f0


def helium_rate(x_H, x_He, xe, Tmat, z, a, n_H, Hz, Tcmb, Nnow, fHe):
    """``d x_He / d z``: both Saha stages while ``x_He > 0.9``, the singlet and triplet rates below.

    With ``SMOOTH_SWITCHES`` the two are blended over ``x_He`` in [0.9, 0.95].
    The rate equation is called from one place only and the blend is
    branch-free: a helper inlined at two call sites, or a local assigned in
    several branches, gives numba-cuda-mlir a memref phi that LLVM rejects.
    """

    saha = (
        saha_slope(z, Tcmb, Nnow, CB1_He1, 1.0, 4.0, fHe)
        + saha_slope(z, Tcmb, Nnow, const_EionHe12s, 1.0 + fHe, 1.0, fHe)
    ) / fHe
    f1 = 0.0
    if a < 3.0e-4 or x_He >= 0.95 or (x_He > 0.9 and not SMOOTH_SWITCHES):
        f1 = saha
    elif x_He >= 1.0e-8:
        rate = helium_rate_equation(x_H, x_He, xe, Tmat, z, n_H, Hz, fHe)
        w_saha = min(max((x_He - 0.9) / 0.05, 0.0), 1.0) if SMOOTH_SWITCHES else 0.0
        w_saha = w_saha * w_saha * (3.0 - 2.0 * w_saha)
        f1 = w_saha * saha + (1.0 - w_saha) * rate
    return f1


def helium_rate_equation(x_H, x_He, xe, Tmat, z, n_H, Hz, fHe):
    """The singlet and triplet rate equations of HeI recombination, ``d x_He / d z``."""

    Hz_z_term = Hz * (1.0 + z)
    n_He = fHe * n_H
    one_minus_x_He = 1.0 - x_He
    one_minus_x_H = 1.0 - x_H
    n_He_1_minus_x_He = n_He * one_minus_x_He
    CR_Tmat_15 = (CR * Tmat) ** 1.5
    sq_0 = math.sqrt(Tmat / T_0)
    sq_1 = math.sqrt(Tmat / T_1)
    Rdown_He = a_VF / (sq_0 * (1.0 + sq_0) ** (1.0 - b_VF) * (1.0 + sq_1) ** (1.0 + b_VF))
    Rup_He = 4.0 * Rdown_He * CR_Tmat_15 * math.exp(-CDB_He / Tmat)
    He_Boltz = math.exp(min(680.0, Bfact / Tmat))
    Rdown_trip = a_trip / (sq_0 * (1.0 + sq_0) ** (1.0 - b_trip) * (1.0 + sq_1) ** (1.0 + b_trip))
    Rup_trip = (
        Rdown_trip * math.exp(-const_h * const_c * L_He2St_ion / (const_kB * Tmat)) * CR_Tmat_15 * (4.0 / 3.0)
    )

    tauHe_s = A2P_s * CK_He * 3.0 * n_He_1_minus_x_He / Hz
    pHe_s = (1.0 - math.exp(-tauHe_s)) / tauHe_s
    Doppler_term = math.sqrt(2.0 * const_kB * Tmat / (const_mH * const_mHe_mH * const_c**2))
    if x_H < 0.9999999:
        Doppler_2p = const_c * L_He_2p * Doppler_term
        gamma_2Ps = (
            3.0 * A2P_s * fHe * one_minus_x_He * const_c**2
            / (math.sqrt(math.pi) * sigma_He_2Ps * 8.0 * math.pi * Doppler_2p * one_minus_x_H * (const_c * L_He_2p) ** 2)
        )
        AHcon = A2P_s / (1.0 + 0.36 * gamma_2Ps**0.86)
        K_He = 1.0 / ((A2P_s * pHe_s + AHcon) * 3.0 * n_He_1_minus_x_He)
    else:
        K_He = 1.0 / (A2P_s * pHe_s * 3.0 * n_He_1_minus_x_He)

    rate_diff_He = xe * x_He * n_H * Rdown_He - Rup_He * one_minus_x_He * math.exp(-CL_He / Tmat)
    K_He_Lambda = K_He * Lambda_He * n_He_1_minus_x_He * He_Boltz
    He_denom_term = 1.0 + K_He * (Lambda_He + Rup_He) * n_He_1_minus_x_He * He_Boltz
    f1 = rate_diff_He * (1.0 + K_He_Lambda) / (Hz_z_term * He_denom_term)

    tauHe_t = A2P_t * n_He_1_minus_x_He * 3.0 / (8.0 * math.pi * Hz * L_He_2Pt**3)
    pHe_t = (1.0 - math.exp(-tauHe_t)) / tauHe_t
    exp_CL_PSt = math.exp(-CL_PSt / Tmat)
    if x_H > 0.99999:
        CfHe_t = A2P_t * pHe_t * exp_CL_PSt
    else:
        Doppler_2Pt = const_c * L_He_2Pt * Doppler_term
        gamma_2Pt = (
            3.0 * A2P_t * fHe * one_minus_x_He * const_c**2
            / (math.sqrt(math.pi) * sigma_He_2Pt * 8.0 * math.pi * Doppler_2Pt * one_minus_x_H * (const_c * L_He_2Pt) ** 2)
        )
        AHcon_t = A2P_t / (1.0 + 0.66 * gamma_2Pt**0.9) / 3.0
        CfHe_t = (A2P_t * pHe_t + AHcon_t) * exp_CL_PSt
    CfHe_t = CfHe_t / (Rup_trip + CfHe_t)
    trip_rate_diff = (
        xe * x_He * n_H * Rdown_trip
        - one_minus_x_He * 3.0 * Rup_trip * math.exp(-const_h_c_L_He_2St_kB / Tmat)
    )
    f1 = f1 + trip_rate_diff * CfHe_t / Hz_z_term
    return f1


# --- the component ------------------------------------------------------------------


def _thermo_initial(self, lna_start, Tcmb, output):
    """Fully ionized and in Compton equilibrium: ``x_H = 1``, ``x_He = 2``, ``T_m = T_cmb / a``."""

    output[0] = 1.0
    if self.N_THERMO == 3:
        output[1] = 2.0
        output[2] = Tcmb * math.exp(-lna_start)


class Recombination(Component):
    """The expansion and the recombination history as states of each trajectory.

    ``lna`` is ``ln a``, integrated from the model's own conformal Hubble rate
    (``d ln a / d tau = adotoa``); ``thermo`` is RECFAST's ``(x_H, x_He, T_m)``,
    whose rates in ``z`` are converted with ``dz/dtau = -adotoa / a``. From them
    the component provides what the hierarchy reads of the thermal history:
    ``a``, the Thomson opacity ``kappa' = x_e n_H sigma_T a``, the baryon
    sound speed ``c_s^2`` (from ``T_m`` and its own rate, so no spline
    derivative is needed) and the massive-neutrino density ratio ``rhonu`` by
    the ``NQ_NU_BACKGROUND``-node quadrature of the RECFAST kernel.
    """

    N_THERMO = Setting(3)
    """3 (default): ``(x_H, x_He, T_m)`` as RECFAST; 1 (experiment, alters the
    physics): ``x_H`` alone, with helium in Saha equilibrium and ``T_m`` from
    :func:`matter_temperature_fit` (``DISCO_N_THERMO``)."""

    lna = State(initial=lambda lna_start: lna_start, driver=True)
    thermo = State(lambda self: self.N_THERMO, initial=_thermo_initial, driver=True)

    def __init__(self, **settings):
        super().__init__(**settings)
        if self.N_THERMO not in (1, 3):
            raise ValueError("N_THERMO is 1 or 3")
        q, w, _ = neutrino_momentum_bins(NQ_NU_BACKGROUND)
        self.nu_q = tuple(float(x) for x in q)
        self.nu_w = tuple(float(x) for x in w)
        # The providers that differ between the two layouts are set per instance.
        if self.N_THERMO == 3:
            self.x_He, self.Tmat, self.dlnTm_dlna, self.d_thermo = _x_He_state, _Tmat_state, _dlnTm_dlna_state, _d_thermo_full
        else:
            self.x_He, self.Tmat, self.dlnTm_dlna, self.d_thermo = _x_He_saha, _Tmat_fit, _dlnTm_dlna_fit, _d_thermo_hydrogen

    # -- what the hierarchy reads -----------------------------------------------------

    def a(self, lna):
        a = math.exp(lna)
        return freeze(a, 4611686018427387904.0) if FROZEN_COUPLING else a  # 2^62: a <= 1.01

    def rhonu(self, a, amnu):
        """``rho_nu / rho_nu(massless)`` of one flavour, 1 when massless."""
        total = 1.0
        if amnu > 0.0:  # eight square roots, skipped for massless neutrinos
            total = 0.0
            for i in range(len(self.nu_q)):
                total += self.nu_w[i] * math.sqrt(1.0 + (a * amnu / self.nu_q[i]) ** 2)
        return freeze(total, 1073741824.0) if FROZEN_COUPLING else total  # 2^30

    def fHe(self, YHe):
        return YHe / (const_mHe_mH * (1.0 - YHe))

    def x_H(self, thermo):
        """Held to ``[0, 1]``: a step of the hierarchy's size can overshoot the
        Saha-to-rate handover at ``z = 1600``, and above 1 the rate equation's
        denominator changes sign and drives ``x_H`` away instead of back."""
        return min(max(thermo[0], 0.0), 1.0)


    def xe(self, x_H, x_He, fHe):
        """Free electrons per hydrogen nucleus."""
        return x_H + fHe * x_He

    def akthom(self, YHe, Omegab, H0):
        return 2.3038921003709498e-9 * (1.0 - YHe) * Omegab * H0**2

    def opacity(self, xe, akthom, inv_a2):
        opacity = max(xe * akthom * inv_a2, 1.0e-30)
        if FROZEN_COUPLING:  # 2^56 on the log: |ln opacity| < 70
            opacity = math.exp(freeze(math.log(opacity), 72057594037927936.0))
        return opacity

    def cs2(self, Tmat, xe, YHe, dlnTm_dlna):
        """``c_s^2 = k_B T_m / (mu m_H c^2) (1 - (1/3) d ln T_m / d ln a)``."""
        mu = 1.0 / (1.0 + (1.0 / const_mHe_mH - 1.0) * YHe + (1.0 - YHe) * xe)
        cs2 = max(const_kB / const_mH / const_c**2 / mu * Tmat * (3.0 - dlnTm_dlna) / 3.0, 0.0)
        return freeze(cs2, 4611686018427387904.0) if FROZEN_COUPLING else cs2  # 2^62: cs2 < 1e-4

    # -- the rates ----------------------------------------------------------------------

    def zred(self, inv_a):
        return inv_a - 1.0

    def Hz(self, adotoa, inv_a):
        """``H`` in 1/s, from the model's conformal Hubble rate in 1/Mpc."""
        return adotoa * inv_a * const_c_Mpc_s

    def Nnow(self, H0, Omegab, YHe):
        """The hydrogen number density today, in 1/m^3."""
        return const_dens_fac * (H0 / 100.0) ** 2 * Omegab * (1.0 - YHe)

    def n_H(self, Nnow, inv_a):
        return Nnow * inv_a**3

    def K_peebles(self, Hz, zred):
        """The Peebles ``K`` with the Gaussian fudges of Rubino-Martin et al. (2010)."""
        log_z_term = math.log(1.0 + zred)
        return CK / Hz * (
            1.0
            + AGauss1 * math.exp(-(((log_z_term - zGauss1) / wGauss1) ** 2))
            + AGauss2 * math.exp(-(((log_z_term - zGauss2) / wGauss2) ** 2))
        )

    def dTm_dz(self, Tmat, xe, fHe, zred, Hz, H0, Tcmb):
        """RECFAST's matter-temperature equation, with ``T_m`` locked to ``T_rad`` in tight Compton coupling."""
        z_term = 1.0 + zred
        Trad = Tcmb * z_term
        timeTh = (1.0 / (CT * Trad**4)) * (1.0 + xe + fHe) / xe
        timeH = 2.0 / (3.0 * (H0 / 100.0) * bigH * z_term**1.5)
        locked = Tmat / z_term
        full = CT * Trad**4 * xe / (1.0 + xe + fHe) * (Tmat - Trad) / (Hz * z_term) + 2.0 * Tmat / z_term
        if SMOOTH_SWITCHES:
            # smoothstep in log2 of the ratio timeTh / (H_frac timeH) over [1/2, 1], branch-free
            w_full = min(max(math.log(timeTh / (H_frac * timeH)) / math.log(2.0) + 1.0, 0.0), 1.0)
            w_full = w_full * w_full * (3.0 - 2.0 * w_full)
        else:
            w_full = 1.0 if timeTh >= H_frac * timeH else 0.0
        return w_full * full + (1.0 - w_full) * locked

    def d_lna(self, adotoa):
        return adotoa



# -- the providers that depend on the layout (set on the instance in __init__) ----------


def _x_He_state(thermo):
    return min(max(thermo[1], 0.0), 2.0)


def _Tmat_state(thermo):
    return abs(thermo[2])


def _dlnTm_dlna_state(dTm_dz, inv_a, Tmat):
    return -dTm_dz * inv_a / Tmat


def _d_thermo_full(x_H, x_He, xe, Tmat, zred, a, n_H, Hz, Tcmb, Nnow, fHe, K_peebles, dTm_dz, adotoa, inv_a, output):
    dz_dtau = 0.0 if FREEZE_RATES else -adotoa * inv_a
    output[0] = hydrogen_rate(x_H, xe, Tmat, zred, n_H, Hz, Tcmb, Nnow, K_peebles) * dz_dtau
    output[1] = helium_rate(x_H, x_He, xe, Tmat, zred, a, n_H, Hz, Tcmb, Nnow, fHe) * dz_dtau
    output[2] = dTm_dz * dz_dtau


def _x_He_saha(zred, Tcmb, Nnow, fHe):
    return helium_saha(zred, Tcmb, Nnow, fHe)


def _Tmat_fit(zred, Tcmb, xe, fHe, Hz):
    return matter_temperature_fit(zred, Tcmb, xe, fHe, Hz)[0]


def _dlnTm_dlna_fit(zred, Tcmb, xe, fHe, Hz):
    return matter_temperature_fit(zred, Tcmb, xe, fHe, Hz)[1]


def _d_thermo_hydrogen(x_H, xe, Tmat, zred, n_H, Hz, Tcmb, Nnow, K_peebles, adotoa, inv_a, output):
    dz_dtau = 0.0 if FREEZE_RATES else -adotoa * inv_a
    output[0] = hydrogen_rate(x_H, xe, Tmat, zred, n_H, Hz, Tcmb, Nnow, K_peebles) * dz_dtau


# --- the model and its kernel ---------------------------------------------------------

_KERNELS: dict = {}


def fused_model(layout):
    """:func:`discoeb.eb_model.perturbation_model` with :class:`Recombination` as its thermal history."""

    return perturbation_model(layout, Recombination())


def fused_kernel(layout):
    """``(model, rhs, sparsity, error_weights)`` of a layout, built once."""

    key = (layout, THERMO_ERROR_WEIGHT, FROZEN_COUPLING, SMOOTH_SWITCHES, Recombination.N_THERMO)
    cached = _KERNELS.get(key)
    if cached is None:
        model = fused_model(layout)
        rhs = model.build(PARAM_COLUMNS)
        weights = np.ones(model.nvar)
        weights[model.drivers()] = THERMO_ERROR_WEIGHT
        sparsity = model.sparsity()
        if FROZEN_COUPLING:  # the hierarchy's rows lose the driver columns: block diagonal
            drivers = set(model.drivers())
            sparsity = np.array([(r, c) for r, c in sparsity if r in drivers or c not in drivers])
        cached = _KERNELS[key] = (model, rhs, sparsity, weights)
    return cached


@jax.jit
def background_endpoints(stacked):
    """``(tau0, lna_start)`` of stacked background parameters, one vmap lane each.

    The conformal-time quadrature of :func:`discoeb.perturbations._thermo_tables_device`
    on its ``N_BACKGROUND_GRID`` geometric grid from ``a = 1e-9``, with the same
    zero point ``taumin = amin / adotrad``; ``lna_start`` is ``ln a`` at
    ``TAU_START`` interpolated the same way the tables interpolate ``a(tau)``.
    """

    def one(params):
        d = density_parameters({k: v for k, v in params.items() if k != "tau_reio"})
        a_bg = jnp.geomspace(1.0e-9, 1.0, P.N_BACKGROUND_GRID)
        dt = jax.vmap(dtauda, in_axes=(0, None))(a_bg, d)
        steps = 0.5 * (dt[1:] + dt[:-1]) * jnp.diff(a_bg)
        radiation = (
            d["grhog"] + d["grhornomass"]
            + d["grhor_nu"] * d["n_mnu"] * neutrino_density_ratio(1.0e-9, d["amnu"])
        )
        taumin = 1.0e-9 / jnp.sqrt(radiation / 3.0)
        tau_bg = jnp.concatenate((jnp.zeros((1,)), jnp.cumsum(steps))) + taumin
        lna_start = jnp.log(jnp.interp(P.TAU_START, tau_bg, a_bg))
        return tau_bg[-1], lna_start

    return jax.vmap(one)(stacked)


def pack_fused_batch(cosmologies, k_values, tau_save=None):
    """``(model, rhs, sparsity, weights, params, y0, t_span, order, k_sorted, tau0)`` of a batch.

    Trajectories are packed as in :func:`discoeb.perturbations._pack_batch`
    (row ``k_idx * n_cosmo + cosmology_idx``), each with its own ``tau0`` in
    ``IX_TAU_END``; the initial state is the hierarchy's adiabatic mode plus the
    fully ionized thermal state at ``lna_start``.
    """

    cosmologies = tuple(P._as_cosmology(c) for c in cosmologies)
    layout = P._layout_for(cosmologies[0])
    for c in cosmologies[1:]:
        if P._layout_for(c) != layout:
            raise ValueError("all cosmologies in a batch must share a perturbation layout")
    model, rhs, sparsity, weights = fused_kernel(layout)

    stacked = P._stack_background_params(cosmologies)
    tau0, lna_start = (np.asarray(x) for x in background_endpoints(stacked))

    k_arr = np.ascontiguousarray(np.asarray(k_values, dtype=np.float64))
    order = np.argsort(k_arr, kind="stable")
    k_sorted = k_arr[order]
    n_k, n_cosmo = len(k_sorted), len(cosmologies)
    params = np.empty((n_k * n_cosmo, N_PARAM), dtype=np.float64)
    y0 = np.empty((n_k * n_cosmo, model.nvar), dtype=np.float64)
    for ci, c in enumerate(cosmologies):
        base = np.zeros((n_k, N_PARAM))
        base[:, : P.N_PARAM] = P.make_params(c, k_sorted, float(tau0[ci]))
        base[:, P.IX_COSMOLOGY] = float(ci)
        base[:, IX_YHE], base[:, IX_TCMB], base[:, IX_H0], base[:, IX_OMEGAB] = c.Y_He, c.T_cmb, c.H0, c.Omegab
        params[ci::n_cosmo] = base
        grhog, grhornomass, grhoc, grhob, grhov = P.density_coefficients(c)
        y0[ci::n_cosmo] = model.initial_state(
            k_sorted, tau=P.TAU_START, grhog=grhog, grhornomass=grhornomass, grhoc=grhoc, grhob=grhob,
            grhov=grhov, w_DE_0=c.w_DE_0, w_DE_a=c.w_DE_a, lna_start=float(lna_start[ci]), Tcmb=c.T_cmb,
        )
    if tau_save is None:
        t_span = np.asarray((P.TAU_START, float(np.max(tau0))), dtype=np.float64)
    else:
        t_span = np.ascontiguousarray(np.asarray(tau_save, dtype=np.float64))
    return model, rhs, sparsity, weights, np.ascontiguousarray(params), np.ascontiguousarray(y0), t_span, order, k_sorted, tau0


def solve_fused_batch(
    k_values,
    cosmologies,
    *,
    tau_save=None,
    rtol: float = P.PERTURB_RTOL,
    atol: float = P.PERTURB_ATOL,
    first_step: float = P.PERTURB_FIRST_STEP,
    max_steps: int = P.PERTURB_MAX_STEPS,
    lu_precision: str = "fp32",
    return_stats: bool = False,
    error_weights=None,
):
    """The fused solve of a batch: ``(history, k_sorted, model, tau0[, stats])``.

    ``history`` is ``(n_k, n_cosmo, n_save, nvar)`` on the device in ascending
    ``k``, the four driver states first in ``nvar``; ``error_weights`` overrides
    the kernel's (``THERMO_ERROR_WEIGHT`` on the drivers, 1 elsewhere).
    """

    from modax.rodas5P import solve as rodas5P_solve

    model, rhs, sparsity, weights, params, y0, t_span, order, k_sorted, tau0 = pack_fused_batch(
        cosmologies, k_values, tau_save
    )
    result = rodas5P_solve(
        rhs, y0, t_span, params,
        sparsity=sparsity,
        lu_precision=lu_precision,
        rtol=rtol, atol=atol, first_step=first_step, max_steps=max_steps,
        pcoeff=0.3, icoeff=0.4,
        trajectories_per_block=P.TRAJECTORIES_PER_BLOCK,
        tf_index=P.IX_TAU_END,
        max_registers=P.CUDA_MAX_REGISTERS,
        error_weights=weights if error_weights is None else error_weights,
        return_stats=return_stats,
    )
    sol, stats = result if return_stats else (result, None)
    n_k, n_cosmo = len(k_sorted), len(cosmologies)
    hist = jnp.asarray(sol).reshape(n_k, n_cosmo, len(t_span), model.nvar)
    return (hist, k_sorted, model, tau0) + ((stats,) if return_stats else ())


def solve_matter_power_spectrum_fused_batch(k_values, cosmologies, **solve_kwargs):
    """``P(k)`` in Mpc^3, shape ``(n_cosmo, n_k)``, from one fused launch."""

    hist, k_sorted, model, _ = solve_fused_batch(k_values, cosmologies, **solve_kwargs)[:4]
    y_final = np.asarray(hist[:, :, -1, :])  # (n_k, n_cosmo, nvar)
    ic, ib = model.index["clxc"], model.index["clxb"]
    cosmologies = [P._as_cosmology(c) for c in cosmologies]
    dens = np.array([P.density_coefficients(c) for c in cosmologies])  # (n_cosmo, 5)
    grhoc, grhob = dens[:, 2][None, :], dens[:, 3][None, :]
    delta_m = (grhoc * y_final[:, :, ic] + grhob * y_final[:, :, ib]) / (grhoc + grhob)
    A_s = np.array([c.A_s for c in cosmologies])[None, :]
    n_s = np.array([c.n_s for c in cosmologies])[None, :]
    k_p = np.array([c.k_pivot for c in cosmologies])[None, :]
    ks = k_sorted[:, None]
    pk_sorted = 2.0 * np.pi**2 / ks**3 * A_s * (ks / k_p) ** (n_s - 1.0) * delta_m**2  # (n_k, n_cosmo)
    order = np.argsort(np.asarray(k_values, dtype=np.float64), kind="stable")
    pk = np.empty_like(pk_sorted.T)
    pk[:, order] = pk_sorted.T
    return pk
