"""The recombination history as states of each wave mode's trajectory.

RECFAST's ``(x_H, x_He, T_m)`` (:mod:`discoeb.thermodynamics_recfast`, statement
for statement), ``ln a`` from the model's own Friedmann sum and the optical depth
``kappa = int kappa' dtau`` ride in front of the Einstein-Boltzmann hierarchy as
five more states of the joint system, and :class:`Recombination` provides the
hierarchy with ``a``, the Thomson opacity, ``c_s^2`` and the massive-neutrino
density ratio from them. These states read nothing of the hierarchy, so the
Jacobian pattern traced through the providers is block lower triangular, and
modax (``split=True``) finds the block in the pattern and integrates it ahead of
the hierarchy in every step, handing it over as a quintic polynomial of time. Nothing is tabulated and no device buffer is refilled: the
right-hand side is compiled once per layout and serves every batch.

RECFAST's equations are unaltered. Two implementation choices sit beside them:
``x_H`` is held to ``[0, 1]`` (a hierarchy-sized step can overshoot the
Saha-to-rate handover at ``z = 1600``, after which the rate equation's
denominator changes sign and drives ``x_H`` away), and the transcendental
functions of the rates are evaluated in single precision by default
(:data:`FP32_MATH`).

Reionization is CAMB's: hydrogen and HeII reionize together in a tanh step of
width 0.5 in ``y = (1+z)^{3/2}`` about ``z_re``, HeIII at ``z = 3.5``;
``z_re`` is a parameter of the row, solved from the optical depth
``tau_reio`` on the background by :func:`discoeb.perturbations.background_endpoints`.
"""

from __future__ import annotations

import math
import os

import numpy as np

from .component import Component, State, float32, float64
from .eb_model import neutrino_momentum_bins
from .thermodynamics_recfast import (
    A2P_s, A2P_t, AGauss1, AGauss2, Bfact, CB1, CB1_He1, CDB, CDB_He, CK, CK_He, CL, CL_He, CL_PSt,
    CR, CT, DZ_H_SAHA_BLEND, H_frac, L_He2St_ion, L_He_2Pt, L_He_2p, Lambda, Lambda_He,
    NQ_NU_BACKGROUND, T_0, T_1, Z_H_SAHA_END, a_PPB, a_VF, a_trip, b_PPB, b_VF, b_trip, bigH, c_PPB,
    const_EionHe12s, const_c, const_c_Mpc_s, const_dens_fac, const_h, const_h_c_L_He_2St_kB,
    const_kB, const_mH, const_mHe_mH, d_PPB, sigma_He_2Ps, sigma_He_2Pt, wGauss1, wGauss2, zGauss1, zGauss2,
)

THERMAL_PARAMETERS = ("YHe", "Tcmb", "H0", "Omegab", "z_re")
"""What the recombination rates and reionization read of the cosmology, as
columns of the parameter row after the hierarchy's own
(:data:`discoeb.perturbations.PARAM_COLUMNS`). ``z_re = 0`` means no reionization."""

THERMO_ERROR_WEIGHT = 1.0
"""Weight of ``ln a`` and the three RECFAST states in the step-size error norm
(their error is held to ``rtol / weight``). ``P(k)`` is unchanged from 0.3 to
100; the weight sets the thermal state's own accuracy and the step count
(opacity to 4e-2 at 0.3, 2e-2 at 1, 2e-4 at 100, against 3e-4 for the
2048-point spline tables this replaced), and 0.1 or 0 blows up. The optical
depth is excluded from the norm: its increments are integrated to the state's
accuracy whatever its magnitude."""

AKTHOM_PER_OMEGAB_H0SQ = 2.3038921003709498e-9
"""``n_H sigma_T c / (H_0 units)``: ``kappa' = x_e akthom / a^2`` with ``akthom = this (1 - Y_He) Omega_b H_0^2``."""

FP32_MATH = os.environ.get("DISCO_THERMO_FP32_MATH", "1") == "1"
"""Evaluate the exponentials, fractional powers and logarithms of RECFAST's
rates in single precision (on by default; ``DISCO_THERMO_FP32_MATH=0`` keeps
them double): the same equations,
with those functions good to 1e-7 instead of 1e-16, on a card whose fp64
transcendentals run at 1/64 rate. Only where the argument range allows it:
``(CR T)^{3/2}`` reaches 1e42 and the Boltzmann factor ``e^{680}``, so those
stay double; the ``e^{-E/T}`` factors stay inside fp32's range wherever the
term they weight matters. This is what puts the fused solve ahead of the
table pipeline at 128 cosmologies x 128 modes (0.53 s against 0.61 s) at the
same P(k) and thermal accuracy."""

if FP32_MATH:

    def exp_(x):
        return float64(math.exp(float32(x)))

    def pow_(x, y):
        return float64(float32(x) ** float32(y))

    def log_(x):
        return float64(math.log(float32(x)))

else:

    def exp_(x):
        return math.exp(x)

    def pow_(x, y):
        return x**y

    def log_(x):
        return math.log(x)




# --- RECFAST's rates as plain device helpers --------------------------------------------


def saha_slope(z, Tcmb, Nnow, E_ion, offset, g_factor, f_factor):
    """``d x / d z`` of the Saha ionization fraction at ``z`` (:func:`thermodynamics_recfast.Saha`).

    One exit: numba-cuda-mlir gives a function returning from several branches
    a result slot, and inlining it at two call sites of one caller merges the
    two slots into a memref phi that LLVM's typed-pointer verifier rejects.
    """

    T = Tcmb * (1.0 + z)
    phi = (CR * T) ** 1.5 / Nnow / (1.0 + z) ** 3
    R = g_factor * phi * exp_(-E_ion / T)
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


def hydrogen_rate(x_H, xe, Tmat, z, n_H, Hz, Tcmb, Nnow, K_peebles):
    """``d x_H / d z``: Saha above ``Z_H_SAHA_END``, blended into the rate equation below it."""

    saha = saha_slope(z, Tcmb, Nnow, CB1, 0.0, 1.0, 1.0)
    f0 = saha
    if z < Z_H_SAHA_END:
        fu = 1.105
        one_minus_x_H = 1.0 - x_H
        Tmat_1e4 = Tmat / 1.0e4
        Rdown = 1.0e-19 * a_PPB * pow_(Tmat_1e4, b_PPB) / (1.0 + c_PPB * pow_(Tmat_1e4, d_PPB))
        Rup = Rdown * (CR * Tmat) ** 1.5 * exp_(-CDB / Tmat)
        rate_diff = xe * x_H * n_H * Rdown - Rup * one_minus_x_H * exp_(-CL / Tmat)
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

    The rate equation is called from one place only: a helper inlined at two
    call sites, or a local assigned in several branches, gives numba-cuda-mlir
    a memref phi that LLVM rejects.
    """

    saha = (
        saha_slope(z, Tcmb, Nnow, CB1_He1, 1.0, 4.0, fHe)
        + saha_slope(z, Tcmb, Nnow, const_EionHe12s, 1.0 + fHe, 1.0, fHe)
    ) / fHe
    f1 = 0.0
    if x_He > 0.9 or a < 3.0e-4:
        f1 = saha
    elif x_He >= 1.0e-8:
        f1 = helium_rate_equation(x_H, x_He, xe, Tmat, z, n_H, Hz, fHe)
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
    Rdown_He = a_VF / (sq_0 * pow_(1.0 + sq_0, 1.0 - b_VF) * pow_(1.0 + sq_1, 1.0 + b_VF))
    Rup_He = 4.0 * Rdown_He * CR_Tmat_15 * exp_(-CDB_He / Tmat)
    He_Boltz = math.exp(min(680.0, Bfact / Tmat))
    Rdown_trip = a_trip / (sq_0 * pow_(1.0 + sq_0, 1.0 - b_trip) * pow_(1.0 + sq_1, 1.0 + b_trip))
    Rup_trip = (
        Rdown_trip * exp_(-const_h * const_c * L_He2St_ion / (const_kB * Tmat)) * CR_Tmat_15 * (4.0 / 3.0)
    )

    tauHe_s = A2P_s * CK_He * 3.0 * n_He_1_minus_x_He / Hz
    pHe_s = (1.0 - exp_(-tauHe_s)) / tauHe_s
    Doppler_term = math.sqrt(2.0 * const_kB * Tmat / (const_mH * const_mHe_mH * const_c**2))
    if x_H < 0.9999999:
        Doppler_2p = const_c * L_He_2p * Doppler_term
        gamma_2Ps = (
            3.0 * A2P_s * fHe * one_minus_x_He * const_c**2
            / (math.sqrt(math.pi) * sigma_He_2Ps * 8.0 * math.pi * Doppler_2p * one_minus_x_H * (const_c * L_He_2p) ** 2)
        )
        AHcon = A2P_s / (1.0 + 0.36 * pow_(gamma_2Ps, 0.86))
        K_He = 1.0 / ((A2P_s * pHe_s + AHcon) * 3.0 * n_He_1_minus_x_He)
    else:
        K_He = 1.0 / (A2P_s * pHe_s * 3.0 * n_He_1_minus_x_He)

    rate_diff_He = xe * x_He * n_H * Rdown_He - Rup_He * one_minus_x_He * exp_(-CL_He / Tmat)
    K_He_Lambda = K_He * Lambda_He * n_He_1_minus_x_He * He_Boltz
    He_denom_term = 1.0 + K_He * (Lambda_He + Rup_He) * n_He_1_minus_x_He * He_Boltz
    f1 = rate_diff_He * (1.0 + K_He_Lambda) / (Hz_z_term * He_denom_term)

    tauHe_t = A2P_t * n_He_1_minus_x_He * 3.0 / (8.0 * math.pi * Hz * L_He_2Pt**3)
    pHe_t = (1.0 - exp_(-tauHe_t)) / tauHe_t
    exp_CL_PSt = exp_(-CL_PSt / Tmat)
    if x_H > 0.99999:
        CfHe_t = A2P_t * pHe_t * exp_CL_PSt
    else:
        Doppler_2Pt = const_c * L_He_2Pt * Doppler_term
        gamma_2Pt = (
            3.0 * A2P_t * fHe * one_minus_x_He * const_c**2
            / (math.sqrt(math.pi) * sigma_He_2Pt * 8.0 * math.pi * Doppler_2Pt * one_minus_x_H * (const_c * L_He_2Pt) ** 2)
        )
        AHcon_t = A2P_t / (1.0 + 0.66 * pow_(gamma_2Pt, 0.9)) / 3.0
        CfHe_t = (A2P_t * pHe_t + AHcon_t) * exp_CL_PSt
    CfHe_t = CfHe_t / (Rup_trip + CfHe_t)
    trip_rate_diff = (
        xe * x_He * n_H * Rdown_trip
        - one_minus_x_He * 3.0 * Rup_trip * exp_(-const_h_c_L_He_2St_kB / Tmat)
    )
    f1 = f1 + trip_rate_diff * CfHe_t / Hz_z_term
    return f1


def peebles_K(Hz, zred):
    """The Peebles ``K`` with the Gaussian fudges of Rubino-Martin et al. (2010)."""

    log_z_term = log_(1.0 + zred)
    return CK / Hz * (
        1.0
        + AGauss1 * exp_(-(((log_z_term - zGauss1) / wGauss1) ** 2))
        + AGauss2 * exp_(-(((log_z_term - zGauss2) / wGauss2) ** 2))
    )


def matter_temperature_rate(Tmat, xe, fHe, zred, Hz, H0, Tcmb):
    """RECFAST's matter-temperature equation ``dT_m/dz``, ``T_m`` locked to ``T_rad`` in tight Compton coupling."""

    z_term = 1.0 + zred
    Trad = Tcmb * z_term
    timeTh = (1.0 / (CT * Trad**4)) * (1.0 + xe + fHe) / xe
    timeH = 2.0 / (3.0 * (H0 / 100.0) * bigH * pow_(z_term, 1.5))
    locked = Tmat / z_term
    full = CT * Trad**4 * xe / (1.0 + xe + fHe) * (Tmat - Trad) / (Hz * z_term) + 2.0 * Tmat / z_term
    w_full = 1.0 if timeTh >= H_frac * timeH else 0.0
    return w_full * full + (1.0 - w_full) * locked




def tanh_(x):
    """``tanh`` through ``exp``: Enzyme has no forward rule for libdevice's ``tanh``.

    ``1 - 2 / (1 + e^{2x})`` goes to ``1`` as ``e^{2x}`` overflows and to ``-1`` as it
    underflows, so the step functions stay finite at every redshift; the argument is
    capped where the host's ``exp`` would raise instead of returning ``inf``.
    """

    return 1.0 - 2.0 / (1.0 + math.exp(min(2.0 * x, 700.0)))


# --- the component ------------------------------------------------------------------


def _thermo_initial(lna_start, Tcmb, output):
    """Fully ionized and in Compton equilibrium: ``x_H = 1``, ``x_He = 2``, ``T_m = T_cmb / a``."""

    output[0] = 1.0
    output[1] = 2.0
    output[2] = Tcmb * math.exp(-lna_start)


class Recombination(Component):
    """``ln a``, RECFAST's ``(x_H, x_He, T_m)`` and the optical depth as driver states.

    ``lna`` is integrated from the model's conformal Hubble rate
    (``d ln a / d tau = adotoa``); ``thermo`` holds RECFAST's three states,
    whose rates in ``z`` are converted with ``dz/dtau = -adotoa / a``;
    ``kappa`` is ``int_{tau_start}^tau kappa' dtau``, from which the hook of
    :mod:`discoeb.cmb` forms the visibility as ``kappa'(tau) exp(-(kappa(tau_0) -
    kappa(tau)))``. The quantities the hierarchy reads: ``a``, ``opacity``
    (``kappa' = x_e n_H sigma_T a``, with reionization), ``cs2`` (from ``T_m``
    and its own rate, no spline derivative) and ``rhonu`` (the
    ``NQ_NU_BACKGROUND``-node quadrature of the RECFAST kernel).
    """

    lna = State(initial=lambda lna_start: lna_start)
    thermo = State(3, initial=_thermo_initial)
    kappa = State()

    def __init__(self, **settings):
        super().__init__(**settings)
        q, w, _ = neutrino_momentum_bins(NQ_NU_BACKGROUND)
        self.nu_q = tuple(float(x) for x in q)
        self.nu_w = tuple(float(x) for x in w)

    # -- what the hierarchy reads -----------------------------------------------------

    def a(self, lna):
        return math.exp(lna)

    def rhonu(self, a, amnu):
        """``rho_nu / rho_nu(massless)`` of one flavour, 1 when massless."""
        total = 1.0
        if amnu > 0.0:
            total = 0.0
            for i in range(len(self.nu_q)):
                total += self.nu_w[i] * math.sqrt(1.0 + (a * amnu / self.nu_q[i]) ** 2)
        return total

    def fHe(self, YHe):
        return YHe / (const_mHe_mH * (1.0 - YHe))

    def x_H(self, thermo):
        return min(max(thermo[0], 0.0), 1.0)

    def x_He(self, thermo):
        return min(max(thermo[1], 0.0), 2.0)

    def Tmat(self, thermo):
        return abs(thermo[2])

    def xe_rec(self, x_H, x_He, fHe):
        """Free electrons per hydrogen nucleus from recombination alone."""
        return x_H + fHe * x_He

    def xe_reio(self, inv_a, xe_rec, fHe, z_re):
        """CAMB's reionization step, zero when ``z_re = 0``."""
        z = inv_a - 1.0
        y = math.sqrt(1.0 + z) * (1.0 + z)
        y_re = math.sqrt(1.0 + z_re) * (1.0 + z_re)
        delta_y = 0.75 * math.sqrt(1.0 + z_re)
        step = 0.5 * (1.0 + tanh_((y_re - y) / delta_y))
        he3 = 0.5 * (1.0 + tanh_((3.5 - z) / 0.5))
        on = min(max(z_re, 0.0), 1.0)
        return ((1.0 + fHe - xe_rec) * step + fHe * he3) * on

    def xe(self, xe_rec, xe_reio):
        return xe_rec + xe_reio

    def akthom(self, YHe, Omegab, H0):
        return AKTHOM_PER_OMEGAB_H0SQ * (1.0 - YHe) * Omegab * H0**2

    def opacity(self, xe, akthom, inv_a2):
        return max(xe * akthom * inv_a2, 1.0e-30)

    def cs2(self, Tmat, xe_rec, YHe, dTm_dz, inv_a):
        """``c_s^2 = k_B T_m / (mu m_H c^2) (1 - (1/3) d ln T_m / d ln a)``."""
        mu = 1.0 / (1.0 + (1.0 / const_mHe_mH - 1.0) * YHe + (1.0 - YHe) * xe_rec)
        dTm_dlna = -dTm_dz * inv_a
        return max(const_kB / const_mH / const_c**2 / mu * Tmat * (3.0 - dTm_dlna / Tmat) / 3.0, 0.0)

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
        return peebles_K(Hz, zred)

    def dTm_dz(self, Tmat, xe_rec, fHe, zred, Hz, H0, Tcmb):
        return matter_temperature_rate(Tmat, xe_rec, fHe, zred, Hz, H0, Tcmb)

    def d_lna(self, adotoa):
        return adotoa

    def d_thermo(self, x_H, x_He, xe_rec, Tmat, zred, a, n_H, Hz, Tcmb, Nnow, fHe, K_peebles, dTm_dz, adotoa, inv_a, output):
        dz_dtau = -adotoa * inv_a
        output[0] = hydrogen_rate(x_H, xe_rec, Tmat, zred, n_H, Hz, Tcmb, Nnow, K_peebles) * dz_dtau
        output[1] = helium_rate(x_H, x_He, xe_rec, Tmat, zred, a, n_H, Hz, Tcmb, Nnow, fHe) * dz_dtau
        output[2] = dTm_dz * dz_dtau

    def d_kappa(self, opacity):
        return opacity
