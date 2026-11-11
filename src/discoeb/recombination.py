####################################################################################################################
# This file is not licensed under the GNU GPL license!
# Large portions of this code, which is part of the DISCO-EB module have been adapted from the recfast code base.
# The original recfast code can be found at: https://www.astro.ubc.ca/people/scott/recfast.html
# The recfast code base is distributed under the following license:
####################################################################################################################
# Integrator for Cosmic Recombination of Hydrogen and Helium,
# developed by Douglas Scott (dscott@astro.ubc.ca)
# based on calculations in the papers Seager, Sasselov & Scott
# (ApJ, 523, L1, 1999; ApJS, 128, 407, 2000)
# and "fudge" updates in Wong, Moss & Scott (2008).
#
# Permission to use, copy, modify and distribute without fee or royalty at
# any tier, this software and its documentation, for any purpose and without
# fee or royalty is hereby granted, provided that you agree to comply with
# the following copyright notice and statements, including the disclaimer,
# and that the same appear on ALL copies of the software and documentation,
# including modifications that you make for internal use or for distribution:
#
# Copyright 1999-2010 by University of British Columbia.  All rights reserved.
####################################################################################################################
# This file contains a numba-CUDA version of the recfast equations, adapted for use in the DISCO-EB module.
# The DISCO-EB module is distributed under the GNU GPL license.
####################################################################################################################

"""The recombination history as states of each wave mode's trajectory.

RECFAST's ``(x_H, x_He, T_m)`` (:mod:`discoeb.thermodynamics_recfast`, statement
for statement), ``ln a`` from the model's own Friedmann sum and the optical depth
``kappa = int kappa' dtau`` ride in front of the Einstein-Boltzmann hierarchy as
five more states of the joint system, and :class:`Recombination` provides the
hierarchy with ``a``, the Thomson opacity, ``c_s^2`` and the massive-neutrino
density ratio from them. These states read nothing of the hierarchy, so the
joint model is a modax hierarchy of two blocks (:func:`discoeb.perturbations.joint_kernel`):
the recombination block steps on its own ahead of the hierarchy and publishes
the four quantities, which modax hands over as quintic polynomials of time
between its steps. Nothing is tabulated and no device buffer is refilled: the
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
    CB1,
    CDB,
    CK,
    CL,
    CR,
    CT,
    DZ_H_SAHA_BLEND,
    FUDGE_H,
    NQ_NU_BACKGROUND,
    T_0,
    T_1,
    Z_H_SAHA_END,
    A2P_s,
    A2P_t,
    AGauss1,
    AGauss2,
    Bfact,
    CB1_He1,
    CDB_He,
    CK_He,
    CL_He,
    CL_PSt,
    H_frac,
    L_He2St_ion,
    L_He_2p,
    L_He_2Pt,
    Lambda,
    Lambda_He,
    a_PPB,
    a_trip,
    a_VF,
    b_PPB,
    b_trip,
    b_VF,
    bigH,
    c_PPB,
    const_c,
    const_c_Mpc_s,
    const_dens_fac,
    const_EionHe12s,
    const_h,
    const_h_c_L_He_2St_kB,
    const_kB,
    const_mH,
    const_mHe_mH,
    d_PPB,
    sigma_He_2Ps,
    sigma_He_2Pt,
    wGauss1,
    wGauss2,
    zGauss1,
    zGauss2,
)

THERMAL_PARAMETERS = (
    "YHe_nominal",
    "Tcmb_nominal",
    "H0_nominal",
    "Omegab_nominal",
    "z_re",
)
"""What the recombination rates and reionization read of the cosmology, as
columns of the parameter row after the hierarchy's own
(:data:`discoeb.perturbations.PARAM_COLUMNS`). ``z_re = 0`` means no reionization."""

THERMO_ERROR_WEIGHT = float(os.environ.get("DISCO_THERMO_WEIGHT", "1.0"))
"""Weight of ``ln a`` and the three RECFAST states in the error norm of the
recombination block's own steps (their error is held to ``rtol / weight``).
The block steps ahead of the hierarchy on steps of its own, so the weight sets
the thermal state's accuracy and the number of RECFAST evaluations, not the
hierarchy's steps; ``P(k)`` is unchanged from 0.3 to 100, and 0 leaves the
block's steps uncontrolled."""

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
term they weight matters. The scale factor the recombination block publishes
to the hierarchy stays double: single precision rounds ``ln a`` and makes ``a``
a staircase in the parameters whose quintic interpolant the hierarchy reads
(:meth:`Recombination.a`). The powers of the reionization step are single
precision: in double they cost 7 percent of a P(k) batch, and the step matters
only where the opacity is too small for its rounding to show."""

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


def saha_slope(z, Tcmb, Nnow, E_ion, offset, g_factor, f_factor):
    """``d x / d z`` of the Saha ionization fraction at ``z`` (:func:`thermodynamics_recfast.Saha`).

    One exit: numba-cuda-mlir gives a function returning from several branches
    a result slot, and inlining it at two call sites of one caller merges the
    two slots into a memref phi that LLVM's typed-pointer verifier rejects.
    """

    T = Tcmb * (1.0 + z)
    CR_T = CR * T
    phi = CR_T * math.sqrt(CR_T) / Nnow / (1.0 + z) ** 3
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
        slope = (2.0 * f_factor * dR * denom - 2.0 * beta * (droot + dR)) / (
            denom * denom
        )
    return slope


def saha_fraction(z, Tcmb, Nnow, E_ion, offset, g_factor, f_factor):
    """The Saha ionization fraction at ``z`` whose rate :func:`saha_slope` is.

    Fully ionized (``f_factor``) where :func:`saha_slope` is zero at high
    temperature, ``2 beta / (sqrt(alpha^2 + 4 beta) + alpha)`` below; a state
    integrated from full ionization with :func:`saha_slope` follows it. One exit.
    """

    T = Tcmb * (1.0 + z)
    CR_T = CR * T
    phi = CR_T * math.sqrt(CR_T) / Nnow / (1.0 + z) ** 3
    R = g_factor * phi * exp_(-E_ion / T)
    frac = f_factor
    if T <= E_ion * 0.1:
        alpha = R + offset
        beta = R * f_factor
        frac = 2.0 * beta / (math.sqrt(alpha * alpha + 4.0 * beta) + alpha)
    return frac


def hydrogen_rate(x_H, xe, Tmat, z, n_H, Hz, Tcmb, Nnow, K_peebles):
    """``d x_H / d z``: Saha above ``Z_H_SAHA_END``, blended into the rate equation below it."""

    saha = saha_slope(z, Tcmb, Nnow, CB1, 0.0, 1.0, 1.0)
    f0 = saha
    if z < Z_H_SAHA_END:
        fu = FUDGE_H
        one_minus_x_H = 1.0 - x_H
        Tmat_1e4 = Tmat / 1.0e4
        Rdown = (
            1.0e-19
            * a_PPB
            * pow_(Tmat_1e4, b_PPB)
            / (1.0 + c_PPB * pow_(Tmat_1e4, d_PPB))
        )
        CR_Tmat = CR * Tmat
        Rup = Rdown * (CR_Tmat * math.sqrt(CR_Tmat)) * exp_(-CDB / Tmat)
        rate_diff = xe * x_H * n_H * Rdown - Rup * one_minus_x_H * exp_(-CL / Tmat)
        K_Lambda_n_1_minus_x_H = K_peebles * Lambda * n_H * one_minus_x_H
        denom_term = (
            1.0 / fu
            + K_Lambda_n_1_minus_x_H / fu
            + K_peebles * Rup * n_H * one_minus_x_H
        )
        f0 = rate_diff * (1.0 + K_Lambda_n_1_minus_x_H) / (Hz * (1.0 + z) * denom_term)
        if z > Z_H_SAHA_END - 2.0 * DZ_H_SAHA_BLEND:
            s_blend = (Z_H_SAHA_END - z) / (2.0 * DZ_H_SAHA_BLEND)
            w_full = s_blend * s_blend * (3.0 - 2.0 * s_blend)
            f0 = w_full * f0 + (1.0 - w_full) * saha
    return f0


def helium_rate(x_H, x_He, xe, Tmat, z, a, n_H, Hz, Tcmb, Nnow, fHe):
    """``d x_He / d z``: both Saha stages while their fraction exceeds 0.9, the singlet and triplet rates below.

    The switch is RECFAST's ``x_He > 0.9``, tested on the Saha fraction at
    ``z`` rather than on the state. While the Saha slope drives it the state
    is that fraction, so the switch falls at the same redshift; but a state
    test can switch back, and a lane whose step landed just below 0.9 was
    pulled back across it and crept along ``x_He = 0.9`` for a couple of
    hundred tiny steps (a few lanes of most warps, at 40 percent of the
    recombination block's cost). On the fraction it is a function of ``z``
    alone, the treatment hydrogen's switch has (CLASS 3 does the same).

    The rate equation is called from one place only: a helper inlined at two
    call sites, or a local assigned in several branches, gives numba-cuda-mlir
    a memref phi that LLVM rejects.
    """

    saha = (
        saha_slope(z, Tcmb, Nnow, CB1_He1, 1.0, 4.0, fHe)
        + saha_slope(z, Tcmb, Nnow, const_EionHe12s, 1.0 + fHe, 1.0, fHe)
    ) / fHe
    saha_x = (
        saha_fraction(z, Tcmb, Nnow, CB1_He1, 1.0, 4.0, fHe)
        + saha_fraction(z, Tcmb, Nnow, const_EionHe12s, 1.0 + fHe, 1.0, fHe)
    ) / fHe
    f1 = 0.0
    if saha_x > 0.9 or a < 3.0e-4:
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
    CR_Tmat = CR * Tmat
    CR_Tmat_15 = CR_Tmat * math.sqrt(CR_Tmat)
    sq_0 = math.sqrt(Tmat / T_0)
    sq_1 = math.sqrt(Tmat / T_1)
    Rdown_He = a_VF / (
        sq_0 * pow_(1.0 + sq_0, 1.0 - b_VF) * pow_(1.0 + sq_1, 1.0 + b_VF)
    )
    Rup_He = 4.0 * Rdown_He * CR_Tmat_15 * exp_(-CDB_He / Tmat)
    He_Boltz = math.exp(min(680.0, Bfact / Tmat))
    Rdown_trip = a_trip / (
        sq_0 * pow_(1.0 + sq_0, 1.0 - b_trip) * pow_(1.0 + sq_1, 1.0 + b_trip)
    )
    Rup_trip = (
        Rdown_trip
        * exp_(-const_h * const_c * L_He2St_ion / (const_kB * Tmat))
        * CR_Tmat_15
        * (4.0 / 3.0)
    )

    tauHe_s = A2P_s * CK_He * 3.0 * n_He_1_minus_x_He / Hz
    pHe_s = (1.0 - exp_(-tauHe_s)) / tauHe_s
    Doppler_term = math.sqrt(
        2.0 * const_kB * Tmat / (const_mH * const_mHe_mH * const_c**2)
    )
    if x_H < 0.9999999:
        Doppler_2p = const_c * L_He_2p * Doppler_term
        gamma_2Ps = (
            3.0
            * A2P_s
            * fHe
            * one_minus_x_He
            * const_c**2
            / (
                math.sqrt(math.pi)
                * sigma_He_2Ps
                * 8.0
                * math.pi
                * Doppler_2p
                * one_minus_x_H
                * (const_c * L_He_2p) ** 2
            )
        )
        AHcon = A2P_s / (1.0 + 0.36 * pow_(gamma_2Ps, 0.86))
        K_He = 1.0 / ((A2P_s * pHe_s + AHcon) * 3.0 * n_He_1_minus_x_He)
    else:
        K_He = 1.0 / (A2P_s * pHe_s * 3.0 * n_He_1_minus_x_He)

    rate_diff_He = xe * x_He * n_H * Rdown_He - Rup_He * one_minus_x_He * exp_(
        -CL_He / Tmat
    )
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
            3.0
            * A2P_t
            * fHe
            * one_minus_x_He
            * const_c**2
            / (
                math.sqrt(math.pi)
                * sigma_He_2Pt
                * 8.0
                * math.pi
                * Doppler_2Pt
                * one_minus_x_H
                * (const_c * L_He_2Pt) ** 2
            )
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
    return (
        CK
        / Hz
        * (
            1.0
            + AGauss1 * exp_(-(((log_z_term - zGauss1) / wGauss1) ** 2))
            + AGauss2 * exp_(-(((log_z_term - zGauss2) / wGauss2) ** 2))
        )
    )


def matter_temperature_rate(Tmat, xe, fHe, zred, Hz, H0, Tcmb):
    """RECFAST's matter-temperature equation ``dT_m/dz``, ``T_m`` locked to ``T_rad`` in tight Compton coupling."""

    z_term = 1.0 + zred
    inv_z = 1.0 / z_term
    Trad = Tcmb * z_term
    compton = CT * Trad**4 * xe / (1.0 + xe + fHe)  # 1 / timeTh
    inv_timeH = 1.5 * (H0 / 100.0) * bigH * pow_(z_term, 1.5)
    locked = Tmat * inv_z
    full = compton * (Tmat - Trad) / Hz * inv_z + 2.0 * locked
    w_full = 1.0 if inv_timeH >= H_frac * compton else 0.0  # timeTh >= H_frac timeH
    return w_full * full + (1.0 - w_full) * locked


def logistic(x):
    """``(1 + tanh x) / 2 = 1 / (1 + e^{-2x})``, a smooth step.

    Through ``exp`` (Enzyme has no forward rule for libdevice's ``tanh``) in
    single precision where the rates' transcendentals are. The argument is
    capped at 30: the step is then ``1e-13`` instead of ``0`` where it is off,
    and the exponential's derivatives, which the sensitivity sweeps multiply by
    tangents of order a thousand in single precision, stay finite (capped at
    80, ``e^80 = 5e34`` sat at the edge of single precision and the
    second-order sweep overflowed it to infinity and the optical depth's
    sensitivity to NaN).
    """

    return 1.0 / (1.0 + exp_(min(-2.0 * x, 30.0)))


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
    :mod:`discocmb.cmb` forms the visibility as ``kappa'(tau) exp(-(kappa(tau_0) -
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
        """``exp(ln a)`` in double precision: what the hierarchy reads must be smooth in the parameters.

        In single precision ``ln a`` itself is rounded first, so ``a`` is a
        staircase of relative steps 5e-7 while its derivative (Enzyme's,
        through the cast) is smooth; the hierarchy reads the published value
        and rates as one quintic, and the staircase made the spectra jump by
        1e-5 between neighbouring parameter values whatever the tolerance.
        Published once per recombination step, the double-precision ``exp``
        costs nothing measurable.
        """
        return math.exp(lna)

    def rhonu(self, a, amnu):
        """``rho_nu / rho_nu(massless)`` of one flavour, 1 when massless."""
        total = 1.0
        if amnu > 0.0:
            total = 0.0
            for i in range(len(self.nu_q)):
                total += self.nu_w[i] * math.sqrt(1.0 + (a * amnu / self.nu_q[i]) ** 2)
        return total

    def pnu(self, a, amnu):
        """``p_nu / rho_nu(massless)`` of one flavour: ``(rhonu - d rhonu / d ln a) / 3``, 1/3 when massless."""
        total = 1.0 / 3.0
        if amnu > 0.0:
            total = 0.0
            for i in range(len(self.nu_q)):
                total += self.nu_w[i] / math.sqrt(1.0 + (a * amnu / self.nu_q[i]) ** 2)
            total *= 1.0 / 3.0
        return total

    # The inputs a derivative may move are read through the trajectory's
    # direction, ``x = x_nominal + eps dir_x`` (:data:`discoeb.perturbations.IX_EPS`):
    # ``eps`` is zero in every primal solve, and ``d/d eps`` is the derivative
    # along ``dir``.

    def YHe(self, YHe_nominal, eps, dir_YHe):
        return YHe_nominal + eps * dir_YHe

    def Tcmb(self, Tcmb_nominal, eps, dir_Tcmb):
        return Tcmb_nominal + eps * dir_Tcmb

    def H0(self, H0_nominal, eps, dir_H0):
        return H0_nominal + eps * dir_H0

    def Omegab(self, Omegab_nominal, eps, dir_Omegab):
        return Omegab_nominal + eps * dir_Omegab

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
        # double precision, as for ``a``: the opacity is published to the hierarchy
        y = pow_(1.0 + z, 1.5)
        y_re = pow_(1.0 + z_re, 1.5)
        delta_y = 0.75 * pow_(1.0 + z_re, 0.5)
        step = logistic((y_re - y) / delta_y)
        he3 = logistic((3.5 - z) / 0.5)
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
        inv_mu = 1.0 + (1.0 / const_mHe_mH - 1.0) * YHe + (1.0 - YHe) * xe_rec
        dTm_dlna = -dTm_dz * inv_a
        return max(
            const_kB / (3.0 * const_mH * const_c**2) * inv_mu * (3.0 * Tmat - dTm_dlna),
            0.0,
        )

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

    def d_thermo(
        self,
        x_H,
        x_He,
        xe_rec,
        Tmat,
        zred,
        a,
        n_H,
        Hz,
        Tcmb,
        Nnow,
        fHe,
        K_peebles,
        dTm_dz,
        adotoa,
        inv_a,
        output,
    ):
        dz_dtau = -adotoa * inv_a
        output[0] = (
            hydrogen_rate(x_H, xe_rec, Tmat, zred, n_H, Hz, Tcmb, Nnow, K_peebles)
            * dz_dtau
        )
        output[1] = (
            helium_rate(x_H, x_He, xe_rec, Tmat, zred, a, n_H, Hz, Tcmb, Nnow, fHe)
            * dz_dtau
        )
        output[2] = dTm_dz * dz_dtau

    def d_kappa(self, opacity):
        return opacity


_ALONE: list = []


def recombination_model():
    """:class:`Recombination` as a model of its own, for evaluating its providers on the host.

    The reionization fraction, the opacity and the sound speed are then one
    piece of code wherever they are needed: the device kernel, the background
    solve for ``z_re`` and the diagnostics' thermal histories.
    """

    from .component import Model

    if not _ALONE:
        _ALONE.append(Model([Recombination()]))
    return _ALONE[0]
