####################################################################################################################
# This file is not licensed under the MIT license that covers the rest of DISCO-EB!
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
# The rest of the DISCO-EB module is distributed under the MIT license.
####################################################################################################################
"""
The thermal recombination history as calculated by RECFAST, which is reimplemented
here in pure Python, based on the original code by Douglas Scott.
"""

from __future__ import annotations

import math
import os

from jax.typing import ArrayLike

from ._component import Component, Parameter, State, derivative, float32, float64
from .gravity import H100_SQ

# RECFAST's physical and atomic constants (PDG 2023 where noted).
const_mH = 1.67353284e-27  # H atom mass [kg], PDG 2023
const_me = 9.1093837015e-31  # Electron mass [kg], PDG 2023
const_mHe_mH = 3.97146570884  # Helium / Hydrogen mass ratio
const_c = 2.99792458e08  # Speed of light [m/s]
const_h = 6.62607015e-34  # Planck's constant [Js], PDG 2023
const_kB = 1.380649e-23  # Boltzmann's constant [J/K], PDG 2023
const_sigma = 6.6524587321e-29  # Thomson scattering cross section [m^2], PDG 2023
const_arad = 4 * 5.670374419e-8 / const_c  # Radiation constant [Ws/m^3/K^4], PDG 2023

const_dens_fac = (
    # 3 bigH^2 / (8 pi G m_H), bigH = 100 km/s/Mpc in 1/s, G = 6.67430e-11 m^3/kg/s^2
    11.223810928601939
)

const_c_Mpc_s = 9.71561189e-15  # Mpc/s

# The comments on the constants below follow the original RECFAST code.

const_EionHe12s = 6.314878282674e5  # HeII 1s ionization energy in [K]

# Two-photon rates and atomic levels
Lambda = 8.2245809  # H 2s-1s two photon rate
Lambda_He = 51.3  # HeI 2s-1s two photon rate
L_H_ion = 1.096787737e7  # level for H ionization in m^-1
L_H_alpha = 8.225916453e6  # level for H Ly alpha in m^-1
L_He1_ion = 1.98310772e7  # level for HeI ionization in m^-1
L_He_2s = 1.66277434e7  # level for HeI 2s in m^-1
L_He_2p = 1.71134891e7  # level for He 2p (21P1-11S0) in m^-1

# Atomic data for HeI
A2P_s = 1.798287e9  # Einstein A coefficient for He 21P1-11S0
A2P_t = 177.58  # Einstein A coefficient for He 23P1-11S0
L_He_2Pt = 1.690871466e7  # level for 23P012-11S0 in m^-1
L_He_2St = 1.5985597526e7  # level for 23S1-11S0 in m^-1
L_He2St_ion = 3.8454693845e6  # level for 23S1-continuum in m^-1
sigma_He_2Ps = 1.436289e-22  # H ionization x-section at HeI 21P1-11S0 freq. in m^2
sigma_He_2Pt = 1.484872e-22  # H ionization x-section at HeI 23P1-11S0 freq. in m^2

# Gaussian corrections to the hydrogen fudge (Rubino-Martin et al. 2010) at
# the RECFAST 1.5.2 values, as in CLASS (recfast_AGauss2, recfast_zGauss2),
# CAMB and the Planck 2018 analysis. RECFAST 1.5.0's values (AGauss2 = 0.05,
# zGauss2 = 6.75, fudge 1.105) put x_e 1.8 percent above CLASS after
# recombination and TT 0.14 percent low at l = 300-2000.
AGauss1 = -0.14  # amplitude of the 1st Gaussian for the H fudging
AGauss2 = 0.079  # amplitude of the 2nd Gaussian for the H fudging
zGauss1 = 7.28  # ln(1+z) central value of the 1st Gaussian
zGauss2 = 6.73  # ln(1+z) central value of the 2nd Gaussian
wGauss1 = 0.18  # width of the 1st Gaussian
wGauss2 = 0.33  # width of the 2nd Gaussian
FUDGE_H = 1.125  # hydrogen Peebles fudge with the Gaussian corrections (RECFAST 1.5.2: 1.14 - 0.015)

H_frac = 1.0e-3

Lalpha = 1.0 / L_H_alpha  # Ly alpha wavelength in SI units
Lalpha_He = 1.0 / L_He_2p  # Helium I 2p-1s wavelength in SI units
DeltaB = (
    const_h * const_c * (L_H_ion - L_H_alpha)
)  # energy of first excited state from continuum = 3.4eV
CDB = DeltaB / const_kB  # DeltaB / k_B in K
DeltaB_He = (
    const_h * const_c * (L_He1_ion - L_He_2s)
)  # energy of first excited state from cont. for He = 3.4eV
CDB_He = DeltaB_He / const_kB  # DeltaB_He / k_B (n = 2 to continuum) in K
CB1 = (
    const_h * const_c * L_H_ion / const_kB
)  # H ionization energy / k_B in K (13.6 eV, 4 CDB)
CB1_He1 = const_h * const_c * L_He1_ion / const_kB  # CB1 for HeI ionization potential

CR = 2.0 * math.pi * (const_me / const_h) * (const_kB / const_h)
CK = Lalpha**3 / (8.0 * math.pi)
CK_He = Lalpha_He**3 / (8.0 * math.pi)
CL = const_c * const_h / (const_kB * Lalpha)
CL_He = const_c * const_h / (const_kB / L_He_2s)
CT = (8.0 / 3.0) * (const_sigma / (const_me * const_c)) * const_arad
Bfact = const_h * const_c * (L_He_2p - L_He_2s) / const_kB
CL_PSt = const_h * const_c * (L_He_2Pt - L_He_2St) / const_kB
const_h_c_L_He_2St_kB = const_h * const_c * L_He_2St / (const_kB)

# Pequignot, Petitjean & Boisson fit parameters for hydrogen
a_PPB = 4.309
b_PPB = -0.6166
c_PPB = 0.6703
d_PPB = 0.5300

# Verner & Ferland type fit parameters for helium
a_VF = 10.0 ** (-16.744)
b_VF = 0.711
T_0 = 10 ** (0.477121)
T_1 = 10 ** (5.114)

# HeI triplet recombination rate fit parameters
a_trip = 10.0 ** (-16.306)
b_trip = 0.761

Z_H_SAHA_END = 1600.0
"""Redshift below which hydrogen follows the rate equation instead of Saha
equilibrium (CLASS's ``recfast_z_full_H_recombination``)."""

DZ_H_SAHA_BLEND = 50.0
"""Half-width in redshift of the blend from the Saha slope into the rate
equation (CLASS's ``recfast_delta_z_full_H_recombination``)."""

THERMO_ERROR_WEIGHT = float(os.environ.get("DISCO_THERMO_WEIGHT", "1.0"))
"""Weight of ``ln a`` and the three RECFAST states in the error norm of the
recombination block's steps, which keep their error below ``rtol / weight``.
The block takes its own steps ahead of the hierarchy, so the weight sets the
thermal state's accuracy and the number of RECFAST evaluations, not the
hierarchy's steps. ``P(k)`` does not change between 0.3 and 100. A weight of
0 leaves the block's steps uncontrolled."""

REIO_ERROR_WEIGHT = 1.0e4
"""Weight of ``kappa_reio`` (:class:`ReionizationDepth`) in the error norm of a sensitivity launch.

Reionization's step is 0.5 wide in ``z``, ~150 Mpc of conformal time. At
the block's weight the steps across it resolve ``tau_reio`` (to 2e-4) but
not its derivative in ``z_re``, which the sensitivities carry without error
control: ``d tau_reio / d z_re`` came out 6 percent high and ``dP/dz_re``
3 percent off. ``kappa_reio`` changes only there, so its weight refines
those steps alone; at 1e4 the derivative is converged. It costs a P(k)
batch with reionization 14 percent, so primal launches go without it
(2026-11-16). The state itself costs 4 to 7 percent of a solve, so it is a
component a model adds, not one of every model's."""

AKTHOM_PER_OMEGAB_H0SQ = 2.3038921003709498e-9
"""``n_H sigma_T c / (H_0 units)``: ``kappa' = x_e akthom / a^2`` with
``akthom = this (1 - Y_He) Omega_b H_0^2``."""

AKTHOM_PER_OMEGAB = 1.0e4 * AKTHOM_PER_OMEGAB_H0SQ
DENS_FAC_PER_OMEGAB = const_dens_fac
"""The same and the hydrogen density's factor per unit ``omega_b = Omega_b
h^2``, the baryon density the kernel holds."""

FP32_MATH = os.environ.get("DISCO_THERMO_FP32_MATH", "1") == "1"
"""Evaluate the exponentials, fractional powers and logarithms of RECFAST's
rates in single precision. On by default, ``DISCO_THERMO_FP32_MATH=0`` keeps
them double. The equations stay the same, the functions are good to 1e-7
instead of 1e-16, and they run on a card whose fp64 transcendentals run at
1/64 rate. This applies only where the argument range allows it.
``(CR T)^{3/2}`` reaches 1e42 and the Boltzmann factor ``e^{680}``, so those
stay double. The ``e^{-E/T}`` factors stay inside fp32's range wherever the
term they weight matters. The scale factor the block publishes to the
hierarchy stays double: single precision rounds ``ln a`` and turns ``a`` into
a staircase in the parameters, and the hierarchy reads its quintic
interpolant (:meth:`Recombination.a`). The powers of the reionization step
are single precision. In double they cost 7 percent of a P(k) batch, and the
step matters only where the opacity is too small for its rounding to show."""

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


def saha_slope(z, T_cmb, Nnow, E_ion, offset, g_factor, f_factor):
    """``d x / d z`` of the Saha ionization fraction at ``z`` (RECFAST's Saha equation).

    Has a single return on purpose. numba-cuda-mlir gives a function that
    returns from several branches a result slot. Inlining it at two call
    sites of one caller merges the two slots into a memref phi, which LLVM's
    typed-pointer verifier rejects.
    """

    T = T_cmb * (1.0 + z)
    CR_T = CR * T
    phi = CR_T * math.sqrt(CR_T) / Nnow / (1.0 + z) ** 3
    R = g_factor * phi * exp_(-E_ion / T)
    slope = 0.0
    if R > 1.0e-20 and T <= E_ion * 0.1:
        alpha = R + offset
        beta = R * f_factor
        dR = R * (-1.5 / (1.0 + z) + E_ion / (T_cmb * (1.0 + z) ** 2))
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


def saha_fraction(z, T_cmb, Nnow, E_ion, offset, g_factor, f_factor):
    """The Saha ionization fraction at ``z``, the integral of :func:`saha_slope`.

    Returns ``f_factor`` (fully ionized) at high temperature, where
    :func:`saha_slope` is zero, and ``2 beta / (sqrt(alpha^2 + 4 beta) +
    alpha)`` below. A state integrated from full ionization with
    :func:`saha_slope` follows it. Single return, as in :func:`saha_slope`.
    """

    T = T_cmb * (1.0 + z)
    CR_T = CR * T
    phi = CR_T * math.sqrt(CR_T) / Nnow / (1.0 + z) ** 3
    R = g_factor * phi * exp_(-E_ion / T)
    frac = f_factor
    if T <= E_ion * 0.1:
        alpha = R + offset
        beta = R * f_factor
        frac = 2.0 * beta / (math.sqrt(alpha * alpha + 4.0 * beta) + alpha)
    return frac


def hydrogen_rate(x_H, xe, Tmat, z, n_H, Hz, T_cmb, Nnow, K_peebles):
    """``d x_H / d z``: Saha above ``Z_H_SAHA_END``, blended into the rate equation below it."""

    saha = saha_slope(z, T_cmb, Nnow, CB1, 0.0, 1.0, 1.0)
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


def helium_rate(x_H, x_He, xe, Tmat, z, a, n_H, Hz, T_cmb, Nnow, fHe):
    """``d x_He / d z``: both Saha stages while their fraction exceeds 0.9, the singlet and triplet rates below.

    The switch is RECFAST's ``x_He > 0.9``, tested on the Saha fraction at
    ``z`` instead of the state. While the Saha slope drives the state, the
    state equals that fraction, so the switch falls at the same redshift. A
    test on the state can switch back, though. A lane whose step landed just
    below 0.9 got pulled back across and crept along ``x_He = 0.9`` for a
    couple of hundred tiny steps (a few lanes of most warps, 40 percent of
    the recombination block's cost). Tested on the fraction, the switch
    depends on ``z`` alone, like hydrogen's (CLASS 3 does the same).

    Only one place calls the rate equation. A helper inlined at two call
    sites, or a local assigned in several branches, gives numba-cuda-mlir a
    memref phi that LLVM rejects.
    """

    saha = (
        saha_slope(z, T_cmb, Nnow, CB1_He1, 1.0, 4.0, fHe)
        + saha_slope(z, T_cmb, Nnow, const_EionHe12s, 1.0 + fHe, 1.0, fHe)
    ) / fHe
    saha_x = (
        saha_fraction(z, T_cmb, Nnow, CB1_He1, 1.0, 4.0, fHe)
        + saha_fraction(z, T_cmb, Nnow, const_EionHe12s, 1.0 + fHe, 1.0, fHe)
    ) / fHe
    f1 = 0.0
    if saha_x > 0.9 or a < 3.0e-4:
        f1 = saha
    elif x_He >= 1.0e-8:
        f1 = helium_rate_equation(x_H, x_He, xe, Tmat, z, n_H, Hz, fHe)
    return f1


def helium_rate_equation(x_H, x_He, xe, Tmat, z, n_H, Hz, fHe):
    """The singlet and triplet rate equations of HeI recombination, ``d x_He / d z``.

    ``1 - x_He`` has a floor of 1e-6. The equations hold for ``x_He < 1``
    (at the handover from Saha it is 1e-3 or more), and a solver stage that
    probes ``x_He >= 1`` makes the escape probabilities ``(1 - exp(-tau)) /
    tau`` overflow or 0/0. That NaN stalled one sweep cosmology's block at
    ``max_steps`` once a sixth state changed its step sequence (2026-11-16).
    """

    Hz_z_term = Hz * (1.0 + z)
    n_He = fHe * n_H
    one_minus_x_He = max(1.0 - x_He, 1.0e-6)
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


def matter_temperature_rate(Tmat, xe, fHe, zred, Hz, T_cmb):
    """RECFAST's matter-temperature equation ``dT_m/dz``, ``T_m`` locked to ``T_rad`` in tight Compton coupling.

    The Hubble time of the switch is the model's own, ``2 / (3 H)``.
    RECFAST estimates it as ``2 / (3 H_0 (1+z)^1.5)``, which needs ``H_0``
    ahead of the solve.
    """

    z_term = 1.0 + zred
    inv_z = 1.0 / z_term
    Trad = T_cmb * z_term
    compton = CT * Trad**4 * xe / (1.0 + xe + fHe)  # 1 / timeTh
    inv_timeH = 1.5 * Hz
    locked = Tmat * inv_z
    full = compton * (Tmat - Trad) / Hz * inv_z + 2.0 * locked
    w_full = 1.0 if inv_timeH >= H_frac * compton else 0.0  # timeTh >= H_frac timeH
    return w_full * full + (1.0 - w_full) * locked


def logistic(x):
    """``(1 + tanh x) / 2 = 1 / (1 + e^{-2x})``, a smooth step.

    Goes through ``exp`` because Enzyme has no forward rule for libdevice's
    ``tanh``, and runs in single precision when the rates' transcendentals
    do. The exponent is capped at 30, so the step reads ``1e-13`` instead of
    ``0`` where it is off. The cap keeps the exponential's derivatives finite,
    which the sensitivity sweeps multiply by tangents of order a thousand in
    single precision. A cap of 80 (``e^80 = 5e34``) sat at the edge of single
    precision: the second-order sweep overflowed it to infinity and turned the
    optical depth's sensitivity into NaN.
    """

    return 1.0 / (1.0 + exp_(min(-2.0 * x, 30.0)))


# The component


def _thermo_initial(a, T_cmb, output):
    """Fully ionized and in Compton equilibrium: ``x_H = 1``, ``x_He = 2``, ``T_m = T_cmb / a``."""

    output[0] = 1.0
    output[1] = 2.0
    output[2] = T_cmb / a


def _tau_initial(aH, rho_matter, a):
    """The conformal time at the start, deep in radiation domination.

    With radiation and matter alone, ``a'^2 = (R + M a) / 3`` integrates to
    ``tau = 2 sqrt(3) a / (sqrt(R + M a) + sqrt(R))``, which is ``1 / aH``
    times ``2 sqrt(1 + e) / (sqrt(1 + e) + 1)`` with ``e = M a / R`` the
    matter-to-radiation ratio. The model's own ``aH`` carries every
    species, so the radiation is what it leaves beside the matter (massive
    neutrinos included, relativistic then), and the ratio, 1e-4 at the
    start, only corrects it.
    """

    matter = H100_SQ * a * a * rho_matter
    root = math.sqrt(1.0 + matter / (aH * aH - matter))
    return 2.0 * root / (root + 1.0) / aH


class Recombination(Component):
    """The conformal time, RECFAST's ``(x_H, x_He, T_m)`` and the optical depth as driver states.

    The solve integrates in ``ln a``, so ``a`` is the time itself and the
    conformal time ``tau`` a state whose rate is one (per unit of itself:
    the solve divides every rate by ``a'``,
    :class:`~discoeb.components._component.Equations`). It starts from the
    radiation era's closed form, and its value today is the conformal age.
    ``thermo`` holds RECFAST's three states,
    whose rates in ``z`` convert with ``dz/dtau = -aH / a``. ``kappa`` is
    ``int_{tau_start}^tau kappa' dtau``, from which the hook of
    :mod:`discocmb.cmb` forms the visibility
    ``kappa'(tau) exp(-(kappa(tau_0) - kappa(tau)))``. The hierarchy reads
    ``a``, ``opacity``
    (``kappa' = x_e n_H sigma_T a``, with reionization) and ``cs2`` (from
    ``T_m`` and its own rate, no spline derivative). Nothing marks these
    states as the background: their rates read no wave number, and
    :meth:`Equations.stages <discoeb.components._component.Equations.stages>` finds that.
    """

    tau = State(initial=_tau_initial)
    thermo = State(3, initial=_thermo_initial)
    kappa = State()

    # The rates read Y_He, T_cmb and omega_b, which the species declare.
    z_re: ArrayLike = Parameter(0.0, column=True)
    """The redshift half way through CAMB's reionization step (0: none)."""

    def fHe(self, Y_He):
        return Y_He / (const_mHe_mH * (1.0 - Y_He))

    def x_H(self, thermo):
        return min(max(thermo[0], 0.0), 1.0)

    def x_He(self, thermo):
        return min(max(thermo[1], 0.0), 2.0)

    def Tmat(self, thermo):
        return abs(thermo[2])

    def xe_rec(self, x_H, x_He, fHe):
        """Free electrons per hydrogen nucleus from recombination alone."""
        return x_H + fHe * x_He

    def xe_reio(self, a, xe_rec, fHe, z_re):
        """CAMB's reionization step, zero if ``z_re = 0``."""
        inv_a = 1.0 / a
        z = inv_a - 1.0
        # fp32 under FP32_MATH, unlike ``a``. The step matters only where the
        # opacity is too small for the rounding to show.
        y = pow_(1.0 + z, 1.5)
        y_re = pow_(1.0 + z_re, 1.5)
        delta_y = 0.75 * pow_(1.0 + z_re, 0.5)
        step = logistic((y_re - y) / delta_y)
        he3 = logistic((3.5 - z) / 0.5)
        on = min(max(z_re, 0.0), 1.0)
        return ((1.0 + fHe - xe_rec) * step + fHe * he3) * on

    def xe(self, xe_rec, xe_reio):
        return xe_rec + xe_reio

    def akthom(self, Y_He, omega_b):
        return AKTHOM_PER_OMEGAB * (1.0 - Y_He) * omega_b

    def opacity(self, xe, akthom, a):
        inv_a = 1.0 / a
        inv_a2 = inv_a * inv_a
        return max(xe * akthom * inv_a2, 1.0e-30)

    def cs2(self, Tmat, xe_rec, Y_He, dTm_dz, a):
        """``c_s^2 = k_B T_m / (mu m_H c^2) (1 - (1/3) d ln T_m / d ln a)``."""
        inv_a = 1.0 / a
        inv_mu = 1.0 + (1.0 / const_mHe_mH - 1.0) * Y_He + (1.0 - Y_He) * xe_rec
        dTm_dlna = -dTm_dz * inv_a
        return max(
            const_kB / (3.0 * const_mH * const_c**2) * inv_mu * (3.0 * Tmat - dTm_dlna),
            0.0,
        )

    # The rates

    def zred(self, a):
        inv_a = 1.0 / a
        return inv_a - 1.0

    def Hz(self, aH, a):
        """``H`` in 1/s, from the model's conformal Hubble rate in 1/Mpc."""
        inv_a = 1.0 / a
        return aH * inv_a * const_c_Mpc_s

    def Nnow(self, omega_b, Y_He):
        """The hydrogen number density today, in 1/m^3."""
        return DENS_FAC_PER_OMEGAB * omega_b * (1.0 - Y_He)

    def n_H(self, Nnow, a):
        inv_a = 1.0 / a
        return Nnow * inv_a**3

    def K_peebles(self, Hz, zred):
        return peebles_K(Hz, zred)

    def dTm_dz(self, Tmat, xe_rec, fHe, zred, Hz, T_cmb):
        return matter_temperature_rate(Tmat, xe_rec, fHe, zred, Hz, T_cmb)

    @derivative("tau")
    def conformal_time(self):
        return 1.0

    @derivative("thermo")
    def recfast(
        self,
        x_H,
        x_He,
        xe_rec,
        Tmat,
        zred,
        a,
        n_H,
        Hz,
        T_cmb,
        Nnow,
        fHe,
        K_peebles,
        dTm_dz,
        aH,
        output,
    ):
        inv_a = 1.0 / a
        dz_dtau = -aH * inv_a
        output[0] = (
            hydrogen_rate(x_H, xe_rec, Tmat, zred, n_H, Hz, T_cmb, Nnow, K_peebles)
            * dz_dtau
        )
        output[1] = (
            helium_rate(x_H, x_He, xe_rec, Tmat, zred, a, n_H, Hz, T_cmb, Nnow, fHe)
            * dz_dtau
        )
        output[2] = dTm_dz * dz_dtau

    @derivative("kappa")
    def optical_depth(self, opacity):
        return opacity


class ReionizationDepth(Component):
    """The optical depth of reionization as a state: ``kappa_reio``, whose value today is ``tau_reio``.

    A model adds it to read ``tau_reio`` off a solve or to differentiate in
    the reionization redshift, ``LCDM.add(ReionizationDepth, name=...)``.
    ``kappa_reio`` integrates the opacity of reionization's electrons alone,
    a background state beside :class:`Recombination`'s. A sensitivity
    launch weights it (:data:`REIO_ERROR_WEIGHT`) so that the steps resolve
    the reionization step's derivative in ``z_re``: without the state,
    ``dP/dz_re`` is 3 to 5 percent off, and :func:`~discoeb.solver.solve_eb`
    refuses to differentiate ``z_re``.
    """

    kappa_reio = State()

    @derivative("kappa_reio")
    def reionization_depth(self, xe_reio, akthom, a):
        """The opacity of reionization's electrons alone, whose integral is ``tau_reio``."""
        inv_a = 1.0 / a
        inv_a2 = inv_a * inv_a
        return xe_reio * akthom * inv_a2
