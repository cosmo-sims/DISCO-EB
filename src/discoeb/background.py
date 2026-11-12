"""The expansion history: density coefficients, the Friedmann sum and the neutrino background.

The background uses the same species as the perturbations.
:meth:`~discoeb.component.Model.evaluate` computes the kernel's Friedmann sum
``grho`` on JAX arrays at ``a``, which gives the expansion rate the
conformal-time quadrature and the thermal history integrate
(:func:`expansion_rate`, :func:`dtauda`). :func:`density_parameters` closes
the density budget at ``a = 1`` with the same sum.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from .component import Model
from .gravity import SUMS, Background, Curvature
from .species import (
    Baryons,
    ColdDarkMatter,
    DarkEnergy,
    MassiveNeutrinos,
    MasslessNeutrinos,
    Photons,
)
from .util import neutrino_momentum_bins

# Density coefficients in CAMB's convention, ``8 pi G rho / c^2`` in 1/Mpc^2:
# ``H^2 = sum_i grho_i / 3`` gives the Hubble rate in 1/Mpc and ``a'/a = a H``
# the conformal one. The perturbation kernel's ``grho_i_t = 8 pi G rho_i a^2``
# are these times ``a^2``.

GRHOM_PER_H0SQ = 3.33795017e-11  # 8 pi G rho_crit / c^2 per (H0 in km/s/Mpc)^2
GRHOG_PER_TCMB4 = 1.49594245e-13  # photons, per K^4
GRHOR_PER_TCMB4 = 3.39739477e-14  # one massless neutrino flavour, per K^4
C2OK = 1.62581581e4  # K / eV: amnu = m_nu c^2 / (k_B T_nu0) = mnu * C2OK / Tcmb

NQ_NEUTRINO_TABLE = 100
"""Momentum nodes for the massive-neutrino background density. ``rho_nu(a)``
and the budget closing both integrate the Fermi-Dirac distribution on
these."""


def neutrino_density_ratio(a, amnu, xp=jnp):
    """``rho_nu(a) / rho_nu,massless`` of one flavour of mass ``amnu`` (1 when massless).

    Sums the Fermi-Dirac quadrature on ``NQ_NEUTRINO_TABLE`` momentum nodes,
    which depends only on ``y = amnu * a``. With ``xp=jnp`` (the default) it
    traces under ``jit``; with ``xp=np`` it runs on the host, for closing the
    budget of a cosmology given as floats.
    """

    q, w, _ = neutrino_momentum_bins(NQ_NEUTRINO_TABLE)
    if xp is np:
        y = np.asarray(a, dtype=np.float64) * float(amnu)
        return np.sum(w * np.sqrt(1.0 + (y[..., None] / q) ** 2), axis=-1)
    y = jnp.asarray(a) * amnu
    q, w = jnp.asarray(q, dtype=y.dtype), jnp.asarray(w, dtype=y.dtype)
    ratio = jnp.sum(w * jnp.sqrt(1.0 + (y[..., None] / q) ** 2), axis=-1)
    return jnp.where(amnu > 0.0, ratio, 1.0)


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
