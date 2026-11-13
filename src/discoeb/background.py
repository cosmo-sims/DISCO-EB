"""The expansion history: density coefficients, the Friedmann sum and the neutrino background.

The background uses the same species as the perturbations.
:meth:`~discoeb.component.Model.evaluate` computes the kernel's Friedmann sum
``grho`` on JAX arrays at ``a``, which gives the expansion rate the
conformal-time quadrature and the thermal history integrate
(:func:`expansion_rate`, :func:`dtauda`). :func:`density_parameters` closes
the density budget at ``a = 1`` with the same sum.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np

from .component import Model, derive_parameters
from .cosmology import _as_cosmology, _ParamCosmology
from .gravity import (  # noqa: F401 (the density constants, re-exported)
    C2OK,
    GRHOG_PER_TCMB4,
    GRHOM_PER_H0SQ,
    GRHOR_PER_TCMB4,
)
from .models import W0WACDM_MNU, CosmologyModel
from .recombination import recombination_model
from .util import neutrino_momentum_bins

# The density coefficients are in CAMB's convention (see discoeb.gravity), and
# the species declare how each derives from the cosmology's keys.

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
    # An elementwise sum over the nodes, not jnp.sum over a trailing node
    # axis: vmapped over a batch's background grid, XLA emits that reduction
    # as one unrolled kernel per row block, which ptxas took half an hour to
    # assemble for 128 cosmologies.
    ratio = sum(
        float(wi) * jnp.sqrt(1.0 + (y / float(qi)) ** 2) for qi, wi in zip(q, w)
    )
    return jnp.where(amnu > 0.0, ratio, 1.0)


def background_model(model: CosmologyModel | None = None) -> Model:
    """The species of ``model`` whose ``grho`` is the Friedmann sum.

    Without a model it is :data:`~discoeb.models.W0WACDM_MNU`, whose
    components declare every built-in parameter. For a cosmology another
    built-in model accepts, its sum is that model's: at ``w = -1`` the
    dark-energy fluid is the cosmological constant, and a massive-neutrino
    density with ``n_mnu = 0`` vanishes. The hierarchy truncations don't
    affect the background.
    """

    return _background_model(W0WACDM_MNU if model is None else model)


@functools.lru_cache(maxsize=None)
def _background_model(model: CosmologyModel) -> Model:
    return model.build()


def density_parameters(param, xp=jnp, model: CosmologyModel | None = None) -> dict:
    """The parameters of ``model`` for a parameter dictionary, keyed as the components name them.

    The keys pass through, and every derived parameter comes from its
    declaration (:class:`~discoeb.component.Parameter`): ``grhom`` the
    critical density, ``grhog``, ``grhornomass``, ``grhoc``, ``grhob`` and
    ``grhok`` the photons, massless neutrinos, cold dark matter, baryons and
    curvature, ``grhor_nu``, ``n_mnu`` and ``amnu`` the massive neutrinos.
    ``grhov``, the dark energy's, closes the budget at ``a = 1`` with the
    model's own Friedmann sum: ``grhov = grhom - grho(a=1 | grhov=0)``.
    Without a model these are :data:`~discoeb.models.W0WACDM_MNU`'s. Pass
    ``xp=jnp`` for a traced cosmology and ``xp=np`` for one given as floats.
    """

    background = background_model(model)
    math_module = math if xp is np else jnp

    def close(name, values):
        amnu = values.get("amnu", 0.0)
        without = background.evaluate(
            "grho",
            math=math_module,
            a=1.0,
            rhonu=neutrino_density_ratio(1.0, amnu, xp),
            **{**values, name: 0.0},
        )
        return values["grhom"] - without

    parameters = (W0WACDM_MNU if model is None else model).parameters()
    return derive_parameters(parameters, param, math=math_module, close=close)


def expansion_rate(a, densities, model: CosmologyModel | None = None):
    """The conformal Hubble rate ``a'/a = a H`` at scalar ``a``, from the model's ``adotoa``."""

    rhonu = neutrino_density_ratio(a, densities.get("amnu", 0.0))
    return background_model(model).evaluate(
        "adotoa", math=jnp, a=a, rhonu=rhonu, **densities
    )


def dtauda(a, densities, model: CosmologyModel | None = None):
    """``d tau / d a = 1 / (a^2 H)`` at scalar ``a`` (``jax.vmap`` over an array)."""

    return 1.0 / (a * expansion_rate(a, densities, model))


N_BACKGROUND_GRID = 4096
TAU_START = 0.1


def _densities(cosmology) -> dict:
    """Return :func:`density_parameters` of a cosmology as floats, memoised."""

    return _densities_of(_as_cosmology(cosmology))


@functools.lru_cache(maxsize=4096)
def _densities_of(cosmology: _ParamCosmology) -> dict:
    d = density_parameters(cosmology.to_background_params(), xp=np)
    return {k: float(v) for k, v in d.items()}


def density_coefficients(cosmology) -> tuple[float, float, float, float, float]:
    """Return ``(grhog, grhornomass, grhoc, grhob, grhov)`` of :func:`density_parameters`."""

    d = _densities(cosmology)
    return (d["grhog"], d["grhornomass"], d["grhoc"], d["grhob"], d["grhov"])


def _background_tau(d, model: CosmologyModel | None = None):
    """Return ``(a_grid, tau_grid)`` of the conformal-time quadrature for densities ``d`` (JAX)."""

    a_bg = jnp.geomspace(1.0e-9, 1.0, N_BACKGROUND_GRID)
    dt = jax.vmap(lambda a: dtauda(a, d, model))(a_bg)
    steps = 0.5 * (dt[1:] + dt[:-1]) * jnp.diff(a_bg)
    # the radiation that sets tau at the grid's start
    radiation = (
        d["grhog"]
        + d["grhornomass"]
        + d.get("grhor_nu", 0.0)
        * d.get("n_mnu", 0.0)
        * neutrino_density_ratio(1.0e-9, d.get("amnu", 0.0))
    )
    taumin = 1.0e-9 / jnp.sqrt(radiation / 3.0)
    return a_bg, jnp.concatenate((jnp.zeros((1,)), jnp.cumsum(steps))) + taumin


def _reionization_redshift(a_bg, tau_bg, params, tau_reio):
    """Bisect for ``z_re`` of CAMB's step on the background grid.

    Returns 0 unless ``tau_reio > 0``. Neglects the residual ionization left by
    recombination (``xe_rec = 0``), which shifts ``tau_reio`` by 2e-4 of itself.
    """

    dtau = jnp.gradient(tau_bg)
    thermal = {name: params[name] for name in ("YHe", "Omegab", "H0")}

    def tau_of(z_re):
        xe_reio, akthom = recombination_model().evaluate(
            "xe_reio",
            "akthom",
            math=jnp,
            inv_a=1.0 / a_bg,
            xe_rec=0.0,
            z_re=z_re,
            **thermal,
        )
        return jnp.sum(xe_reio * akthom / a_bg**2 * dtau)

    def bisect(carry, _):
        lo, hi = carry
        mid = 0.5 * (lo + hi)
        too_much = tau_of(mid) > tau_reio
        return (jnp.where(too_much, lo, mid), jnp.where(too_much, mid, hi)), None

    (lo, hi), _ = jax.lax.scan(
        bisect, (jnp.asarray(1.0), jnp.asarray(40.0)), None, length=40
    )
    return jnp.where(tau_reio > 0.0, 0.5 * (lo + hi), 0.0)


@functools.partial(jax.jit, static_argnums=(1,))
def background_endpoints(stacked, model: CosmologyModel | None = None):
    """Return ``(tau0, lna_start, z_re)`` of stacked background parameters, one vmap lane each.

    ``model`` gives the species (:func:`background_model`).

    ``tau0`` is the conformal age, where each trajectory stops. ``lna_start`` is
    ``ln a`` at :data:`TAU_START`, where it starts. ``z_re`` is the reionization
    redshift that gives ``tau_reio`` (0 without reionization). The bisection for
    it runs on the background and neglects the residual ionization left by
    recombination, which shifts ``tau_reio`` by 2e-4 of itself.
    """

    def one(params):
        a_bg, tau_bg = _background_tau(density_parameters(params, model=model), model)
        lna_start = jnp.log(jnp.interp(TAU_START, tau_bg, a_bg))
        z_re = _reionization_redshift(a_bg, tau_bg, params, params["tau_reio"])
        return tau_bg[-1], lna_start, z_re

    return jax.vmap(one)(stacked)


def background_history(cosmology):
    """Return ``(a, tau)`` of one cosmology's background quadrature, host arrays ending at ``tau0``."""

    return _background_history(_as_cosmology(cosmology))


@functools.lru_cache(maxsize=64)
def _background_history(cosmology: _ParamCosmology):
    d = {k: jnp.asarray(v) for k, v in _densities(cosmology).items()}
    return tuple(np.asarray(x) for x in _background_tau(d))
