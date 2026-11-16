"""The expansion history: density coefficients, the Friedmann sum and the conformal time.

The background uses the same species as the perturbations.
:meth:`~discoeb.components._component.Equations.evaluate` computes the kernel's Friedmann sum
``grho`` on JAX arrays at ``a``, which gives the expansion rate the
conformal-time quadrature and the thermal history integrate
(:func:`expansion_rate`, :func:`dtauda`). :func:`density_parameters` closes
the density budget by solving for the parameter that makes the same
expansion rate ``H0`` today.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np

from ..components._component import Equations, derive_parameters
from ..components.gravity import (  # noqa: F401 (the density constants, re-exported)
    C2OK,
    GRHOG_PER_TCMB4,
    GRHOM_PER_H0SQ,
    GRHOR_PER_TCMB4,
)
from ..models._model import Model, ModelSpec


def background_model(model: ModelSpec) -> Equations:
    """The species of ``model`` whose ``grho`` is the Friedmann sum.

    The hierarchy truncations don't affect the background.
    """

    return _background_model(model)


@functools.lru_cache(maxsize=None)
def _background_model(model: ModelSpec) -> Equations:
    return model.build()


def density_parameters(param, model: ModelSpec, xp=jnp) -> dict:
    """The parameters of ``model`` for a parameter dictionary, keyed as the components name them.

    The keys pass through, and every derived parameter comes from its
    declaration (:class:`~discoeb.components._component.Parameter`): ``grhog``,
    ``grhornomass``, ``grhoc``, ``grhob``, ``grhov`` and ``grhok`` the
    photons, massless neutrinos, cold dark matter, baryons, dark energy and
    curvature, ``grhor_nu``, ``n_mnu`` and ``amnu`` the massive neutrinos.

    Pass ``xp=jnp`` for a traced cosmology and ``xp=np`` for one given as floats.
    """

    parameters = model.parameters()
    return derive_parameters(parameters, param, math=math if xp is np else jnp)


def expansion_rate(a, densities, model: ModelSpec):
    """The conformal Hubble rate ``a'/a = a H`` at scalar ``a``, from the model's ``adotoa``."""

    return background_model(model).evaluate("adotoa", math=jnp, a=a, **densities)


def dtauda(a, densities, model: ModelSpec):
    """``d tau / d a = 1 / (a^2 H)`` at scalar ``a`` (``jax.vmap`` over an array)."""

    return 1.0 / (a * expansion_rate(a, densities, model))


N_BACKGROUND_GRID = 4096
TAU_START = 0.1


def _hashable(cosmology) -> tuple[ModelSpec, tuple]:
    """Return a cosmology's model and its parameters as floats, a cache key."""

    if not isinstance(cosmology, Model):
        raise TypeError(
            "expected a cosmology of a model, such as LCDM(H0=..., ...), not "
            f"{type(cosmology).__name__}"
        )
    values = tuple(sorted((k, float(v)) for k, v in cosmology.to_dict().items()))
    return type(cosmology).spec, values


def _densities(cosmology) -> dict:
    """Return :func:`density_parameters` of a cosmology as floats, memoised."""

    return _densities_of(*_hashable(cosmology))


@functools.lru_cache(maxsize=4096)
def _densities_of(model: ModelSpec, values: tuple) -> dict:
    d = density_parameters(dict(values), model, xp=np)
    return {k: float(v) for k, v in d.items()}


def density_coefficients(cosmology) -> tuple[float, float, float, float, float]:
    """Return ``(grhog, grhornomass, grhoc, grhob, grhov)`` of :func:`density_parameters`.

    ``cosmology`` is a cosmology of a model, ``LCDM(H0=..., ...)``.
    """

    d = _densities(cosmology)
    return (d["grhog"], d["grhornomass"], d["grhoc"], d["grhob"], d["grhov"])


A_START = 1.0e-9
"""The scale factor the background's conformal time is integrated from."""


def _background_solution(d, model: ModelSpec):
    """Return ``(a_grid, tau_grid, states)`` of the background for the parameters ``d`` (JAX).

    The grid is geometric in ``a`` from :data:`A_START` to 1, where the
    time starts at ``1 / (a'/a)``, its value under radiation. If the
    model's Friedmann rate is a function of ``a`` (every built-in model's),
    the time is the quadrature of ``d tau / d a`` and ``states`` is empty.

    A model may also evolve background states of its own: a scalar field,
    a density that decays (:meth:`Equations.stages
    <discoeb.components._component.Equations.stages>` finds them). Their rates in ``ln a``,
    ``y' / (a'/a)``, are then integrated along the grid with the time by
    fixed Runge-Kutta steps of 0.005 in ``ln a``, from the states' initial
    conditions at :data:`A_START`. ``states`` holds each one on the grid.
    That resolves anything that varies on a Hubble time and not a field
    oscillating faster.
    """

    background = background_model(model)
    names = background.stages()[0]
    a_bg = jnp.geomspace(A_START, 1.0, N_BACKGROUND_GRID)
    if not names:
        dt = jax.vmap(lambda a: dtauda(a, d, model))(a_bg)
        steps = 0.5 * (dt[1:] + dt[:-1]) * jnp.diff(a_bg)
        tau_start = 1.0 / expansion_rate(A_START, d, model)
        tau = jnp.concatenate((jnp.zeros((1,)), jnp.cumsum(steps))) + tau_start
        return a_bg, tau, {}
    vectors = [n for n in names if background.states[n].shape]
    if vectors:
        raise NotImplementedError(
            f"the background states {vectors} are vectors; the background "
            "integrates scalar states"
        )
    rate_names = tuple(f"d_{n}" for n in names)

    def rates(lna, y):
        tau, values = y
        adotoa, *moving = background.evaluate(
            "adotoa",
            *rate_names,
            math=jnp,
            a=jnp.exp(lna),
            tau=tau,
            **dict(zip(names, values)),
            **d,
        )
        return 1.0 / adotoa, tuple(rate / adotoa for rate in moving)

    def step(y, span):
        lna, h = span

        def moved(by, slope):
            return jax.tree.map(lambda v, s: v + by * s, y, slope)

        k1 = rates(lna, y)
        k2 = rates(lna + 0.5 * h, moved(0.5 * h, k1))
        k3 = rates(lna + 0.5 * h, moved(0.5 * h, k2))
        k4 = rates(lna + h, moved(h, k3))
        new = jax.tree.map(
            lambda v, a, b, c, e: v + h / 6.0 * (a + 2.0 * b + 2.0 * c + e),
            y,
            k1,
            k2,
            k3,
            k4,
        )
        return new, new

    lna = jnp.log(a_bg)
    initial = background.evaluate(
        *names, "adotoa", math=jnp, a=A_START, lna_start=lna[0], **d
    )
    y0 = (
        1.0 / initial[-1],
        tuple(jnp.asarray(v, dtype=jnp.float64) for v in initial[:-1]),
    )
    _, (tau, values) = jax.lax.scan(step, y0, (lna[:-1], jnp.diff(lna)))
    tau = jnp.concatenate((y0[0][None], tau))
    states = {
        name: jnp.concatenate((first[None], rest))
        for name, first, rest in zip(names, y0[1], values)
    }
    return a_bg, tau, states


@functools.partial(jax.jit, static_argnums=(1,))
def background_endpoints(stacked, model: ModelSpec):
    """Return ``(tau0, lna_start, start)`` of stacked background parameters, one vmap lane each.

    ``model`` gives the species (:func:`background_model`).

    ``tau0`` is the conformal age, where each trajectory stops. ``lna_start`` is
    ``ln a`` at :data:`TAU_START`, where it starts. ``start``
    holds the model's own background states at :data:`TAU_START` by name
    (:func:`_background_solution`), empty for the built-in models.
    """

    def one(params):
        a_bg, tau_bg, states = _background_solution(
            density_parameters(params, model=model), model
        )
        lna_start = jnp.log(jnp.interp(TAU_START, tau_bg, a_bg))
        start = {n: jnp.interp(TAU_START, tau_bg, v) for n, v in states.items()}
        return tau_bg[-1], lna_start, start

    return jax.vmap(one)(stacked)


def background_history(cosmology):
    """Return ``(a, tau)`` of one cosmology's background quadrature, host arrays ending at ``tau0``.

    ``cosmology`` is a cosmology of a model, ``LCDM(H0=..., ...)``.
    """

    return _background_history(*_hashable(cosmology))


@functools.lru_cache(maxsize=64)
def _background_history(model: ModelSpec, values: tuple):
    d = {k: jnp.asarray(v) for k, v in _densities_of(model, values).items()}
    return tuple(np.asarray(x) for x in _background_solution(d, model)[:2])
