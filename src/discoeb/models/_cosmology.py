"""Cosmologies as the solver reads them: parameter dictionaries.

A cosmology is an instance of a model (:mod:`discoeb.models`), ``LCDM(omega_b=...,
...)``. Inside, the solver reads its parameters as a dictionary (``omega_b``,
``omega_c``, ``omega_de``, ``Tcmb``, ``YHe``, ``Neff``, ``A_s``, ``n_s``, ...).
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from ..components._component import REQUIRED, declarations_of


def _is_parameter(key) -> bool:
    """Whether a component declares ``key`` as a cosmology-dictionary key."""

    return any(p.is_key for _, p in declarations_of(key))


def _default(key):
    """A key's default, as the components declaring it give it (:data:`REQUIRED` if none)."""

    for _, parameter in declarations_of(key):
        if parameter.is_key:
            return parameter.default
    return REQUIRED


def _is_model_instance(cosmology) -> bool:
    """Whether ``cosmology`` is an instance of a :class:`~discoeb.models.Model`."""

    from ._model import Model

    return isinstance(cosmology, Model)


def _as_batch(cosmologies) -> tuple:
    """Return a cosmology or a sequence of them as a tuple (dictionaries unchanged)."""

    if isinstance(cosmologies, dict) or _is_model_instance(cosmologies):
        return (cosmologies,)
    return tuple(cosmologies)


DIFFERENTIABLE_PARAMETERS = (
    "omega_b",
    "omega_c",
    "omega_de",
    "Tcmb",
    "Neff",
    "omega_k",
    "mnu",
    "w_DE_0",
    "w_DE_a",
    "z_re",
)
"""The cosmology-dictionary keys :func:`solve_eb` differentiates with respect to.

They reach the solve through the directed inputs
(:data:`~discoeb.solver.DIRECTED_NAMES` and those of the model's components):
the background densities, the recombination block's thermal inputs, the
curvature density, the massive neutrinos' density and mass, the
dark-energy equation of state and the reionization redshift. The forward sensitivities carry them through
both the hierarchy and the recombination history. A model differentiates
those its components declare (:func:`~discoeb.solver.differentiable_parameters`):
``mnu`` needs :class:`~discoeb.components.species.MassiveNeutrinos` and ``w_DE_0`` and
``w_DE_a`` :class:`~discoeb.components.species.DarkEnergy`. The derivative holds fixed
the conformal age ``tau0`` at which a trajectory stops
(:func:`matter_power_spectrum` adds its term), ``ln a`` at the start and
the initial state. Every key may be traced under ``jit``. These raise if
differentiated: ``Nmnu``, a count, and ``YHe`` and ``cs2_DE``, whose
derivatives of ``P(k)`` miss central differences by 0.8 and 1 percent. The solve never
reads ``A_s``, ``n_s`` and ``k_p``."""


def _cosmology_dictionary(cosmology) -> dict:
    """Return a cosmology's parameter dictionary, given a cosmology, a dictionary or a ``_ParamCosmology``."""

    if isinstance(cosmology, dict):
        return dict(cosmology)
    return cosmology.to_dict()


UNUSED_BY_THE_SOLVE = ("A_s", "n_s", "k_p")
"""Keys the Einstein-Boltzmann solve never reads: the primordial spectrum's."""


@functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
def _not_differentiable(value, name):
    """Return ``value``, and raise if anything differentiates with respect to it."""

    return value


@_not_differentiable.defjvp
def _not_differentiable_jvp(name, primals, tangents):
    raise ValueError(
        f"{name!r} cannot be differentiated through solve_eb of this model: the "
        f"differentiable parameters are those of {DIFFERENTIABLE_PARAMETERS} "
        "the model's components declare (differentiable_parameters)"
    )


def _columns(
    cosmologies, keys, held: bool = True, differentiable=DIFFERENTIABLE_PARAMETERS
) -> dict:
    """Return a batch's parameters as JAX columns ``(n_cosmologies,)`` by key.

    ``keys`` maps the keys a model declares to their defaults
    (:meth:`ModelSpec.parameter_keys
    <discoeb.models.ModelSpec.parameter_keys>`). A cosmology may leave out a key with a default.
    Keys other components declare are kept when a cosmology gives them, so
    that differentiating one raises, and keys no component declares are
    dropped. The cosmologies are dictionaries or cosmology objects, their
    values numbers or JAX values, traced or not. With ``held``, a key the
    solve reads but does not differentiate (outside ``differentiable``, by
    default :data:`DIFFERENTIABLE_PARAMETERS`, and :data:`UNUSED_BY_THE_SOLVE`) raises
    if differentiated: its derivative would otherwise come back as a silent
    zero.
    """

    rows = [_cosmology_dictionary(c) for c in cosmologies]
    keys = dict(keys)
    for row in rows:
        for key in row:
            if key not in keys and _is_parameter(key):
                keys[key] = _default(key)
    columns = {}
    for key, default in keys.items():
        values = []
        for row in rows:
            if key in row:
                values.append(row[key])
            elif default is REQUIRED:
                raise KeyError(key)
            else:
                values.append(default)
        if any(isinstance(v, jax.Array) for v in values):
            column = jnp.stack([jnp.asarray(v, dtype=jnp.float64) for v in values])
        else:
            column = jnp.asarray(np.asarray(values, dtype=np.float64))
        if held and key not in (*differentiable, *UNUSED_BY_THE_SOLVE):
            column = _not_differentiable(column, key)
        columns[key] = column
    return columns


def _is_concrete(value) -> bool:
    """Whether ``value`` is a number now: not a JAX tracer of ``jit`` or a derivative."""

    return not isinstance(value, jax.core.Tracer)
