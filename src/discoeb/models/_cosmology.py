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
        "differentiable parameters are those its components declare "
        "differentiable (differentiable_parameters(model))"
    )


def _columns(cosmologies, keys, held: bool = True, differentiable=()) -> dict:
    """Return a batch's parameters as JAX columns ``(n_cosmologies,)`` by key.

    ``keys`` maps the keys a model declares to their defaults
    (:meth:`ModelSpec.parameter_keys
    <discoeb.models.ModelSpec.parameter_keys>`). A cosmology may leave out a key with a default.
    Keys other components declare are kept when a cosmology gives them, so
    that differentiating one raises, and keys no component declares are
    dropped. The cosmologies are parameter dictionaries
    (:func:`~discoeb.models._model._batch_of` gives a batch's), their values
    numbers or JAX values, traced or not. With ``held``, a key the
    solve reads but does not differentiate (outside ``differentiable``,
    :meth:`ModelSpec.differentiable_parameters
    <discoeb.models.ModelSpec.differentiable_parameters>`, and
    :data:`UNUSED_BY_THE_SOLVE`) raises
    if differentiated: its derivative would otherwise come back as a silent
    zero.
    """

    rows = [dict(c) for c in cosmologies]
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
