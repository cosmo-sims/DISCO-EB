"""Cosmologies: the parameter dictionary and its hashable form.

A cosmology is a parameter dictionary (``Omegam``, ``Omegab``, ``H0``,
``Tcmb``, ``YHe``, ``Neff``, ``A_s``, ``n_s``, ...) or the hashable object
:func:`as_cosmology` returns. It is a point in a model's parameter space; the
model (:mod:`discoeb.models`) says which species it evolves.
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .component import REQUIRED, declarations_of


class _ParamCosmology(NamedTuple):
    """Hashable form of the DISCO-EB parameter dictionary."""

    Omegam: float
    Omegab: float
    w_DE_0: float
    w_DE_a: float
    cs2_DE: float
    Omegak: float
    A_s: float
    n_s: float
    H0: float
    T_cmb: float
    Y_He: float
    Neff_massless: float
    num_massive_neutrinos: float
    mnu: float
    k_pivot: float
    tau_reio: float = 0.0  # reionization optical depth (0: no reionization)
    extra: tuple = ()  # (key, value) of the parameters other components declare

    def to_background_params(self):
        return {
            "Omegam": self.Omegam,
            "Omegab": self.Omegab,
            "w_DE_0": self.w_DE_0,
            "w_DE_a": self.w_DE_a,
            "cs2_DE": self.cs2_DE,
            "Omegak": self.Omegak,
            "A_s": self.A_s,
            "n_s": self.n_s,
            "H0": self.H0,
            "Tcmb": self.T_cmb,
            "YHe": self.Y_He,
            "Neff": self.Neff_massless,
            "Nmnu": self.num_massive_neutrinos,
            "mnu": self.mnu,
            "tau_reio": self.tau_reio,
            **dict(self.extra),
        }


_FIELD_KEYS = {
    "Omegam": "Omegam",
    "Omegab": "Omegab",
    "w_DE_0": "w_DE_0",
    "w_DE_a": "w_DE_a",
    "cs2_DE": "cs2_DE",
    "Omegak": "Omegak",
    "A_s": "A_s",
    "n_s": "n_s",
    "H0": "H0",
    "T_cmb": "Tcmb",
    "Y_He": "YHe",
    "Neff_massless": "Neff",
    "num_massive_neutrinos": "Nmnu",
    "mnu": "mnu",
    "k_pivot": "k_p",
    "tau_reio": "tau_reio",
}
"""The cosmology object's fields and the dictionary keys they hold."""


def _is_parameter(key) -> bool:
    """Whether a component declares ``key`` as a cosmology-dictionary key."""

    return any(p.is_key for _, p in declarations_of(key))


def _default(key):
    """A key's default, as the components declaring it give it (:data:`REQUIRED` if none)."""

    for _, parameter in declarations_of(key):
        if parameter.is_key:
            return parameter.default
    return REQUIRED


def _as_cosmology(cosmology):
    """Return a parameter dictionary as a ``_ParamCosmology``, anything else as is.

    Keys a component declares besides the fields go into ``extra``. A key no
    component declares is dropped, and a field's key left out takes its
    declared default.
    """

    if not isinstance(cosmology, dict):
        return cosmology
    c = dict(cosmology)
    for key in _FIELD_KEYS.values():
        if key not in c and _default(key) is not REQUIRED:
            c[key] = _default(key)
    fields = set(_FIELD_KEYS.values())
    extra = tuple(
        sorted(
            (k, float(v))
            for k, v in cosmology.items()
            if k not in fields and _is_parameter(k)
        )
    )
    return _ParamCosmology(
        Omegam=float(c["Omegam"]),
        Omegab=float(c["Omegab"]),
        w_DE_0=float(c["w_DE_0"]),
        w_DE_a=float(c["w_DE_a"]),
        cs2_DE=float(c["cs2_DE"]),
        Omegak=float(c["Omegak"]),
        A_s=float(c["A_s"]),
        n_s=float(c["n_s"]),
        H0=float(c["H0"]),
        T_cmb=float(c["Tcmb"]),
        Y_He=float(c["YHe"]),
        Neff_massless=float(c["Neff"]),
        num_massive_neutrinos=float(c["Nmnu"]),
        mnu=float(c["mnu"]),
        k_pivot=float(c["k_p"]),
        tau_reio=float(c["tau_reio"]),
        extra=extra,
    )


def _as_batch(cosmologies) -> tuple:
    """Return a cosmology or a sequence of them as a tuple (dictionaries unchanged)."""

    if isinstance(cosmologies, dict) or hasattr(cosmologies, "Omegam"):
        return (cosmologies,)
    return tuple(cosmologies)


DIFFERENTIABLE_PARAMETERS = ("Omegam", "Omegab", "H0", "Tcmb", "Neff")
"""The cosmology-dictionary keys :func:`solve_eb` differentiates with respect to.

They reach the solve through the five background densities and the
recombination block's thermal inputs (:data:`DIRECTED_NAMES`). The forward
sensitivities carry them through both the hierarchy and the recombination
history. The derivative holds fixed the conformal age ``tau0`` at which a
trajectory stops, the reionization redshift found for ``tau_reio``, and the
initial state. Every key may be traced under ``jit``. Those the solve reads
as parameter columns the sensitivities do not carry (``Omegak``, ``Nmnu``,
``mnu``, ``w_DE_0``, ``w_DE_a``, ``cs2_DE``, ``YHe``, ``tau_reio``) raise if
differentiated. The solve never reads ``A_s``, ``n_s`` and ``k_p``."""


def _cosmology_dictionary(cosmology) -> dict:
    """Return a cosmology's parameter dictionary, given a dictionary or a cosmology object."""

    if isinstance(cosmology, dict):
        return dict(cosmology)
    d = cosmology.to_background_params()
    d["k_p"] = cosmology.k_pivot
    return d


UNUSED_BY_THE_SOLVE = ("A_s", "n_s", "k_p")
"""Keys the Einstein-Boltzmann solve never reads: the primordial spectrum's."""


@functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
def _not_differentiable(value, name):
    """Return ``value``, and raise if anything differentiates with respect to it."""

    return value


@_not_differentiable.defjvp
def _not_differentiable_jvp(name, primals, tangents):
    raise ValueError(
        f"{name!r} cannot be differentiated through solve_eb: the differentiable "
        f"parameters are {DIFFERENTIABLE_PARAMETERS}"
    )


def _columns(cosmologies, held: bool = True, keys=None) -> dict:
    """Return a batch's parameters as JAX columns ``(n_cosmologies,)`` by key.

    ``keys`` maps the keys a model declares to their defaults
    (:meth:`CosmologyModel.parameter_keys
    <discoeb.models.CosmologyModel.parameter_keys>`), by default those of
    every built-in species. A cosmology may leave out a key with a default.
    Keys other components declare are kept when a cosmology gives them, so
    that differentiating one raises, and keys no component declares are
    dropped. The cosmologies are dictionaries or cosmology objects, their
    values numbers or JAX values, traced or not. With ``held``, a key the
    solve reads but does not differentiate (outside
    :data:`DIFFERENTIABLE_PARAMETERS` and :data:`UNUSED_BY_THE_SOLVE`) raises
    if differentiated: its derivative would otherwise come back as a silent
    zero.
    """

    if keys is None:
        from .models import W0WACDM_MNU

        keys = W0WACDM_MNU.parameter_keys()
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
        if held and key not in DIFFERENTIABLE_PARAMETERS + UNUSED_BY_THE_SOLVE:
            column = _not_differentiable(column, key)
        columns[key] = column
    return columns


def _is_concrete(value) -> bool:
    """Whether ``value`` is a number now: not a JAX tracer of ``jit`` or a derivative."""

    return not isinstance(value, jax.core.Tracer)


as_cosmology = _as_cosmology
"""Normalise a cosmology given as a parameter dictionary (``Omegam``, ``Omegab``,
``H0``, ``Tcmb``, ``YHe``, ``Neff``, ``A_s``, ``n_s``, ... ) to the solver's
cosmology object. Returns one already normalised as it is."""
