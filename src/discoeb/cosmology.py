"""Cosmologies as the solver reads them: parameter dictionaries and their hashable form.

A cosmology is an instance of a model (:mod:`discoeb.models`), ``LCDM(H0=...,
...)``. Inside, the solver reads its parameters as a dictionary (``Omegam``,
``Omegab``, ``H0``, ``Tcmb``, ``YHe``, ``Neff``, ``A_s``, ``n_s``, ...), and
:func:`as_cosmology` gives a hashable form for caching the background.
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
    """Return a cosmology or a parameter dictionary as a ``_ParamCosmology``, anything else as is.

    Keys a component declares besides the fields go into ``extra``. A key no
    component declares is dropped, and a field's key left out takes its
    declared default.
    """

    if _is_model_instance(cosmology):
        cosmology = cosmology.to_dict()
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


def _is_model_instance(cosmology) -> bool:
    """Whether ``cosmology`` is an instance of a :class:`~discoeb.models.Model`."""

    from .models import Model

    return isinstance(cosmology, Model)


def _as_batch(cosmologies) -> tuple:
    """Return a cosmology or a sequence of them as a tuple (dictionaries unchanged)."""

    if isinstance(cosmologies, dict) or hasattr(cosmologies, "Omegam"):
        return (cosmologies,)
    return tuple(cosmologies)


DIFFERENTIABLE_PARAMETERS = (
    "Omegam",
    "Omegab",
    "H0",
    "Tcmb",
    "Neff",
    "Omegak",
    "mnu",
    "w_DE_0",
    "w_DE_a",
)
"""The cosmology-dictionary keys :func:`solve_eb` differentiates with respect to.

They reach the solve through the directed inputs (:data:`DIRECTED_NAMES`):
the background densities, the recombination block's thermal inputs, the
curvature density, the massive neutrinos' density and mass and the
dark-energy equation of state. The forward sensitivities carry them through
both the hierarchy and the recombination history. A model differentiates
those its components declare (:func:`~discoeb.solver.differentiable_parameters`):
``mnu`` needs :class:`~discoeb.species.MassiveNeutrinos` and ``w_DE_0`` and
``w_DE_a`` :class:`~discoeb.species.DarkEnergy`. The derivative holds fixed
the conformal age ``tau0`` at which a trajectory stops
(:func:`matter_power_spectrum` adds its term), the reionization redshift
found for ``tau_reio``, ``ln a`` at the start and the initial state. Every
key may be traced under ``jit``. These raise if differentiated: ``Nmnu``, a
count, and ``YHe``, ``cs2_DE`` and ``tau_reio``, whose derivatives of
``P(k)`` miss central differences by 0.8, 1 and 1.8 percent. The solve never
reads ``A_s``, ``n_s`` and ``k_p``."""


def _cosmology_dictionary(cosmology) -> dict:
    """Return a cosmology's parameter dictionary, given a cosmology, a dictionary or a ``_ParamCosmology``."""

    if isinstance(cosmology, dict):
        return dict(cosmology)
    if _is_model_instance(cosmology):
        return cosmology.to_dict()
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
        f"{name!r} cannot be differentiated through solve_eb of this model: the "
        f"differentiable parameters are those of {DIFFERENTIABLE_PARAMETERS} "
        "the model's components declare (differentiable_parameters)"
    )


def _columns(
    cosmologies, held: bool = True, keys=None, differentiable=DIFFERENTIABLE_PARAMETERS
) -> dict:
    """Return a batch's parameters as JAX columns ``(n_cosmologies,)`` by key.

    ``keys`` maps the keys a model declares to their defaults
    (:meth:`ModelSpec.parameter_keys
    <discoeb.models.ModelSpec.parameter_keys>`), by default those of
    every built-in species. A cosmology may leave out a key with a default.
    Keys other components declare are kept when a cosmology gives them, so
    that differentiating one raises, and keys no component declares are
    dropped. The cosmologies are dictionaries or cosmology objects, their
    values numbers or JAX values, traced or not. With ``held``, a key the
    solve reads but does not differentiate (outside ``differentiable``, by
    default :data:`DIFFERENTIABLE_PARAMETERS`, and :data:`UNUSED_BY_THE_SOLVE`) raises
    if differentiated: its derivative would otherwise come back as a silent
    zero.
    """

    if keys is None:
        from .models import DEFAULT_SPEC

        keys = DEFAULT_SPEC.parameter_keys()
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


as_cosmology = _as_cosmology
"""Return a cosmology, or a parameter dictionary (``Omegam``, ``Omegab``, ``H0``,
``Tcmb``, ``YHe``, ``Neff``, ``A_s``, ``n_s``, ...), in the hashable form the
background caches on. Returns one already in that form as it is. The form
keeps the parameters, not the model."""
