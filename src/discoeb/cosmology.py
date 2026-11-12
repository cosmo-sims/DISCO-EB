"""Cosmologies: the parameter dictionary and its hashable form.

A cosmology is a parameter dictionary (``Omegam``, ``Omegab``, ``H0``,
``Tcmb``, ``YHe``, ``Neff``, ``A_s``, ``n_s``, ...) or the hashable object
:func:`as_cosmology` returns. It is a point in a model's parameter space; the
model (:mod:`discoeb.models`) says which species it evolves.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


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
        }


def _as_cosmology(cosmology):
    """Return a parameter dictionary as a ``_ParamCosmology``, anything else as is."""

    if not isinstance(cosmology, dict):
        return cosmology
    return _ParamCosmology(
        Omegam=float(cosmology["Omegam"]),
        Omegab=float(cosmology["Omegab"]),
        w_DE_0=float(cosmology.get("w_DE_0", -1.0)),
        w_DE_a=float(cosmology.get("w_DE_a", 0.0)),
        cs2_DE=float(cosmology.get("cs2_DE", 1.0)),
        Omegak=float(cosmology.get("Omegak", 0.0)),
        A_s=float(cosmology["A_s"]),
        n_s=float(cosmology["n_s"]),
        H0=float(cosmology["H0"]),
        T_cmb=float(cosmology["Tcmb"]),
        Y_He=float(cosmology["YHe"]),
        Neff_massless=float(cosmology["Neff"]),
        num_massive_neutrinos=float(cosmology.get("Nmnu", 0.0)),
        mnu=float(cosmology.get("mnu", 0.0)),
        k_pivot=float(cosmology.get("k_p", 0.05)),
        tau_reio=float(cosmology.get("tau_reio", 0.0)),
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
initial state. The other keys must be plain numbers. Some enter as parameter
columns the sensitivities do not carry (``Omegak``, ``Nmnu``, ``mnu``,
``w_DE_0``, ``w_DE_a``, ``cs2_DE``, ``YHe``, ``tau_reio``), the rest not at
all (``A_s``, ``n_s``, ``k_p``)."""


def _cosmology_dictionary(cosmology) -> dict:
    """Return a cosmology's parameter dictionary, given a dictionary or a cosmology object."""

    if isinstance(cosmology, dict):
        return dict(cosmology)
    d = cosmology.to_background_params()
    d["k_p"] = cosmology.k_pivot
    return d


def _split_cosmologies(cosmologies):
    """Return ``(fixed, theta)``: each cosmology's plain and differentiable parameters.

    ``fixed`` holds one dictionary per cosmology of the keys outside
    :data:`DIFFERENTIABLE_PARAMETERS`, as floats. The float conversion raises
    ``ValueError`` for a JAX tracer given for one of those keys. ``theta`` is
    the ``(n_cosmologies, 5)`` array of the differentiable keys, traced or not.
    """

    fixed, rows = [], []
    for cosmology in cosmologies:
        d = _cosmology_dictionary(cosmology)
        plain = {}
        for key, value in d.items():
            if key in DIFFERENTIABLE_PARAMETERS:
                continue
            try:
                plain[key] = float(value)
            except TypeError as error:
                raise ValueError(
                    f"{key!r} cannot be differentiated through solve_eb: the "
                    f"differentiable parameters are {DIFFERENTIABLE_PARAMETERS}"
                ) from error
        fixed.append(plain)
        values = [d[k] for k in DIFFERENTIABLE_PARAMETERS]
        if any(isinstance(v, jax.Array) for v in values):
            rows.append(jnp.stack([jnp.asarray(v, dtype=jnp.float64) for v in values]))
        else:
            rows.append(np.asarray(values, dtype=np.float64))
    if all(isinstance(r, np.ndarray) for r in rows):
        # plain numbers: one host array, no per-cosmology device work
        return tuple(fixed), np.stack(rows)
    return tuple(fixed), jnp.stack(rows)


as_cosmology = _as_cosmology
"""Normalise a cosmology given as a parameter dictionary (``Omegam``, ``Omegab``,
``H0``, ``Tcmb``, ``YHe``, ``Neff``, ``A_s``, ``n_s``, ... ) to the solver's
cosmology object. Returns one already normalised as it is."""
