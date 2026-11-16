"""
Components are the building blocks that define cosmological models in DISCO-EB, including gravity terms, fluid species and interactions between species.

This module defines the public API for components.
"""

from . import gravity, interactions, recombination, species
from ._component import (
    Component,
    Derived,
    Equations,
    Gravity,
    Interaction,
    Parameter,
    Setting,
    Species,
    State,
    contributes,
)

__all__ = [
    "Component",
    "Derived",
    "Equations",
    "Gravity",
    "Interaction",
    "Parameter",
    "Setting",
    "Species",
    "State",
    "contributes",
    "gravity",
    "interactions",
    "recombination",
    "species",
]
