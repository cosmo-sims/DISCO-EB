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
    Sum,
    contributes,
    derivative,
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
    "Sum",
    "contributes",
    "derivative",
    "gravity",
    "interactions",
    "recombination",
    "species",
]
