"""Components: the framework, and the species, gravity and interactions built on it.

:mod:`~discoeb.components.component` is the framework that turns components
into right-hand sides. :mod:`~discoeb.components.species`,
:mod:`~discoeb.components.gravity`, :mod:`~discoeb.components.interactions`
and :mod:`~discoeb.components.recombination` hold the components a model
lists as its bases.
"""

from . import gravity, interactions, recombination, species
from .component import (
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
