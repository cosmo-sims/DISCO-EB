"""Components: the framework, and the species, gravity and interactions built on it.

The framework that turns components into right-hand sides is private
(``_component``); its public names are exported here. :mod:`~discoeb.components.species`,
:mod:`~discoeb.components.gravity`, :mod:`~discoeb.components.interactions`
and :mod:`~discoeb.components.recombination` hold the components a model
lists as its bases.
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
