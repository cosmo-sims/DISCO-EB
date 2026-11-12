"""Cosmological models built from the species, interactions and gravity components.

A model lists the components it evolves (:class:`CosmologyModel`), and every
solve names its model: ``solve_eb(cosmologies, k, tau, model=LCDM)``. Models
derive from each other, so a new one is usually a line::

    W0WACDM = LCDM.replace(CosmologicalConstant, DarkEnergy, name="w0waCDM")
    LCDM_MNU = LCDM.add(MassiveNeutrinos, name="LCDM+mnu")
    COARSE = LCDM.with_settings(LMAX_G=10, LMAX_NR=10)
"""

from .base import CosmologyModel
from .lcdm import LCDM, LCDM_MNU
from .w0wacdm import W0WACDM, W0WACDM_MNU

__all__ = ["LCDM", "LCDM_MNU", "W0WACDM", "W0WACDM_MNU", "CosmologyModel"]
