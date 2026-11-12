"""w0waCDM: LCDM with the cosmological constant replaced by a CPL dark-energy fluid."""

from ..species import CosmologicalConstant, DarkEnergy, MassiveNeutrinos
from .lcdm import LCDM

W0WACDM = LCDM.replace(CosmologicalConstant, DarkEnergy, name="w0waCDM")
"""LCDM with a dark-energy fluid of equation of state ``w = w_DE_0 + w_DE_a (1 -
a)`` and sound speed ``cs2_DE``, perturbed. Needs ``w != -1``."""

W0WACDM_MNU = W0WACDM.add(MassiveNeutrinos, name="w0waCDM+mnu")
"""w0waCDM with massive neutrinos."""
