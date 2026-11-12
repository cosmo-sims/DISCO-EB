"""LCDM: cold dark matter and a cosmological constant, with or without massive neutrinos."""

from ..gravity import AdiabaticMode, Background, Curvature, Metric
from ..interactions import ThomsonScattering
from ..species import (
    Baryons,
    ColdDarkMatter,
    CosmologicalConstant,
    MassiveNeutrinos,
    MasslessNeutrinos,
    Photons,
    RadiationStreaming,
)
from .base import CosmologyModel

LCDM = CosmologyModel(
    "LCDM",
    (
        Background,
        RadiationStreaming,
        AdiabaticMode,
        Curvature,
        Metric,
        ColdDarkMatter,
        Baryons,
        Photons,
        MasslessNeutrinos,
        CosmologicalConstant,
        ThomsonScattering,
    ),
)
"""LCDM with massless neutrinos: CDM, baryons, photons coupled by Thomson
scattering, ``Neff`` massless neutrino flavours and a cosmological constant.
Flat unless the cosmology sets ``Omegak``."""

LCDM_MNU = LCDM.add(MassiveNeutrinos, name="LCDM+mnu")
"""LCDM with ``Nmnu`` massive neutrino flavours of mass ``mnu`` on ``NQMAX``
momentum bins."""
