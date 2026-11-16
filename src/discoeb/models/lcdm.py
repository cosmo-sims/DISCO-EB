"""LCDM, with massless or massive neutrinos."""

from __future__ import annotations

from ..components.gravity import (
    AdiabaticMode,
    Background,
    Curvature,
    Kinematics,
    Metric,
)
from ..components.interactions import ThomsonScattering
from ..components.species import (
    Baryons,
    ColdDarkMatter,
    CosmologicalConstant,
    MassiveNeutrinos,
    MasslessNeutrinos,
    Photons,
    RadiationStreaming,
)
from ._model import Model

__all__ = ["LCDM", "LCDM_MNU"]


class LCDM(
    Model,
    Kinematics,
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
):
    """LCDM with massless neutrinos.

    CDM, baryons, photons coupled by Thomson scattering, ``Neff`` massless
    neutrino flavours and a cosmological constant. Flat unless ``Omegak`` is
    given.
    """


class LCDM_MNU(LCDM, MassiveNeutrinos, name="LCDM+mnu"):
    """LCDM with ``Nmnu`` massive neutrino flavours of mass ``mnu`` on ``NQMAX`` momentum bins."""
