"""LCDM, with massless or massive neutrinos."""

from __future__ import annotations

from ..components.gravity import (
    AdiabaticMode,
    Background,
    Curvature,
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

    CDM, baryons, photons coupled by Thomson scattering, ``N_eff`` massless
    neutrino flavours and a cosmological constant. Flat unless ``Omega_k`` is
    given.
    """


class LCDM_MNU(LCDM, MassiveNeutrinos, name="LCDM+mnu"):
    """LCDM with ``N_mnu`` massive neutrino flavours of mass ``m_nu`` on ``NQMAX`` momentum bins."""
