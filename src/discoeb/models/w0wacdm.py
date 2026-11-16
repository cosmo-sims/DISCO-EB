"""w0waCDM: a CPL dark-energy fluid, with massless or massive neutrinos."""

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
    DarkEnergy,
    MassiveNeutrinos,
    MasslessNeutrinos,
    Photons,
    RadiationStreaming,
)
from ._model import Model

__all__ = ["W0WACDM", "W0WACDM_MNU"]


class W0WACDM(
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
    DarkEnergy,
    ThomsonScattering,
    name="w0waCDM",
):
    """LCDM with a perturbed dark-energy fluid in place of the cosmological constant.

    Its equation of state is ``w = w_DE_0 + w_DE_a (1 - a)`` and its sound
    speed ``cs2_DE``. Needs ``w != -1``.
    """


class W0WACDM_MNU(W0WACDM, MassiveNeutrinos, name="w0waCDM+mnu"):
    """w0waCDM with massive neutrinos."""
