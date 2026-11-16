"""
This module defines comsological models. Each model is built by inheriting
from :class:`Model` and base components.
"""

from ._cosmology import DIFFERENTIABLE_PARAMETERS, as_cosmology
from ._model import Model, ModelSpec
from .lcdm import LCDM, LCDM_MNU
from .w0wacdm import W0WACDM, W0WACDM_MNU

__all__ = [
    "DIFFERENTIABLE_PARAMETERS",
    "LCDM",
    "LCDM_MNU",
    "W0WACDM",
    "W0WACDM_MNU",
    "Model",
    "ModelSpec",
    "as_cosmology",
]
