"""
This module defines comsological models. Each model is built by inheriting
from :class:`Model` and base components.
"""

from ._model import Model, ModelSpec
from .lcdm import LCDM, LCDM_MNU
from .w0wacdm import W0WACDM, W0WACDM_MNU

__all__ = [
    "LCDM",
    "LCDM_MNU",
    "W0WACDM",
    "W0WACDM_MNU",
    "Model",
    "ModelSpec",
]
