"""Cosmological models. A model is a class whose bases are its components, and a cosmology is an instance of it."""

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
