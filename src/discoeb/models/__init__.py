"""Cosmological models: classes whose bases are components, and their cosmologies.

Each built-in model has a module of its own (:mod:`~discoeb.models.lcdm`,
:mod:`~discoeb.models.w0wacdm`); the base classes are in ``_model``.
"""

from ._cosmology import DIFFERENTIABLE_PARAMETERS, as_cosmology
from ._model import Model, ModelSpec
from .lcdm import LCDM, LCDM_MNU
from .w0wacdm import W0WACDM, W0WACDM_MNU

DEFAULT_MODEL = W0WACDM_MNU
"""The model a bare parameter dictionary stands for where nothing names one.

:func:`~discoeb.solver._background.density_coefficients`,
:func:`~discoeb.solver._background.background_history`,
:func:`~discoeb.solver.conformal_ages` and
:func:`~discoeb.solver.directed_inputs` take a cosmology, or a dictionary
for this model. Its components declare every built-in parameter, and on a
cosmology another built-in model accepts, its background is that model's. A
dictionary that sets a parameter of a component outside it raises there
(:meth:`ModelSpec.check_keys`)."""

DEFAULT_SPEC = DEFAULT_MODEL.spec
"""The :class:`ModelSpec` of :data:`DEFAULT_MODEL`."""

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_SPEC",
    "DIFFERENTIABLE_PARAMETERS",
    "LCDM",
    "LCDM_MNU",
    "W0WACDM",
    "W0WACDM_MNU",
    "Model",
    "ModelSpec",
    "as_cosmology",
]
