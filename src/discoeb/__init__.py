"""DISCO-EB's public API."""

# First: double precision and JAX's GPU memory policy, before anything builds
# an array or touches the device.
from . import _jax_config  # noqa: F401
from .components import Component
from .components.gravity import H100
from .components.recombination import THERMO_ERROR_WEIGHT
from .models import (
    LCDM,
    LCDM_MNU,
    W0WACDM,
    W0WACDM_MNU,
    Model,
)
from .observables import matter_power_spectrum
from .solver import (
    A_START,
    DIRECTED_NAMES,
    PERTURB_ATOL,
    PERTURB_FIRST_STEP,
    PERTURB_MAX_STEPS,
    PERTURB_RTOL,
    differentiable_parameters,
    directed_inputs,
    joint_model,
    solve_eb,
    state_names,
)

__all__ = [
    "Component",
    "Model",
    "DIRECTED_NAMES",
    "A_START",
    "H100",
    "PERTURB_ATOL",
    "PERTURB_FIRST_STEP",
    "PERTURB_MAX_STEPS",
    "PERTURB_RTOL",
    "LCDM",
    "LCDM_MNU",
    "W0WACDM",
    "W0WACDM_MNU",
    "THERMO_ERROR_WEIGHT",
    "differentiable_parameters",
    "directed_inputs",
    "joint_model",
    "matter_power_spectrum",
    "solve_eb",
    "state_names",
]
