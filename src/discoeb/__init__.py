"""DISCO-EB's public API."""

# First: double precision and JAX's GPU memory policy, before anything builds
# an array or touches the device.
from . import _jax_config  # noqa: F401
from .components import Component
from .components.recombination import THERMO_ERROR_WEIGHT
from .models import (
    DIFFERENTIABLE_PARAMETERS,
    LCDM,
    LCDM_MNU,
    W0WACDM,
    W0WACDM_MNU,
    Model,
    ModelSpec,
    as_cosmology,
)
from .observables import matter_power_spectrum
from .solver import (
    DIRECTED_NAMES,
    PERTURB_ATOL,
    PERTURB_FIRST_STEP,
    PERTURB_MAX_STEPS,
    PERTURB_RTOL,
    TAU_START,
    background_history,
    conformal_ages,
    density_coefficients,
    differentiable_parameters,
    directed_inputs,
    joint_model,
    solve_eb,
    state_names,
)

__all__ = [
    "Component",
    "Model",
    "ModelSpec",
    "DIFFERENTIABLE_PARAMETERS",
    "DIRECTED_NAMES",
    "PERTURB_ATOL",
    "PERTURB_FIRST_STEP",
    "PERTURB_MAX_STEPS",
    "PERTURB_RTOL",
    "LCDM",
    "LCDM_MNU",
    "W0WACDM",
    "W0WACDM_MNU",
    "TAU_START",
    "THERMO_ERROR_WEIGHT",
    "as_cosmology",
    "background_history",
    "conformal_ages",
    "density_coefficients",
    "differentiable_parameters",
    "directed_inputs",
    "joint_model",
    "matter_power_spectrum",
    "solve_eb",
    "state_names",
]
