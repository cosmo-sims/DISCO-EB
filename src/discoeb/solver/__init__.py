"""The solver. It holds the joint kernel and ``solve_eb``."""

from ._kernel import (
    A_START,
    PERTURB_ATOL,
    PERTURB_FIRST_STEP,
    PERTURB_MAX_STEPS,
    PERTURB_RTOL,
    joint_model,
)
from ._solver import (
    solve_eb,
    solve_inputs,
    state_names,
)

__all__ = [
    "A_START",
    "PERTURB_ATOL",
    "PERTURB_FIRST_STEP",
    "PERTURB_MAX_STEPS",
    "PERTURB_RTOL",
    "joint_model",
    "solve_eb",
    "solve_inputs",
    "state_names",
]
