# Keeping quantities

`solve_eb` records only what you name in `keep`. A name can be a state element, such as `clxc`, `theta_2` or `psi_0_1`, which `state_names(cosmology)` lists. It can also be any quantity the model provides, such as the comoving curvature perturbation `R`, the shear `sigma`, the expansion rate `adotoa` or the Thomson `opacity`. The recombination block adds `xe`, `Tmat`, `cs2` and the optical depth `kappa`.

The device holds `n_cosmologies * n_k * n_tau * len(keep)` numbers and never the full state history.

## Derivatives

Give `Omegam`, `Omegab`, `H0`, `Tcmb` or `Neff` as JAX values and the kept values become differentiable. `jax.jvp`, `jax.jacfwd` and `jax.grad` each cost one joint launch, which forms the derivatives inside the kernel by forward sensitivities:

```python
import jax.numpy as jnp

def R_today(omega_m):
    param = {**cosmology, "Omegam": omega_m}
    return solve_eb(param, k, tau, keep=("R",))["R"][0, :, -1]

dR = jax.jacfwd(R_today)(jnp.asarray(cosmology["Omegam"]))  # (64,)
```

The derivative holds the conformal age and the reionization redshift fixed. The other parameters must be plain numbers.

## Extending the model

The equations are components in `discoeb.eb_model`, and `solve_eb(..., extra=...)` appends your own. A stateless `Component` subclass declares providers whose parameter names are the quantities they read. DISCO-CMB defines its line-of-sight sources this way, so they compile into the same kernel.
