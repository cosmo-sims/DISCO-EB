# Solving a batch

{py:func}`~discoeb.solve_eb` solves one cosmology or a list of cosmologies
of one model:

```python
import numpy as np
from discoeb import solve_eb

k = np.geomspace(1e-4, 1.0, 64)      # 1/Mpc
a = np.geomspace(1e-6, 1.0, 100)     # scale factors, the last one today

values = solve_eb(
    [planck, planck.replace(omega_c=0.125)],
    k,
    a,
    keep=("R", "delta_c"),
)
# values: (2, 2, 64, 100) = (cosmology, name, k, a)
```

Every (wave mode, cosmology) pair will be integrated in a single GPU launch. The solve
runs in coordinates of the scale factor from `discoeb.A_START` to 1, where `a = 1` is today
for every cosmology.

## Choosing what the solve should return

To save memory, `solve_eb` will return only the quantities named in `keep`, such as:

| Name | Quantity |
|---|---|
| `delta_c`, `delta_b`, `theta_b` | overdensities and velocity divergences |
| `F_0` ... `F_15`, `E_2` ..., `N_0` ..., `psi_0_1` | multipoles of the photon, polarization and neutrino hierarchies |
| `R`, `alpha`, `eta`, `h_prime` | the comoving curvature perturbation and the metric quantities |
| `aH`, `tau` | the conformal expansion rate and the conformal time |
| `xe`, `Tmat`, `cs2`, `opacity`, `kappa` | the recombination history |

Any method defined on a component is a valid return quantity. The solve returns an array of shape 
`n_cosmologies * len(keep) * n_k * n_a` numbers. You can use `jnp.unstack(values, axis=1)` to split the result by name.

## The matter power spectrum

{py:func}`~discoeb.matter_power_spectrum` computes $P(k)$ from the two matter contrasts:

```python
import jax
import jax.numpy as jnp
from discoeb import matter_power_spectrum

@jax.jit
def solve_matter_power_spectrum(cosmology):
    values = solve_eb(
        cosmology,
        k,
        [1.0],                        # today
        keep=("delta_c", "delta_b"),
        thermo_weight=0.1,            # a looser recombination history suffices
    )
    delta_c, delta_b = jnp.unstack(values[0, :, :, -1])
    return matter_power_spectrum(
        k,
        delta_c=delta_c,
        delta_b=delta_b,
        omega_c=cosmology.omega_c,
        omega_b=cosmology.omega_b,
        A_s=cosmology.A_s,
        n_s=cosmology.n_s,
        k_p=cosmology.k_p,
    )
```

For a batch, solve the list of cosmologies and map `matter_power_spectrum` over the
cosmologies with `jax.vmap`, as the {doc}`notebook example <../examples/batched_power_spectra>` does.

## Tolerances

`rtol` and `atol` are the tolerances of the Rodas5P step controller.
`error_weights={"F_2": 10.0}` weights named states in its error norm, and a
weight can be a function of the wave numbers. `thermo_weight` weights the
error of the recombination block relative to the perturbations.

## Custom quantities

You can add functions to derive new quantities from the existing ones on the fly as part of the solve. Pass these functions via the `extra=` argument to the `solve_eb` function. The functions will be compiled into the same kernel and become available as return values via the `keep` argument. This is how [DISCO-CMB](https://github.com/lawrenceberry/DISCO-CMB) computes its
line-of-sight source functions.
