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

One GPU launch integrates every (wave mode, cosmology) pair. The solve
runs in the scale factor from `discoeb.A_START` to 1, so `a = 1` is today
for every cosmology. `a` must increase strictly. `k` must hold plain
numbers, since it sets the shape of the launch.

## Choosing what to keep

`solve_eb` records only the quantities named in `keep`:

| Name | Quantity |
|---|---|
| `delta_c`, `delta_b`, `theta_b`, `eta` | states, which {py:func}`~discoeb.state_names` lists for a model |
| `F_0` ... `F_15`, `E_2` ..., `N_0` ..., `psi_0_1` | multipoles of the photon, polarization and neutrino hierarchies |
| `R`, `alpha`, `h_prime` | the comoving curvature perturbation and the metric quantities |
| `aH`, `tau` | the conformal expansion rate and the conformal time |
| `xe`, `Tmat`, `cs2`, `opacity`, `kappa` | the recombination history |

Any quantity a component of the model provides is a valid name. The value
of `tau` today is the conformal age. The device stores
`n_cosmologies * len(keep) * n_k * n_a` numbers and never the state
history.

`jnp.unstack(values, axis=1)` splits the result by name.

## The matter power spectrum

{py:func}`~discoeb.matter_power_spectrum` turns the two matter contrasts of
one cosmology into $P(k)$. Wrapped in `jax.jit`, one compiled function
serves every cosmology of the model:

```python
import jax
import jax.numpy as jnp
from discoeb import matter_power_spectrum

@jax.jit
def power_spectrum(cosmology):
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

For a batch, solve the list and map `matter_power_spectrum` over the
cosmologies with `jax.vmap`, as the {doc}`first example <../examples/flat_lcdm_power_spectrum>` does.

## Tolerances

`rtol` and `atol` are the tolerances of the Rodas5P step controller.
`error_weights={"F_2": 10.0}` weights named states in its error norm, and a
weight can be a function of the wave numbers. `thermo_weight` weights the
recombination block.

## Quantities of your own

`extra=` appends stateless components to the model for one solve. Their
methods read the model's quantities and provide new names for `keep`, which
compile into the same kernel.
[DISCO-CMB](https://github.com/lawrenceberry/DISCO-CMB) defines its
line-of-sight sources this way.
