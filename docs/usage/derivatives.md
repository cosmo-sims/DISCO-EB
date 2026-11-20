# Computing derivatives

DISCO-EB is fully compatible with `jax.jvp` and `jax.jacfwd`:

```python
import jax
import jax.numpy as jnp

def R_today(omega_c):
    cosmology = planck.replace(omega_c=omega_c)
    return solve_eb(cosmology, k, [1.0], keep=("R",))[0, 0, :, -1]

dR = jax.jacfwd(R_today)(jnp.asarray(planck.omega_c))   # (64,)
```

The GPU kernel integrates a sensitivity column beside the state for each direction:
one for `jax.jvp`, one per parameter for `jax.jacfwd`.

Every parameter of a cosmology is differentiable, including those of
`from_density_parameters`:

```python
def pk(Omega_m):
    cosmology = LCDM.from_density_parameters(Omega_m=Omega_m, Omega_b=0.0494, H_0=67.32, ...)
    return power_spectrum(cosmology)

dP_dOmega_m = jax.jacfwd(pk)(jnp.asarray(0.3142))
```

`jax.grad`, `jax.vjp`, `jax.jacrev` and `jax.linearize` raise
`NotImplementedError`. For the gradient of a scalar use `jax.jacfwd`. `has_aux=True` can be used to return the value alongside the gradient in the same GPU launch:

```python
def loss_and_value(theta):
    value = loss(theta)
    return value, value

gradient, value = jax.jacfwd(loss_and_value, has_aux=True)(theta)
```

To compute gradients for many comsologies, wrap `jax.jacfwd` in `jax.vmap` over the batch of parameters:

```python
jacobians = jax.vmap(jax.jacfwd(f))(theta)
```

## Splitting directional derivative work over threads

By default a trajectory and its sensitivity columns share a GPU thread,
which costs the least work and suits a batch that is large enought to saturate the GPU. However
thread's memory requirement grows with the number of directions being differentiated along. `solve_eb(...,
split_derivatives=True)` allows us to split that work over separate GPU threads by assigning each direction to be solved in its own thread. This is strictly less efficient, since each copy must repeat the plain solve, but it can be faster for smaller batches of cosmologies that otherwise don't fill the GPU.
