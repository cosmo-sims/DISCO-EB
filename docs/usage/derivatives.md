# Computing derivatives

Pass a parameter as a JAX value and the kept values are differentiable in
it:

```python
import jax
import jax.numpy as jnp

def R_today(omega_c):
    cosmology = planck.replace(omega_c=omega_c)
    return solve_eb(cosmology, k, [1.0], keep=("R",))[0, 0, :, -1]

dR = jax.jacfwd(R_today)(jnp.asarray(planck.omega_c))   # (64,)
```

`jax.jvp` and `jax.jacfwd` each cost one launch, with or without `jit`. The
kernel integrates a sensitivity column beside the state for each direction:
one for `jax.jvp`, one per parameter for `jax.jacfwd`. The derivative holds
the scale factor fixed, so it is the derivative at fixed redshift.

Every parameter of a cosmology is differentiable, including those of
`from_density_parameters`:

```python
def pk(Omega_m):
    cosmology = LCDM.from_density_parameters(Omega_m=Omega_m, Omega_b=0.0494, H_0=67.32, ...)
    return power_spectrum(cosmology)

dP_dOmega_m = jax.jacfwd(pk)(jnp.asarray(0.3142))
```

## Forward mode only

`jax.grad`, `jax.vjp`, `jax.jacrev` and `jax.linearize` raise
`NotImplementedError`. For the gradient of a scalar use `jax.jacfwd`, which
costs one column per parameter. `has_aux=True` returns the value from the
same launch:

```python
def loss_and_value(theta):
    value = loss(theta)
    return value, value

gradient, value = jax.jacfwd(loss_and_value, has_aux=True)(theta)
```

## Batches

`jax.jacfwd` of a function of a batch treats every cosmology's parameter as
a direction for every trajectory. Map over the cosmologies outside the
derivative instead:

```python
jacobians = jax.vmap(jax.jacfwd(f))(theta)   # f takes one cosmology's parameters
```

## Memory and `split_derivatives`

By default a trajectory and its sensitivity columns share a GPU thread,
which costs the least work and suits a batch that fills the device. A
thread's memory grows with its columns. `solve_eb(...,
split_derivatives=True)` gives each direction a copy of every trajectory in
a thread of its own. The copies repeat the plain solve but run side by
side, so one cosmology finishes sooner.

## Inputs computed from parameters

{py:func}`~discoeb.solve_inputs` returns the columns of the parameter row
that the kernel reads. They include the quantities the model computes from
parameters before the solve, such as `omega_g` from `T_cmb`.
`solve_eb(..., inputs={"omega_g": value})` overrides a column, which makes
the solution a function of it:

```python
from discoeb import solve_inputs

omega_g = solve_inputs(planck)["omega_g"]
f = lambda x: solve_eb(planck, k, [1.0], keep=("delta_c",), inputs={"omega_g": x})
d_delta_c = jax.jacfwd(f)(omega_g)
```

An override changes the parameter row only. The initial state stays that of
the cosmology.

## Checking against finite differences

The solve is not smooth in the parameters at very small steps, so a central
difference needs a step well above the solver's noise. The {doc}`derivative example <../examples/matter_power_derivative>` makes the
comparison.
