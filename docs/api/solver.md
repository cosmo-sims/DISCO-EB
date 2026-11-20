# Solver

```{eval-rst}
.. automodule:: discoeb.solver
   :no-members:
```

```{eval-rst}
.. autofunction:: discoeb.solve_eb

.. autofunction:: discoeb.state_names

.. autofunction:: discoeb.solve_inputs

.. autofunction:: discoeb.joint_model
```

## Defaults

| Constant | Value | Meaning |
|---|---|---|
| `discoeb.A_START` | `2e-7` | the scale factor at which every solve starts |
| `discoeb.PERTURB_RTOL` | `1e-4` | default `rtol` |
| `discoeb.PERTURB_ATOL` | `1e-4` | default `atol` |
| `discoeb.PERTURB_FIRST_STEP` | `A_START` | default first step, in the scale factor |
| `discoeb.PERTURB_MAX_STEPS` | `20000` | default `max_steps` |
| `discoeb.H100` | 100 km/s/Mpc in 1/Mpc | converts a density $\omega$ into a rate |
