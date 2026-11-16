# Examples

Each example is a short notebook for one use of DISCO-EB, in [`examples/`](https://github.com/cosmo-sims/DISCO-EB/tree/main/examples). The pages show the notebooks as they were last run. Nothing runs when the site builds.

They need a GPU. The first run of a model compiles its kernel, which takes from 30 s to a few minutes. Later runs load it from the cache.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Matter power spectra of a batch
:link: flat_lcdm_power_spectrum
:link-type: doc

Start here: $P(k)$ of 128 cosmologies in one jitted function, timed.
:::

:::{grid-item-card} Derivative of the matter power spectrum
:link: matter_power_derivative
:link-type: doc

$\partial P / \partial \Omega_m$ by `jax.jacfwd`, against a finite difference.
:::

:::{grid-item-card} Jacobian of the matter power spectrum
:link: power_spectrum_jacobian
:link-type: doc

The response of $P(k, a)$ to ten parameters, in scale and time.
:::

:::{grid-item-card} Comparison with CLASS
:link: compare_with_class
:link-type: doc

CDM and baryon spectra at redshifts 99 and 0.
:::

:::{grid-item-card} Correlation function multipoles
:link: correlation_function_multipoles
:link-type: doc

Kaiser $\xi_0$, $\xi_2$, $\xi_4$ and their derivatives.
:::

:::{grid-item-card} Response of the redshift-space power spectrum
:link: redshift_space_power_response
:link-type: doc

$P(k, \mu)$ with Kaiser and Alcock-Paczynski, as polar maps.
:::

:::{grid-item-card} Fisher forecast
:link: fisher_forecast
:link-type: doc

A Euclid-like galaxy clustering forecast.
:::

:::{grid-item-card} Initial conditions for an N-body simulation
:link: setup_n_body_ics
:link-type: doc

Differentiable Zel'dovich displacements for DISCO-DJ.
:::
::::

```{toctree}
:hidden:

flat_lcdm_power_spectrum
matter_power_derivative
power_spectrum_jacobian
compare_with_class
correlation_function_multipoles
redshift_space_power_response
fisher_forecast
setup_n_body_ics
```
