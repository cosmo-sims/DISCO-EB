# Quickstart

```{include} ../README.md
:start-after: <!-- start:overview -->
:end-before: <!-- end:overview -->
```

## Installation

```{include} ../README.md
:start-after: <!-- start:install -->
:end-before: <!-- end:install -->
```

## Your first solve

```{include} ../README.md
:start-after: <!-- start:quickstart -->
:end-before: <!-- end:quickstart -->
```

The first solve of a model compiles its kernel. Later solves load it from a
cache on disk.

## Where to go next?

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Building models
:link: usage/models
:link-type: doc

A model is a class whose bases are its components.
:::

:::{grid-item-card} Solving a batch
:link: usage/solving
:link-type: doc

`solve_eb`, its arguments and the quantities it can keep.
:::

:::{grid-item-card} Computing derivatives
:link: usage/derivatives
:link-type: doc

`jax.jvp` and `jax.jacfwd` in any cosmological parameter.
:::

:::{grid-item-card} Examples
:link: examples/flat_lcdm_power_spectrum
:link-type: doc

Notebooks, from a first power spectrum to a Fisher forecast.
:::

::::

```{toctree}
:hidden:

Quickstart <self>
motivations
design
development
acknowledgement
```

```{toctree}
:hidden:
:caption: Usage

usage/models
usage/cosmologies
usage/solving
usage/derivatives
usage/approximations
```

```{toctree}
:hidden:
:caption: Examples

examples/flat_lcdm_power_spectrum
examples/matter_power_derivative
examples/power_spectrum_jacobian
examples/compare_with_class
examples/correlation_function_multipoles
examples/redshift_space_power_response
examples/fisher_forecast
examples/setup_n_body_ics
```

```{toctree}
:hidden:
:caption: API Reference

api/components
api/models
api/solver
api/observables
```
