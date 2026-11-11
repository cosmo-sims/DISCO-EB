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

Pass a list of dictionaries to solve a batch of cosmologies in the same launch. A time past a cosmology's conformal age holds its values today, which `solution.tau0` reports.

## Where to go next?

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Keeping quantities
:link: usage
:link-type: doc

What `keep` accepts, and how to take derivatives.
:::

:::{grid-item-card} API reference
:link: api
:link-type: doc

Everything the `discoeb` package exports.
:::

::::

```{toctree}
:hidden:

Quickstart <self>
usage
development
acknowledgement
```

```{toctree}
:hidden:
:caption: API Reference

api
```
